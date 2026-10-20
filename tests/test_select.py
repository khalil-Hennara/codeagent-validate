"""Selection tests. Run: python3 tests/test_select.py"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codeagent_data.select import rank_tasks, should_attempt  # noqa: E402


def patch(n_files=1, lines=4, ext="py"):
    out = ""
    for i in range(n_files):
        body = "".join(f"-a{j}\n+b{j}\n" for j in range(lines // 2))
        out += f"diff --git a/pkg/m{i}.{ext} b/pkg/m{i}.{ext}\n--- a/pkg/m{i}.{ext}\n+++ b/pkg/m{i}.{ext}\n@@ -1,{lines//2} +1,{lines//2} @@\n{body}"
    return out


def make(n_per_source, seed=0):
    rnd = random.Random(seed)
    norm, tasks = [], {}
    repos = {"swebench": ["django/django", "sympy/sympy", "psf/requests"], "swe_gym": ["pandas-dev/pandas", "python/mypy"],
             "r2e_gym": ["numpy/numpy"], "swe_smith": ["oauthlib/oauthlib", "encode/starlette"]}
    for src, k in n_per_source.items():
        for i in range(k):
            uid = f"{src}::{i}"
            r = dict(uid=uid, ok=True, role="train", verifier=src, repo=rnd.choice(repos[src]), language="python",
                     gold_patch=patch(rnd.choice([1, 1, 2, 5]), rnd.choice([4, 10, 400])),
                     fail_to_pass=["t"] * rnd.choice([1, 2, 40]), pass_to_pass=["p"],
                     problem_statement="x" * rnd.choice([100, 800, 3000]), issues=[],
                     env={"image": f"img/{src}:{i % 50}"},
                     extra={"score": rnd.uniform(40, 100), "leak_ratio": rnd.choice([0.0, 0.0, 0.5]), "n_tests": 10})
            norm.append(r)
            tasks[uid] = {"uid": uid, "instance_id": uid, "repo": r["repo"], "verifier": src, "env_image": r["env"]["image"]}
    return norm, tasks


def simulate(ranked, targets, solve_p, seed=1):
    """What the teacher stage will do: walk the list, skip capped sources/repos, stop at N solved."""
    import random as _r
    rnd = _r.Random(seed)
    by_src, by_repo, attempts = Counter(), Counter(), 0
    for t in ranked:
        if sum(by_src.values()) >= targets["solved_target"]:
            break
        if not should_attempt(t, by_src, by_repo, targets):
            continue
        attempts += 1
        if rnd.random() < solve_p.get(t["verifier"], 0.5):
            by_src[t["verifier"]] += 1
            by_repo[t["repo"].lower()] += 1
    return by_src, by_repo, attempts


def test_prefix_mix_folds_and_gates():
    norm, tasks = make({"swebench": 3000, "swe_gym": 3000, "r2e_gym": 3000, "swe_smith": 3000})
    norm[0]["role"] = "heldout_eval"
    norm[1]["issues"] = [{"code": "W_GOLD_TOUCHES_PROTECTED"}]
    res = rank_tasks(norm, tasks, n=400)
    ranked = res["ranked"]
    uids = [t["uid"] for t in ranked]
    assert len(uids) == len(set(uids)) and "swebench::0" not in uids and "swebench::1" not in uids
    assert [t["selection_rank"] for t in ranked] == list(range(len(ranked)))
    head = Counter(t["verifier"] for t in ranked[:200])
    assert head == Counter({"swebench": 80, "swe_gym": 70, "r2e_gym": 30, "swe_smith": 20}), head
    folds = Counter(t["fold"] for t in ranked[:200])
    assert abs(folds["A"] - folds["B"]) <= 10, folds
    for src in ("swebench", "swe_gym"):  # best score first within each source
        sc = [t["selection_score"] for t in ranked if t["verifier"] == src]
        assert sc == sorted(sc, reverse=True)


def test_simulated_teacher_reaches_target_within_caps():
    norm, tasks = make({"swebench": 3000, "swe_gym": 3000, "r2e_gym": 3000, "swe_smith": 3000})
    res = rank_tasks(norm, tasks, n=400, max_per_repo=120)
    t = res["targets"]
    by_src, by_repo, attempts = simulate(res["ranked"], t, {"swebench": 0.6, "swe_gym": 0.4, "r2e_gym": 0.6, "swe_smith": 0.7})
    assert sum(by_src.values()) == 400
    for s, cap in t["max_share"].items():
        assert by_src[s] <= int(cap * 400), (s, by_src)
    assert max(by_repo.values()) <= 120
    assert attempts < len(res["ranked"])


def test_source_running_out_is_absorbed_by_others():
    norm, tasks = make({"swebench": 300, "swe_gym": 3000, "r2e_gym": 3000, "swe_smith": 3000})
    res = rank_tasks(norm, tasks, n=400, max_per_repo=1000)
    by_src, _, _ = simulate(res["ranked"], res["targets"], {"swebench": 0.5, "swe_gym": 0.5, "r2e_gym": 0.5, "swe_smith": 0.5})
    assert sum(by_src.values()) == 400 and by_src["swebench"] < 160, by_src


def test_cli_end_to_end():
    norm, tasks = make({"swebench": 600, "swe_gym": 600, "r2e_gym": 600, "swe_smith": 600})
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "s0/tasks").mkdir(parents=True)
        (d / "s0/normalized.jsonl").write_text("\n".join(json.dumps(r) for r in norm))
        for src in ("swebench", "swe_gym", "r2e_gym", "swe_smith"):
            (d / f"s0/tasks/{src}.jsonl").write_text("\n".join(json.dumps(t) for t in tasks.values() if t["verifier"] == src))
        from codeagent_data.cli import main
        main(["select", "--stage0", str(d / "s0"), "--out", str(d / "sel"), "--n", "100", "--gate", "max_f2p=60"])
        assert (d / "sel/ranked.jsonl").exists() and json.loads((d / "sel/targets.json").read_text())["solved_target"] == 100
        assert "solvable at" in (d / "sel/selection_report.md").read_text()


def test_unicode_line_separators_survive_ingest_and_select():
    """U+2028 / U+0085 / form feed inside text must not split JSONL records or diff lines."""
    import subprocess
    from codeagent_data.cli import main
    from codeagent_data.patches import parse_patch

    weird = "Bug\u2028with\x85odd\x0cseparators and enough words to pass the length gate. " * 3
    diff = ("diff --git a/pkg/m.py b/pkg/m.py\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,2 +1,2 @@\n"
            " \x0c# page break\n-a = 1\n+a = 2  # \u2028 odd\n")
    pp = parse_patch(diff)
    assert not pp.malformed_lines and pp.files[0].added == 1 and pp.files[0].removed == 1
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        row = dict(uid="u1", source="swe_gym", verifier="swe_gym", instance_id="acme__widget-1", repo="acme/widget",
                   base_commit="abc", problem_statement=weird, patch=diff,
                   test_patch="diff --git a/tests/test_m.py b/tests/test_m.py\n--- a/tests/test_m.py\n+++ b/tests/test_m.py\n@@ -1 +1,2 @@\n x\n+def test_a(): pass\n",
                   f2p=["tests/test_m.py::test_a"], p2p=[], env_image="img:1", score=90.0)
        (d / "in.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        main(["ingest", str(d / "in.jsonl"), "--out", str(d / "s0")])
        main(["select", "--stage0", str(d / "s0"), "--out", str(d / "sel"), "--n", "1"])
        ranked = [json.loads(l) for l in (d / "sel/ranked.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
        assert len(ranked) == 1 and ranked[0]["problem_statement"] == weird


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
