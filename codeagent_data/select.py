"""Rank tasks for the teacher runs so that the first N *solved* tasks are the most reliable mix.

The goal is N solved tasks (default 2,400), not N attempted ones: the teacher won't solve every
task, so the teacher stage works down one ranked list and stops once N tasks are solved.
This module builds that list and the limits the teacher stage enforces:

1. Hard gates: tasks our verifier handles well and that fit the localizer/fixer method
   (Python, small source patch, clear issue, sane test counts, no doubtful-label warnings).
2. Order: sources are interleaved by weight (smooth weighted round-robin), so every prefix of
   the list mixes sources in those proportions; within a source, best quality score first.
   Default weights follow verification fidelity: SWE-bench > SWE-Gym > R2E-Gym > SWE-smith.
3. Limits on *solved* tasks (enforced while running, see `should_attempt`): at most
   `max_share` of the N solved tasks per source and `max_per_repo` per repo. When a source runs
   out of tasks the others simply continue, so the target is still reached.

Each task gets a cross-fitting fold (A/B), alternating within every (source, repo) in rank
order, so any prefix is balanced too.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jsonl import iter_jsonl, read_jsonl
from .patches import is_protected_config, is_test_path, parse_patch

# most reliable first
RELIABILITY_ORDER = ["swebench", "swe_gym", "r2e_gym", "swe_smith", "promax"]
# interleaving weights: the mix of attempts (and roughly of solved tasks)
DEFAULT_SHARES = {"swebench": 0.40, "swe_gym": 0.35, "r2e_gym": 0.15, "swe_smith": 0.10}
# hard ceilings on the share of SOLVED tasks per source (synthetic-issue sources kept smaller)
DEFAULT_MAX_SHARE = {"swebench": 0.50, "swe_gym": 0.50, "r2e_gym": 0.35, "swe_smith": 0.20}
# warnings that make the gold label or its tests doubtful
BLOCKING_CODES = {"W_GOLD_TOUCHES_PROTECTED", "W_F2P_NOT_IN_TEST_PATCH", "W_PATCH_REPO_SUSPECT",
                  "I_LARGE_PATCH", "W_VALIDATE_UNSUPPORTED"}


@dataclass
class Gates:
    min_py_files: int = 1
    max_py_files: int = 3
    max_other_src_files: int = 2       # docs / changelog files next to the code change
    min_changed_lines: int = 2
    max_changed_lines: int = 150
    min_f2p: int = 1
    max_f2p: int = 30
    max_r2e_tests: int = 200
    min_ps_chars: int = 150
    max_ps_chars: int = 12_000         # ~3k tokens: leaves room in Gemma's 32k window
    max_leak_ratio: float = 0.3


@dataclass
class Candidate:
    uid: str
    source: str  # verifier family
    repo: str
    image: str
    score: float
    feats: dict[str, Any] = field(default_factory=dict)


def _features(r: dict[str, Any]) -> dict[str, Any]:
    gold = parse_patch(r.get("gold_patch") or "")
    src = [f for f in gold.files if f.path and not is_test_path(f.path) and not is_protected_config(f.path)]
    py = [f for f in src if f.path.endswith(".py")]
    extra = r.get("extra") or {}
    return {
        "py_files": len(py),
        "other_src_files": len(src) - len(py),
        "changed_lines": sum(f.added + f.removed for f in src),
        "n_f2p": len(r.get("fail_to_pass") or []),
        "n_p2p": len(r.get("pass_to_pass") or []),
        "n_tests": extra.get("n_tests"),
        "ps_chars": len(r.get("problem_statement") or ""),
        "leak_ratio": float(extra.get("leak_ratio") or 0.0),
        "language": (r.get("language") or "python").lower(),
        "codes": {i["code"] for i in r.get("issues", [])},
    }


def gate(r: dict[str, Any], f: dict[str, Any], g: Gates) -> str | None:
    """Reason the row is excluded, or None if it passes every gate."""
    if not r.get("ok"):
        return "rejected by ingest"
    if r.get("role") != "train":
        return "held out (evaluation set)"
    if f["language"] != "python":
        return "not Python"
    if f["codes"] & BLOCKING_CODES:
        return "doubtful label (" + ", ".join(sorted(f["codes"] & BLOCKING_CODES)) + ")"
    if not g.min_py_files <= f["py_files"] <= g.max_py_files:
        return f"Python source files edited not in {g.min_py_files}..{g.max_py_files}"
    if f["other_src_files"] > g.max_other_src_files:
        return "too many non-Python source files"
    if not g.min_changed_lines <= f["changed_lines"] <= g.max_changed_lines:
        return f"changed lines not in {g.min_changed_lines}..{g.max_changed_lines}"
    if r.get("verifier") == "r2e_gym":
        n = f["n_tests"]
        if not n or (isinstance(n, float) and math.isnan(n)) or n > g.max_r2e_tests:
            return "R2E test count missing or too large"
    elif not g.min_f2p <= f["n_f2p"] <= g.max_f2p:
        return f"FAIL_TO_PASS count not in {g.min_f2p}..{g.max_f2p}"
    if not g.min_ps_chars <= f["ps_chars"] <= g.max_ps_chars:
        return f"problem statement length not in {g.min_ps_chars}..{g.max_ps_chars} chars"
    if f["leak_ratio"] > g.max_leak_ratio:
        return "issue text leaks the fix"
    return None


def _score(r: dict[str, Any], f: dict[str, Any]) -> float:
    s = (r.get("extra") or {}).get("score")
    base = float(s) if isinstance(s, (int, float)) and not math.isnan(s) else 50.0
    return base + (3.0 if f["n_p2p"] > 0 else 0.0)  # regression tests make verification stronger


def interleave(cands: dict[str, list[Candidate]], weights: dict[str, float]) -> list[Candidate]:
    """Smooth weighted round-robin: every prefix holds sources in proportion to their weights
    (among sources that still have tasks); each source keeps its own best-first order."""
    queues = {s: list(cands.get(s, [])) for s in weights if cands.get(s)}
    credit = {s: 0.0 for s in queues}
    out: list[Candidate] = []
    while queues:
        total = sum(weights[s] for s in queues)
        for s in queues:
            credit[s] += weights[s]
        src = max(queues, key=lambda s: (credit[s], -RELIABILITY_ORDER.index(s) if s in RELIABILITY_ORDER else -99))
        credit[src] -= total
        out.append(queues[src].pop(0))
        if not queues[src]:
            del queues[src], credit[src]
    return out


def should_attempt(task: dict[str, Any], solved_by_source: Counter, solved_by_repo: Counter,
                   targets: dict[str, Any]) -> bool:
    """Teacher-stage rule: skip a task whose source or repo already has its share of solved tasks."""
    n = targets["solved_target"]
    if sum(solved_by_source.values()) >= n:
        return False
    src = task.get("verifier") or ""
    cap = targets["max_share"].get(src)
    if cap is not None and solved_by_source[src] >= math.floor(cap * n):
        return False
    return solved_by_repo[(task.get("repo") or "").lower()] < targets["max_per_repo"]


def rank_tasks(normalized: list[dict[str, Any]], tasks_by_uid: dict[str, dict[str, Any]], *, n: int = 2400,
               shares: dict[str, float] | None = None, max_share: dict[str, float] | None = None,
               max_per_repo: int = 120, gates: Gates | None = None) -> dict[str, Any]:
    shares = dict(shares or DEFAULT_SHARES)
    max_share = {s: v for s, v in (max_share or DEFAULT_MAX_SHARE).items() if s in shares}
    gates = gates or Gates()
    funnel: Counter[str] = Counter()
    cands: dict[str, list[Candidate]] = defaultdict(list)
    for r in normalized:
        src = r.get("verifier") or ""
        if r.get("uid") not in tasks_by_uid:
            funnel["not in task files"] += 1
            continue
        if src not in shares:
            funnel[f"source not selected ({src or 'unknown'})"] += 1
            continue
        f = _features(r)
        why = gate(r, f, gates)
        if why:
            funnel[why] += 1
            continue
        funnel["passed all gates"] += 1
        cands[src].append(Candidate(r["uid"], src, (r.get("repo") or "").lower(),
                                    (r.get("env") or {}).get("image") or "", _score(r, f), f))
    for lst in cands.values():
        lst.sort(key=lambda c: (-c.score, c.uid))

    order = interleave(cands, shares)
    fold_ct: Counter[tuple[str, str]] = Counter()
    ranked = []
    for i, c in enumerate(order):
        row = dict(tasks_by_uid[c.uid])
        k = (c.source, c.repo)
        row.update(selection_rank=i, selection_score=round(c.score, 3), fold="AB"[fold_ct[k] % 2],
                   reliability_tier=RELIABILITY_ORDER.index(c.source) if c.source in RELIABILITY_ORDER else 99)
        fold_ct[k] += 1
        ranked.append(row)
    targets = {"solved_target": n, "shares": shares, "max_share": max_share, "max_per_repo": max_per_repo}
    return {"ranked": ranked, "targets": targets, "funnel": funnel,
            "candidates": {s: len(cands.get(s, [])) for s in shares}}


def capacity(res: dict[str, Any], solve_rate: float) -> dict[str, int]:
    """Solved tasks each source can contribute at a given (valid x solved) rate, after its caps."""
    t = res["targets"]
    cap_repo: Counter[tuple[str, str]] = Counter()
    for row in res["ranked"]:
        cap_repo[(row.get("verifier"), (row.get("repo") or "").lower())] += 1
    out = {}
    for s in t["shares"]:
        by_repo = sum(min(c * solve_rate, t["max_per_repo"]) for (src, _), c in cap_repo.items() if src == s)
        cap = t["max_share"].get(s)
        out[s] = int(min(by_repo, cap * t["solved_target"]) if cap is not None else by_repo)
    return out


def render_report(res: dict[str, Any]) -> str:
    t, ranked = res["targets"], res["ranked"]
    n = t["solved_target"]
    lines = ["# Selection report", "",
             f"solved target: {n}   ranked candidates: {len(ranked)}", "",
             "## Gates (all rows)", ""]
    lines += [f"- {k}: {v}" for k, v in res["funnel"].most_common()]
    lines += ["", "## Sources", "",
              "| source | weight | max share of solved | candidates | solvable at 40% / 55% / 70% (valid x solved) |",
              "| --- | --- | --- | --- | --- |"]
    caps = {r: capacity(res, r) for r in (0.40, 0.55, 0.70)}
    for s, w in t["shares"].items():
        ms = t["max_share"].get(s)
        lines.append(f"| {s} | {w:.2f} | {'–' if ms is None else f'{ms:.0%} ({math.floor(ms * n)})'} | "
                     f"{res['candidates'].get(s, 0)} | {caps[0.40][s]} / {caps[0.55][s]} / {caps[0.70][s]} |")
    totals = {r: sum(c.values()) for r, c in caps.items()}
    lines += ["", "Total solvable: " + " / ".join(f"{min(v, n)} at {int(r * 100)}%" for r, v in totals.items())
              + f" (target {n}).", ""]
    for r, v in totals.items():
        if v < n:
            lines.append(f"- At {int(r * 100)}% the target is out of reach: relax gates (`--gate`), "
                         "raise `--max-share` / `--max-per-repo`, or add sources.")
    for frac, label in ((0.55, "55%"), (0.40, "40%")):
        need = math.ceil(n / frac)
        head = ranked[:need]
        imgs = {x.get("env_image") for x in head}
        lines.append(f"- At {label}, roughly the first {min(need, len(ranked))} tasks get attempted "
                     f"({len(imgs)} distinct images to pull).")
    head = ranked[: math.ceil(n / 0.55)]
    repos = Counter(x["repo"] for x in head)
    folds = Counter(x["fold"] for x in head)
    lines += ["", f"## First {len(head)} tasks", "",
              "- by source: " + ", ".join(f"{k} {v}" for k, v in Counter(x.get('verifier') for x in head).most_common()),
              f"- folds: A={folds.get('A', 0)} B={folds.get('B', 0)}",
              f"- repos: {len(repos)} distinct; top: " + ", ".join(f"{k} {v}" for k, v in repos.most_common(8))]
    return "\n".join(lines) + "\n"


def run_select(stage0: Path, out: Path, n: int, shares: dict[str, float] | None, max_per_repo: int,
               max_share: dict[str, float] | None = None, gates: Gates | None = None) -> dict[str, Any]:
    normalized = read_jsonl(stage0 / "normalized.jsonl")
    tasks_by_uid: dict[str, dict[str, Any]] = {}
    for fp in sorted((stage0 / "tasks").glob("*.jsonl")):
        for t in iter_jsonl(fp):
            tasks_by_uid[t["uid"]] = t
    res = rank_tasks(normalized, tasks_by_uid, n=n, shares=shares, max_share=max_share,
                     max_per_repo=max_per_repo, gates=gates)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ranked.jsonl").open("w") as f:
        for t in res["ranked"]:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (out / "targets.json").write_text(json.dumps(res["targets"], indent=2))
    report = render_report(res)
    (out / "selection_report.md").write_text(report)
    print(report)
    return res
