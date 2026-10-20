"""Verdict logic for stage 1 (independent of Docker, so it is unit-testable)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..jsonl import iter_jsonl
from .specs import EvalPlan, UnsupportedVerifier, build_plan, r2e_normalize

PASS = {"PASSED", "XFAIL"}
# verdicts caused by infrastructure, not by the task: a re-run tries these tasks again
RETRYABLE = ("IMAGE_UNAVAILABLE", "INFRA_ERROR")


def is_final(verdict: dict[str, Any]) -> bool:
    return not str(verdict.get("reason", "")).startswith(RETRYABLE)
_TEST_PATCH_ERR = re.compile(r"^(error: (patch failed|corrupt patch|.*does not apply|.*No such file)).*$", re.M)


def _status(case: str, sm: dict[str, str]) -> str:
    # exact match, same as swebench.harness.grading.test_passed
    return sm.get(case, "MISSING")


@dataclass
class Verdict:
    instance_id: str
    valid: bool
    reason: str
    route: str = ""
    f2p_fail_on_base: int = 0
    f2p_total: int = 0
    f2p_pass_on_base: list[str] = field(default_factory=list)
    f2p_fail_on_gold: list[str] = field(default_factory=list)
    p2p_fail_on_gold: list[str] = field(default_factory=list)
    p2p_fail_on_base: int = 0
    flaky: list[str] = field(default_factory=list)
    strict_file_pass: bool | None = None  # all parsed tests pass on gold (competition-style whole-file rule)
    n_parsed_base: int = 0
    n_parsed_gold: int = 0
    seconds: float = 0.0
    mode: str = "lists"  # lists (F2P/P2P given) | expected (R2E-Gym) | derive (no F2P list)
    derived_f2p: list[str] = field(default_factory=list)  # filled in expected/derive modes
    derived_p2p: list[str] = field(default_factory=list)


def judge(task: dict[str, Any], plan: EvalPlan, base_log: str, gold_logs: list[str]) -> Verdict:
    f2p, p2p = list(task.get("FAIL_TO_PASS", [])), list(task.get("PASS_TO_PASS", []))
    v = Verdict(task["instance_id"], False, "", route=plan.route, f2p_total=len(f2p))

    for name, log in [("base", base_log)] + [("gold", g) for g in gold_logs]:
        # only the eval script's own output: earlier fallback attempts to apply the gold/setup patch
        # may print "error: ..." even though a later strategy succeeded
        eval_part = log.rsplit(">>>>> Eval script\n", 1)[-1]
        pre = eval_part.split(">>>>> Start Test Output", 1)[0]
        m = _TEST_PATCH_ERR.search(pre)
        if m:
            v.reason = f"TEST_PATCH_APPLY_FAIL ({name}): {m.group(0)[:200]}"
            return v

    sb = plan.parse(base_log)
    golds = [plan.parse(g) for g in gold_logs]
    v.n_parsed_base, v.n_parsed_gold = len(sb), len(golds[0])
    if not golds[0]:
        v.reason = "NO_TEST_OUTPUT: parser found no test results on the gold run (env broken or wrong test cmd)"
        return v
    if len(golds) > 1:  # flakiness across repeated gold runs
        keys = set().union(*golds)
        v.flaky = sorted(k for k in keys if len({g.get(k, "MISSING") in PASS for g in golds}) > 1)
    v.strict_file_pass = all(s_ in PASS or s_ == "SKIPPED" for s_ in golds[0].values())

    if task.get("verifier") == "r2e_gym":
        return _judge_expected(task, v, sb, golds)
    if not f2p:
        return _judge_derive(v, sb, golds)

    v.f2p_pass_on_base = [t for t in f2p if _status(t, sb) in PASS]
    v.f2p_fail_on_base = len(f2p) - len(v.f2p_pass_on_base)
    v.p2p_fail_on_base = sum(_status(t, sb) not in PASS for t in p2p)
    v.f2p_fail_on_gold = [t for t in f2p if any(_status(t, g) not in PASS for g in golds)]
    v.p2p_fail_on_gold = [t for t in p2p if any(_status(t, g) not in PASS for g in golds)]
    if v.f2p_pass_on_base:
        v.reason = f"F2P_PASSES_ON_BASE: {len(v.f2p_pass_on_base)}/{len(f2p)} already pass without the fix"
    elif v.f2p_fail_on_gold:
        v.reason = f"GOLD_FAILS_F2P: {len(v.f2p_fail_on_gold)}/{len(f2p)} still fail with the gold patch"
    elif v.p2p_fail_on_gold:
        v.reason = f"GOLD_BREAKS_P2P: {len(v.p2p_fail_on_gold)}/{len(p2p)} fail with the gold patch"
    else:
        v.valid, v.reason = True, "VALID"
    return v


def _judge_expected(task: dict[str, Any], v: Verdict, sb: dict[str, str], golds: list[dict[str, str]]) -> Verdict:
    """R2E-Gym: the gold run must reproduce expected_output_json exactly (R2E's own reward rule)."""
    v.mode = "expected"
    try:
        expected = r2e_normalize(json.loads(task.get("expected_output_json") or "{}"))
    except (TypeError, ValueError):
        v.reason = "NO_EXPECTED_OUTPUT: expected_output_json is not valid JSON"
        return v
    if not expected:
        v.reason = "NO_EXPECTED_OUTPUT: empty expected_output_json"
        return v
    for g in golds:
        mism = [k for k in expected if g.get(k) != expected[k]]
        if len(g) != len(expected) or mism:
            v.reason = (f"GOLD_MISMATCH_EXPECTED: {len(mism)} of {len(expected)} tests differ from "
                        f"expected_output_json (parsed {len(g)})")
            v.f2p_fail_on_gold = mism[:50]
            return v
    v.derived_f2p = [k for k, st in expected.items() if st == "PASSED" and sb.get(k) != "PASSED"]
    v.derived_p2p = [k for k, st in expected.items() if st == "PASSED" and sb.get(k) == "PASSED"]
    v.f2p_total = len(v.derived_f2p)
    if not v.derived_f2p:
        v.reason = "F2P_PASSES_ON_BASE: every expected-passing test already passes without the fix"
        return v
    v.valid, v.reason = True, "VALID"
    return v


def _judge_derive(v: Verdict, sb: dict[str, str], golds: list[dict[str, str]]) -> Verdict:
    """No F2P list (e.g. ProMax: "every test must pass"): all tests pass with gold, some fail without."""
    v.mode = "derive"
    for g in golds:
        bad = [k for k, st in g.items() if st not in PASS and st != "SKIPPED"]
        if bad:
            v.reason = f"GOLD_FAILS_TESTS: {len(bad)} of {len(g)} tests fail with the gold patch"
            v.p2p_fail_on_gold = bad[:50]
            return v
    passing = [k for k, st in golds[0].items() if st in PASS]
    v.derived_f2p = [k for k in passing if sb.get(k) not in PASS]
    v.derived_p2p = [k for k in passing if sb.get(k) in PASS]
    v.f2p_total = len(v.derived_f2p)
    if not v.derived_f2p:
        v.reason = "F2P_PASSES_ON_BASE: no test goes from failing to passing"
        return v
    v.valid, v.reason = True, "VALID"
    return v


def validate_one(task: dict[str, Any], runner: Any, *, repeat: int = 1,
                 timeout_s: int = 1800) -> tuple[Verdict, str, list[str]]:
    """Returns (verdict, base_log, gold_logs)."""
    iid = task["instance_id"]
    image = task.get("env_image")
    if not image:
        return Verdict(iid, False, "NO_IMAGE"), "", []
    try:
        plan = build_plan(task)
    except UnsupportedVerifier as e:
        return Verdict(iid, False, f"UNSUPPORTED: {e}"), "", []
    except Exception as e:  # noqa: BLE001
        return Verdict(iid, False, f"SPEC_ERROR: {e}"), "", []
    setup = [("setup", task["setup_patch"])] if (task.get("setup_patch") or "").strip() else []
    base = runner.run(image, plan.script, workdir=plan.workdir, patches=setup, timeout_s=timeout_s)
    if base.stage == "image":
        return Verdict(iid, False, f"IMAGE_UNAVAILABLE: {base.log[:300]}", route=plan.route), base.log, []
    if base.stage == "setup_apply":
        return Verdict(iid, False, "SETUP_APPLY_FAIL", route=plan.route), base.log, []
    if base.stage == "timeout":
        return Verdict(iid, False, "TIMEOUT (base run)", route=plan.route), base.log, []
    golds, secs = [], base.seconds
    for _ in range(max(1, repeat)):
        g = runner.run(image, plan.script, workdir=plan.workdir, patches=setup + [("gold", task["patch"])],
                       timeout_s=timeout_s)
        secs += g.seconds
        golds.append(g.log)
        if g.stage.endswith("_apply") or g.stage == "timeout":
            reason = "TIMEOUT (gold run)" if g.stage == "timeout" else f"{g.stage.upper()}_FAIL"
            return Verdict(iid, False, reason, route=plan.route, seconds=secs), base.log, golds
    v = judge(task, plan, base.log, golds)
    v.seconds = secs
    return v, base.log, golds


def select_shard(tasks: list[dict[str, Any]], shard: str | None) -> list[dict[str, Any]]:
    """`shard="K/N"` keeps tasks whose position (after sorting by image) is K mod N, so N machines
    split the work without overlap and each machine pulls roughly 1/N of the images."""
    tasks = sorted(tasks, key=lambda t: (t.get("env_image") or "", t["instance_id"]))
    if not shard:
        return tasks
    k, n = (int(x) for x in shard.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard {shard}: need 0 <= K < N")
    by_image: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:  # whole images go to one shard, so no two machines pull the same image
        by_image.setdefault(t.get("env_image") or t["instance_id"], []).append(t)
    return [t for i, (_, group) in enumerate(sorted(by_image.items())) if i % n == k for t in group]


def run_validation(tasks: list[dict[str, Any]], runner_factory: Callable[[], Any], out_dir: Path, *,
                   workers: int = 4, repeat: int = 1, timeout_s: int = 1800, keep_logs: bool = True,
                   prune_images: bool = False, max_minutes: float | None = None) -> list[Verdict]:
    """Validate tasks in parallel, one task per worker thread. Resumable via verdicts.jsonl.

    max_minutes: stop starting new tasks after this long (tasks already running finish), so a
    time-limited machine (e.g. a 6-hour CI job) ends cleanly; re-run to continue.
    """
    import threading
    import time
    from collections import Counter

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    results_fp = out_dir / "verdicts.jsonl"
    done: set[str] = set()
    if results_fp.exists():  # resumable
        done = {v["instance_id"] for v in iter_jsonl(results_fp) if is_final(v)}
    # tasks sharing an image run next to each other, so the image is pulled once and reused
    todo = sorted((t for t in tasks if t["instance_id"] not in done),
                  key=lambda t: (t.get("env_image") or "", t["instance_id"]))
    runner = runner_factory()
    verdicts: list[Verdict] = []
    remaining = Counter(t.get("env_image") for t in todo)  # for --prune-images: delete after last user
    remaining_lock = threading.Lock()

    def work(t: dict[str, Any]) -> Verdict:
        try:
            v, base_log, gold_logs = validate_one(t, runner, repeat=repeat, timeout_s=timeout_s)
            if keep_logs and (base_log or gold_logs):
                safe = t["instance_id"].replace("/", "_")
                (out_dir / "logs" / f"{safe}.base.log").write_text(base_log)
                for i, g in enumerate(gold_logs):
                    (out_dir / "logs" / f"{safe}.gold{i}.log").write_text(g)
            return v
        finally:
            image = t.get("env_image")
            if prune_images and image:
                with remaining_lock:
                    remaining[image] -= 1
                    last = remaining[image] == 0
                if last:
                    runner.remove_image(image)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool, results_fp.open("a") as f:
        futs = {pool.submit(work, t): t for t in todo}
        stopped = False
        for fut in as_completed(futs):
            if fut.cancelled():
                continue
            if max_minutes and not stopped and time.time() - t0 > max_minutes * 60:
                stopped = True
                n_cancel = sum(f_.cancel() for f_ in futs)
                print(f"--max-minutes reached: {n_cancel} tasks not started; re-run to continue", flush=True)
            try:
                v = fut.result()
            except Exception as e:  # noqa: BLE001
                v = Verdict(futs[fut]["instance_id"], False, f"INFRA_ERROR: {e}")
            f.write(json.dumps(asdict(v)) + "\n")
            f.flush()
            verdicts.append(v)
            n = len(verdicts)
            eta = (time.time() - t0) / n * (len(todo) - n) / 60
            print(f"[{n}/{len(todo)} ~{eta:.0f} min left] {v.instance_id}: {v.reason}", flush=True)
    return verdicts
