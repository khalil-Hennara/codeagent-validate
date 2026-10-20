"""Stage 1 tests with a fake Docker runner. Run: python3 tests/test_stage1.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codeagent_data.validate.docker_runner import RunResult  # noqa: E402
from codeagent_data.validate.specs import build_plan  # noqa: E402
from codeagent_data.validate.validator import validate_one  # noqa: E402

F2P = ["test_union_with_values_list_and_order (queries.test_qs_combinators.QuerySetSetOperationTests)"]
P2P = ["test_count_union (queries.test_qs_combinators.QuerySetSetOperationTests)"]
TEST_PATCH = """diff --git a/tests/queries/test_qs_combinators.py b/tests/queries/test_qs_combinators.py
--- a/tests/queries/test_qs_combinators.py
+++ b/tests/queries/test_qs_combinators.py
@@ -1,2 +1,3 @@
 a
 b
+c
"""
DJANGO_TASK = dict(instance_id="django__django-10554", repo="django/django", version="3.0",
                   base_commit="14d026cccb144c6877294ba4cd4e03ebf0842498", patch="diff --git a/x b/x\n",
                   test_patch=TEST_PATCH, FAIL_TO_PASS=F2P, PASS_TO_PASS=P2P,
                   environment_setup_commit="419a78300f7cd27611196e1e464d50fd0385ff27",
                   env_image="swebench/sweb.eval.x86_64.django_1776_django-10554:latest")


def django_log(f2p_ok: bool, p2p_ok: bool = True, patch_err: bool = False) -> str:
    pre = "Checking patch tests/queries/test_qs_combinators.py...\n"
    pre += "error: patch failed: tests/queries/test_qs_combinators.py:1\n" if patch_err else "Applied patch tests/queries/test_qs_combinators.py cleanly.\n"
    body = (
        f"{F2P[0]} ... {'ok' if f2p_ok else 'FAIL'}\n"
        f"{P2P[0]} ... {'ok' if p2p_ok else 'FAIL'}\n"
    )
    return pre + "+ : '>>>>> Start Test Output'\n" + body + "+ : '>>>>> End Test Output'\n"


class FakeRunner:
    def __init__(self, base: str, gold: list[str], gold_stage: str = "done"):
        self.base, self.gold, self.gold_stage, self.calls = base, list(gold), gold_stage, []

    def run(self, image, script, *, workdir, patches=None, timeout_s=1800):
        labels = [l for l, _ in (patches or [])]
        self.calls.append((labels, workdir))
        if "gold" not in labels:
            return RunResult(True, self.base, "done", 1.0)
        return RunResult(self.gold_stage == "done", self.gold.pop(0), self.gold_stage, 1.0)

    def remove_image(self, image):
        pass


def test_swebench_route_and_script():
    plan = build_plan(DJANGO_TASK)
    assert plan.route == "swebench-harness" and plan.workdir == "/testbed"
    assert "./tests/runtests.py" in plan.script and "queries.test_qs_combinators" in plan.script


def test_valid_task():
    r = FakeRunner(django_log(False), [django_log(True), django_log(True)])
    v, _, _ = validate_one(DJANGO_TASK, r, repeat=2)
    assert v.valid, v
    assert r.calls == [([], "/testbed"), (["gold"], "/testbed"), (["gold"], "/testbed")]


def test_f2p_already_passing_on_base():
    v, _, _ = validate_one(DJANGO_TASK, FakeRunner(django_log(True), [django_log(True)]))
    assert not v.valid and v.reason.startswith("F2P_PASSES_ON_BASE")


def test_gold_breaks_p2p():
    v, _, _ = validate_one(DJANGO_TASK, FakeRunner(django_log(False), [django_log(True, p2p_ok=False)]))
    assert not v.valid and v.reason.startswith("GOLD_BREAKS_P2P")


def test_flaky_detected():
    v, _, _ = validate_one(DJANGO_TASK, FakeRunner(django_log(False), [django_log(True), django_log(False)]), repeat=2)
    assert not v.valid and F2P[0] in v.flaky


def test_test_patch_apply_failure():
    v, _, _ = validate_one(DJANGO_TASK, FakeRunner(django_log(False, patch_err=True), [django_log(True)]))
    assert v.reason.startswith("TEST_PATCH_APPLY_FAIL")


def test_gold_apply_failure():
    v, _, _ = validate_one(DJANGO_TASK, FakeRunner(django_log(False), ["x"], gold_stage="gold_apply"))
    assert v.reason == "GOLD_APPLY_FAIL", v.reason


def test_generic_route_for_unknown_repo():
    t = dict(DJANGO_TASK, repo="acme/widget", version="", env_workdir="/testbed",
             test_patch=TEST_PATCH.replace("tests/queries/test_qs_combinators.py", "tests/test_core.py"),
             FAIL_TO_PASS=["tests/test_core.py::test_a"], PASS_TO_PASS=[])
    plan = build_plan(t)
    assert plan.route == "generic" and "pytest -rA tests/test_core.py" in plan.script
    log = "+ : '>>>>> Start Test Output'\nPASSED tests/test_core.py::test_a\n+ : '>>>>> End Test Output'\n"
    assert plan.parse(log) == {"tests/test_core.py::test_a": "PASSED"}


def test_smith_setup_patch_order_and_generic_plan():
    t = dict(instance_id="oauthlib__oauthlib.1fd52536.func_pm_1", repo="oauthlib/oauthlib", version="",
             base_commit="1fd52536", patch="FIX", setup_patch="BUG", test_patch="", verifier="swe_smith",
             FAIL_TO_PASS=["tests/test_a.py::test_x"], PASS_TO_PASS=["tests/test_b.py::test_y"],
             env_image="jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536")
    plan = build_plan(t)
    assert plan.route == "generic" and "pytest -rA tests/test_a.py tests/test_b.py" in plan.script
    assert "git apply" not in plan.script  # no test_patch for SWE-smith
    ok = "+ : '>>>>> Start Test Output'\nPASSED tests/test_a.py::test_x\nPASSED tests/test_b.py::test_y\n+ : '>>>>> End Test Output'\n"
    bad = ok.replace("PASSED tests/test_a.py", "FAILED tests/test_a.py")
    r = FakeRunner(bad, [ok])
    v, _, _ = validate_one(t, r)
    assert v.valid, v
    assert r.calls[0][0] == ["setup"] and r.calls[1][0] == ["setup", "gold"]


def test_row_eval_script_route():
    t = dict(DJANGO_TASK, eval_script="#!/bin/bash\necho official\n", log_parser="parse_log_django")
    plan = build_plan(t)
    assert plan.route == "row-eval-script" and "echo official" in plan.script
    v, _, _ = validate_one(t, FakeRunner(django_log(False), [django_log(True)]))
    assert v.valid and v.route == "row-eval-script"


def test_gold_fallback_errors_do_not_count_as_test_patch_failure():
    noisy = ">>>>> Apply gold\nerror: patch failed: x.py:3\nChecking patch x.py...\nHunk #1 succeeded\n>>>>> Eval script\n" + django_log(True)
    v, _, _ = validate_one(DJANGO_TASK, FakeRunner(django_log(False), [noisy]))
    assert v.valid, v.reason


def test_unsupported_verifier():
    v, _, _ = validate_one(dict(DJANGO_TASK, verifier="swe_bench_pro"), FakeRunner("", []))
    assert v.reason.startswith("UNSUPPORTED")


R2E_TASK = dict(instance_id="pandas__abc", repo="pandas-dev/pandas", base_commit="abc^", verifier="r2e_gym",
                patch="FIX", test_patch="", FAIL_TO_PASS=[], PASS_TO_PASS=[],
                expected_output_json=json.dumps({"TestA.test_x": "PASSED", "test_y": "PASSED", "test_z": "FAILED"}),
                env_image="namanjain12/pandas_final:abc")


def r2e_log(statuses):
    body = "\n".join(f"{st} r2e_tests/test_1.py::{k.replace('.', '::')}" for k, st in statuses.items())
    return f"+ : '>>>>> Start Test Output'\n=== short test summary info ===\n{body}\n+ : '>>>>> End Test Output'\n"


def test_r2e_valid_and_derived_lists():
    base = r2e_log({"TestA.test_x": "PASSED", "test_y": "FAILED", "test_z": "FAILED"})
    gold = r2e_log({"TestA.test_x": "PASSED", "test_y": "PASSED", "test_z": "FAILED"})
    v, _, _ = validate_one(R2E_TASK, FakeRunner(base, [gold]))
    assert v.valid and v.mode == "expected", v.reason
    assert v.derived_f2p == ["test_y"] and v.derived_p2p == ["TestA.test_x"]


def test_r2e_gold_mismatch():
    base = r2e_log({"TestA.test_x": "PASSED", "test_y": "FAILED", "test_z": "FAILED"})
    v, _, _ = validate_one(R2E_TASK, FakeRunner(base, [base]))
    assert v.reason.startswith("GOLD_MISMATCH_EXPECTED")


def test_derive_mode_without_f2p():
    t = dict(DJANGO_TASK, repo="optuna/optuna", version="", eval_script="", FAIL_TO_PASS=[], PASS_TO_PASS=[],
             test_patch=TEST_PATCH.replace("tests/queries/test_qs_combinators.py", "tests/test_acqf.py"))
    ok = "+ : '>>>>> Start Test Output'\nPASSED tests/test_acqf.py::test_a\nPASSED tests/test_acqf.py::test_b\n+ : '>>>>> End Test Output'\n"
    v, _, _ = validate_one(t, FakeRunner(ok.replace("PASSED tests/test_acqf.py::test_b", "FAILED tests/test_acqf.py::test_b"), [ok]))
    assert v.valid and v.mode == "derive" and v.derived_f2p == ["tests/test_acqf.py::test_b"], v.reason


def test_data_file_test_patch_runs_listed_tests():
    tp = "diff --git a/test-data/unit/check-literal.test b/test-data/unit/check-literal.test\n--- a/test-data/unit/check-literal.test\n+++ b/test-data/unit/check-literal.test\n@@ -1 +1,2 @@\n x\n+[case testNew]\n"
    t = dict(DJANGO_TASK, repo="python/mypy", version="", test_patch=tp,
             FAIL_TO_PASS=["mypy/test/testcheck.py::TypeCheckSuite::check-literal.test::testNew"], PASS_TO_PASS=[])
    plan = build_plan(t)
    assert "pytest -rA mypy/test/testcheck.py::TypeCheckSuite::check-literal.test::testNew" in plan.script


def test_shards_partition_by_image():
    from codeagent_data.validate.validator import select_shard
    tasks = [dict(instance_id=f"t{i}", env_image=f"img{i % 5}") for i in range(20)]
    shards = [select_shard(tasks, f"{k}/3") for k in range(3)]
    ids = [t["instance_id"] for s in shards for t in s]
    assert sorted(ids) == sorted(t["instance_id"] for t in tasks)  # complete, no overlap
    imgs = [{t["env_image"] for t in s} for s in shards]
    assert not (imgs[0] & imgs[1] or imgs[0] & imgs[2] or imgs[1] & imgs[2])  # no image pulled twice


def test_parallel_run_validation_with_pruning():
    import tempfile
    from codeagent_data.validate.validator import run_validation

    class Runner(FakeRunner):
        def __init__(self):
            self.removed = []
        def run(self, image, script, *, workdir, patches=None, timeout_s=1800):
            labels = [l for l, _ in (patches or [])]
            return RunResult(True, django_log("gold" in labels), "done", 0.01)
        def remove_image(self, image):
            self.removed.append(image)

    r = Runner()
    tasks = [dict(DJANGO_TASK, instance_id=f"django__django-{i}", env_image=f"img{i % 3}") for i in range(9)]
    with tempfile.TemporaryDirectory() as d:
        vs = run_validation(tasks, lambda: r, Path(d), workers=4, prune_images=True)
        assert len(vs) == 9 and all(v.valid for v in vs)
        assert sorted(r.removed) == ["img0", "img1", "img2"]  # each removed once, after its last task
        assert run_validation(tasks, lambda: r, Path(d), workers=4) == []  # resumable: nothing left


def test_max_minutes_stops_cleanly_and_retryable_reruns():
    import tempfile
    import time as _t
    from codeagent_data.validate.validator import run_validation

    class Slow(FakeRunner):
        def __init__(self, fail_pull=False):
            self.fail_pull = fail_pull
        def run(self, image, script, *, workdir, patches=None, timeout_s=1800):
            if self.fail_pull:
                return RunResult(False, "pull failed: toomanyrequests", "image", 0.0)
            _t.sleep(0.2)
            return RunResult(True, django_log("gold" in [l for l, _ in (patches or [])]), "done", 0.2)
        def remove_image(self, image):
            pass

    tasks = [dict(DJANGO_TASK, instance_id=f"django__django-{i}", env_image=f"img{i}") for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        vs = run_validation(tasks, lambda: Slow(), Path(d), workers=1, max_minutes=0.004)  # ~0.25 s
        assert 0 < len(vs) < 10, len(vs)
        rest = run_validation(tasks, lambda: Slow(), Path(d), workers=2)
        assert len(vs) + len(rest) == 10  # second run finishes exactly the remainder
    with tempfile.TemporaryDirectory() as d:
        vs = run_validation(tasks[:3], lambda: Slow(fail_pull=True), Path(d), workers=2)
        assert all(v.reason.startswith("IMAGE_UNAVAILABLE") for v in vs)
        again = run_validation(tasks[:3], lambda: Slow(), Path(d), workers=2)
        assert len(again) == 3 and all(v.valid for v in again)  # pull failures are retried


def test_merge_combines_shards_later_wins():
    import tempfile
    from codeagent_data.cli import main
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tasks = [dict(DJANGO_TASK, instance_id=f"t{i}") for i in range(3)]
        (d / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
        (d / "a.jsonl").write_text("\n".join(json.dumps(dict(instance_id=f"t{i}", valid=False, reason="IMAGE_UNAVAILABLE: x")) for i in range(3)))
        (d / "b.jsonl").write_text(json.dumps(dict(instance_id="t1", valid=True, reason="VALID", mode="lists", route="r")))
        main(["merge", "--tasks", str(d / "tasks.jsonl"), "--verdicts", str(d / "a.jsonl"), str(d / "b.jsonl"), "--out", str(d / "m")])
        valid = [json.loads(l)["instance_id"] for l in (d / "m/valid_tasks.jsonl").read_text().splitlines()]
        assert valid == ["t1"] and (d / "m/verdicts_merged.jsonl").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
