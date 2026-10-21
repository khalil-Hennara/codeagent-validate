"""Stage 4 (generate) tests: the real swegemma agent loop with a scripted fake model, a local
subprocess sandbox instead of Docker, and a fake verification runner.

Needs the harness on PYTHONPATH (dirs holding swegemma/, adk_eval_core/, adk_submission/):
    PYTHONPATH=harness python3 tests/test_generate.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from codeagent_data.generate import rollout  # noqa: E402
from codeagent_data.generate.prepare import prepare_script  # noqa: E402
from codeagent_data.validate.docker_runner import RunResult  # noqa: E402

BUGGY = "def add(a, b):\n    return a - b\n"
TEST_PATCH = """diff --git a/tests/test_calc.py b/tests/test_calc.py
new file mode 100644
--- /dev/null
+++ b/tests/test_calc.py
@@ -0,0 +1,3 @@
+from calc import add
+def test_add():
+    assert add(2, 3) == 5
"""
TASK = {
    "instance_id": "toy__calc-1", "uid": "toy::1", "repo": "toy/calc", "verifier": "swe_gym",
    "base_commit": "", "problem_statement": "add(2, 3) returns -1 instead of 5.",
    "test_patch": TEST_PATCH, "env_image": "toy:latest", "env_workdir": "/testbed",
    "FAIL_TO_PASS": ["tests/test_calc.py::test_add"], "PASS_TO_PASS": [],
    "selection_rank": 0, "fold": "A", "version": "",
    "patch": "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n",
}


class ScriptedLlm(BaseLlm):
    """Plays a fixed tool-use script; step = number of tool results seen so far."""

    model: str = "scripted"
    script: list = []

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        seen = sum(1 for c in llm_request.contents for p in (c.parts or []) if p.function_response)
        name, args = self.script[min(seen, len(self.script) - 1)]
        parts = [types.Part(text=f"step {seen}: thinking about {name}", thought=True)]
        if name == "final":
            parts.append(types.Part(text="Fixed add()."))
        else:
            parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
        yield LlmResponse(content=types.Content(role="model", parts=parts),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(
                              prompt_token_count=1000, candidates_token_count=100,
                              cached_content_token_count=400, total_token_count=1100))


FIX_SCRIPT = [
    ("read_file", {"filepath": "calc.py"}),
    ("edit_file", {"filepath": "calc.py", "old_string": "return a - b", "new_string": "return a + b"}),
    ("run_command", {"command": "python -c 'from calc import add; print(add(2, 3))'"}),
    ("submit_patch", {}),
    ("final", {}),
]


def toy_sandbox_factory(task):
    """SubprocessManager whose workspace starts as the toy repo at its base commit."""
    from swegemma.sandbox import SubprocessManager

    class ToySandbox(SubprocessManager):
        def start(self) -> str:
            sid = super().start()
            ws = self._sandboxes[sid]["workspace"]
            (ws / "calc.py").write_text(BUGGY)
            subprocess.run("git init -q && git add -A && git -c user.email=a@b -c user.name=a commit -qm base",
                           shell=True, cwd=ws, check=True)
            return sid

    return ToySandbox()


class ApplyAndPytestRunner:
    """Verification stand-in: applies setup + agent patch + test patch to the toy repo and runs pytest -rA."""

    def __init__(self):
        self.calls = []

    def run(self, image, script, *, workdir, patches=None, timeout_s=1800):
        self.calls.append([label for label, _ in patches or []])
        d = Path(tempfile.mkdtemp())
        try:
            (d / "calc.py").write_text(BUGGY)
            subprocess.run("git init -q", shell=True, cwd=d, check=True)
            for label, diff in (patches or []) + [("test", TASK["test_patch"])]:
                p = subprocess.run(["git", "apply", "-"], input=diff, text=True, cwd=d, capture_output=True)
                if p.returncode:
                    return RunResult(False, p.stderr, f"{label}_apply", 0.1)
            # pytest -rA style result line, without needing pytest installed
            ok = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'tests'); "
                                 "import test_calc; test_calc.test_add()"], cwd=d, capture_output=True).returncode == 0
            out = f"{'PASSED' if ok else 'FAILED'} tests/test_calc.py::test_add\n"
            return RunResult(True, ">>>>> Eval script\n>>>>> Start Test Output\n" + out
                             + ">>>>> End Test Output\n", "done", 0.1)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def ensure_image(self, image):
        return None

    def remove_image(self, image):
        pass


def cfg(**kw):
    return rollout.GenConfig(time_minutes=5, tool_calls=20, turns=30, price_in=1.0, price_out=4.0,
                             price_cached=0.1, **kw)


def test_solves_and_records():
    out = Path(tempfile.mkdtemp())
    runner = ApplyAndPytestRunner()
    a = rollout.run_one(TASK, cfg(), out, runner, sandbox_factory=toy_sandbox_factory,
                        model_override=ScriptedLlm(script=FIX_SCRIPT))
    assert a.solved, (a.reason, a.agent_error, a.patch)
    assert "+    return a + b" in a.patch
    assert runner.calls == [["agent"]]
    assert a.stats["tool_calls"] == 4 and a.stats["thinking_entries"] >= 4, a.stats
    assert a.stats["prompt_tokens"] >= 5000, a.stats
    # cost = (uncached * 1 + cached * 0.1 + completion * 4) / 1e6
    st = a.stats
    want = ((st["prompt_tokens"] - st["cached_tokens"]) + 0.1 * st["cached_tokens"] + 4 * st["completion_tokens"]) / 1e6
    assert abs(a.est_cost_usd - round(want, 4)) < 1e-9
    trace = json.loads((out / a.trace_path).read_text())
    kinds = [e["type"] for e in trace["entries"]]
    assert {"thinking", "tool_call", "tool_response", "final"} <= set(kinds), kinds
    assert any(e["type"] == "custom" or e.get("content", "").startswith("You are") for e in trace["entries"])
    print("ok test_solves_and_records", a.stats, f"${a.est_cost_usd}")


def test_wrong_fix_not_solved():
    bad = list(FIX_SCRIPT)
    bad[1] = ("edit_file", {"filepath": "calc.py", "old_string": "return a - b", "new_string": "return a * b"})
    a = rollout.run_one(TASK, cfg(), Path(tempfile.mkdtemp()), ApplyAndPytestRunner(),
                        sandbox_factory=toy_sandbox_factory, model_override=ScriptedLlm(script=bad))
    assert not a.solved and a.reason.startswith("FAILS_F2P"), a.reason
    print("ok test_wrong_fix_not_solved", a.reason[:60])


def test_no_patch():
    script = [("read_file", {"filepath": "calc.py"}), ("final", {})]
    a = rollout.run_one(TASK, cfg(), Path(tempfile.mkdtemp()), ApplyAndPytestRunner(),
                        sandbox_factory=toy_sandbox_factory, model_override=ScriptedLlm(script=script))
    assert not a.solved and a.reason.startswith("NO_PATCH"), a.reason
    print("ok test_no_patch", a.reason[:60])


def test_run_generation_target_limit_and_resume():
    tasks = [{**TASK, "instance_id": f"toy__calc-{i}", "selection_rank": i} for i in range(6)]
    targets = {"solved_target": 2, "max_per_repo": 100, "max_share": {"swe_gym": 1.0},
               "shares": {"swe_gym": 1.0}}
    out = Path(tempfile.mkdtemp())
    kw = dict(runner_factory=ApplyAndPytestRunner, workers=1, sandbox_factory=toy_sandbox_factory,
              model_override=ScriptedLlm(script=FIX_SCRIPT))
    res = rollout.run_generation(tasks, targets, cfg(), out, **kw)
    assert sum(r.solved for r in res) == 2 and len(res) == 2, [r.reason for r in res]
    # resume: finished attempts are kept, the target is already met -> nothing new runs
    res2 = rollout.run_generation(tasks, targets, cfg(), out, **kw)
    assert res2 == []
    # a bigger target picks up where the first run stopped, and --limit caps new attempts
    res3 = rollout.run_generation(tasks, {**targets, "solved_target": 10}, cfg(), out, limit=1, **kw)
    assert [r.instance_id for r in res3] == ["toy__calc-2"]
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    rep = rollout.render_generation_report(rows, targets)
    assert "attempts: 3  solved: 3" in rep and "projection" in rep, rep
    # a second shard folder + merge with global limits: max 1 per repo -> only the best-ranked one
    out2 = Path(tempfile.mkdtemp())
    rollout.run_generation(tasks[3:], {**targets, "solved_target": 10}, cfg(), out2, limit=1, **kw)
    merged = Path(tempfile.mkdtemp())
    st = rollout.merge_generation(tasks, [out / "results.jsonl", out2 / "results.jsonl"],
                                  {**targets, "max_per_repo": 1}, merged)
    assert st == {"rows": 4, "solved": 4, "accepted": 1}, st
    rows = [json.loads(x) for x in (merged / "solved.jsonl").read_text().splitlines()]
    assert rows[0]["instance_id"] == "toy__calc-0" and "+    return a + b" in rows[0]["teacher"]["patch"]
    assert (merged / rows[0]["teacher"]["trace"]).exists()
    # prior results: a new run (fresh folder) skips tasks finished before and counts their solves
    out3 = Path(tempfile.mkdtemp())
    prior = [json.loads(x) for x in (merged / "results_merged.jsonl").read_text().splitlines()]
    res4 = rollout.run_generation(tasks, {**targets, "solved_target": 5}, cfg(), out3, prior=prior, **kw)
    assert [r.instance_id for r in res4] == ["toy__calc-4"], [r.instance_id for r in res4]
    print("ok test_run_generation_target_limit_and_resume")


def test_second_attempt_and_verify_log():
    """attempts=2: a wrong first try is re-queued after the new tasks and the second try solves it;
    the failing try keeps its verification log and both traces are kept."""
    tries: dict = {}

    class FlakyLlm(ScriptedLlm):
        async def generate_content_async(self, llm_request, stream=False):
            seen = sum(1 for c in llm_request.contents for p in (c.parts or []) if p.function_response)
            iid = "x"
            if seen == 0:
                tries[iid] = tries.get(iid, 0) + 1
            bad = list(FIX_SCRIPT)
            bad[1] = ("edit_file", {"filepath": "calc.py", "old_string": "return a - b", "new_string": "return a * b"})
            self.script = bad if tries[iid] == 1 else FIX_SCRIPT
            async for r in super().generate_content_async(llm_request, stream):
                yield r

    out = Path(tempfile.mkdtemp())
    targets = {"solved_target": 5, "max_per_repo": 100, "max_share": {}}
    res = rollout.run_generation([TASK], targets, cfg(), out, runner_factory=ApplyAndPytestRunner, workers=1,
                                 attempts=2, sandbox_factory=toy_sandbox_factory, model_override=FlakyLlm())
    assert [(r.attempt, r.solved) for r in res] == [(1, False), (2, True)], [(r.attempt, r.reason) for r in res]
    assert "failing: tests/test_calc.py::test_add" in res[0].reason, res[0].reason
    assert (out / "logs" / "toy__calc-1__a1.verify.log").exists()
    assert (out / res[0].trace_path).exists() and (out / res[1].trace_path).exists()
    # resume: nothing left; merge picks the solved try
    assert rollout.run_generation([TASK], targets, cfg(), out, runner_factory=ApplyAndPytestRunner,
                                  attempts=2, sandbox_factory=toy_sandbox_factory, model_override=FlakyLlm()) == []
    merged = Path(tempfile.mkdtemp())
    st = rollout.merge_generation([TASK], [out / "results.jsonl"], targets, merged)
    assert st == {"rows": 2, "solved": 1, "accepted": 1}, st
    row = json.loads((merged / "solved.jsonl").read_text())
    assert row["teacher"]["attempt"] == 2 and row["teacher"]["trace"].endswith("__a2.json")
    rep = (merged / "generation_report.md").read_text()
    assert "tasks: 1  solved: 1  (100%; first try 0%)" in rep, rep
    print("ok test_second_attempt_and_verify_log")


def test_retry_only_when_promising_and_flaky_rerun():
    # wrong area (edits another file, no F2P passes): not worth a second try
    wrong_area = [("write_file", {"filepath": "other.py", "content": "x = 1\n"}), ("submit_patch", {}), ("final", {})]
    out = Path(tempfile.mkdtemp())
    res = rollout.run_generation([TASK], {"solved_target": 5, "max_per_repo": 100, "max_share": {}}, cfg(), out,
                                 runner_factory=ApplyAndPytestRunner, workers=1, attempts=3,
                                 sandbox_factory=toy_sandbox_factory, model_override=ScriptedLlm(script=wrong_area))
    assert len(res) == 1 and not res[0].solved, [r.reason for r in res]
    assert res[0].loc == {"gold_files": ["calc.py"], "teacher_files": ["other.py"], "hit": 0, "all_gold_files": False}
    assert res[0].tests["f2p_pass_frac"] == 0 and res[0].gold == {"files": 1, "lines": 2, "hunks": 1}
    assert not rollout.retry_worthy(res[0])

    # a test that fails once and passes on the re-run: accepted, marked flaky
    class FlakyRunner(ApplyAndPytestRunner):
        def run(self, image, script, *, workdir, patches=None, timeout_s=1800):
            r = super().run(image, script, workdir=workdir, patches=patches, timeout_s=timeout_s)
            if len(self.calls) == 1:
                r.log = r.log.replace("PASSED tests/test_calc.py", "FAILED tests/test_calc.py")
            return r
    runner = FlakyRunner()
    ok, reason, _ = rollout.verify_patch(TASK, TASK["patch"], runner)
    assert ok and "flaky" in reason and len(runner.calls) == 2, reason
    rep = rollout.render_generation_report([{**rollout.asdict(res[0])}])
    assert "editing at least one file of the reference fix: 0" in rep and "| 1-10 | 1 | 0 | 0% |" in rep, rep
    print("ok test_retry_only_when_promising_and_flaky_rerun")


def test_recall_of_project_history_is_flagged():
    recalled = ("The actual pandas fix (PR #48425) changed this. Let me recall the upstream commit "
                "6d42d5e1a2e9 from pandas main today; it was fixed in version 1.5.1 per the release notes.")
    assert rollout.recall_mentions(recalled) >= rollout.RECALL_LIMIT
    for innocent in ["the issue #12 says", "the upstream caller passes x", "# GH3485 in the test file",
                     "the actual fix is to change line 3", "git checkout main"]:
        assert rollout.recall_mentions(innocent) <= 1, innocent
    row = {"instance_id": "x", "solved": True, "verifier": "swe_gym", "repo": "a/b", "stats": {"recall_mentions": 9}}
    ok_row = {**row, "instance_id": "y", "stats": {"recall_mentions": 1}}
    out = Path(tempfile.mkdtemp())
    (out / "r.jsonl").write_text("\n".join(json.dumps(r) for r in (row, ok_row)) + "\n")
    tasks = [{**TASK, "instance_id": "x", "repo": "a/b"}, {**TASK, "instance_id": "y", "repo": "a/b"}]
    targets = {"solved_target": 10, "max_per_repo": 100, "max_share": {}}
    st = rollout.merge_generation(tasks, [out / "r.jsonl"], targets, out / "m")  # default: keep, clean later
    assert st["accepted"] == 2 and "recalls project history (to clean with `clean-recall`): 1 of 2" in \
        (out / "m" / "generation_report.md").read_text()
    assert rollout.merge_generation(tasks, [out / "r.jsonl"], targets, out / "m2", keep_recall=False)["accepted"] == 1
    print("ok test_recall_of_project_history_is_flagged")


def test_ledger_resume_never_repeats_a_task():
    """Run 1 solves t0, fails t1 (wrong fix), and t2 hits a credit error; the ledger keeps all of
    it. Run 2 (fresh folder, like a new GitHub run) reads the ledger: only t2 and t3 run."""
    from codeagent_data.generate.ledger import update_ledger

    bad = list(FIX_SCRIPT)
    bad[1] = ("edit_file", {"filepath": "calc.py", "old_string": "return a - b", "new_string": "return a * b"})

    class PerTaskLlm(ScriptedLlm):
        async def generate_content_async(self, llm_request, stream=False):
            prompt = llm_request.contents[0].parts[0].text
            self.script = bad if "TASK-t1" in prompt else FIX_SCRIPT
            async for r in super().generate_content_async(llm_request, stream):
                yield r

    tasks = [{**TASK, "instance_id": f"t{i}", "selection_rank": i,
              "problem_statement": TASK["problem_statement"] + f" TASK-t{i}"} for i in range(4)]
    targets = {"solved_target": 10, "max_per_repo": 100, "max_share": {}}
    kw = dict(runner_factory=ApplyAndPytestRunner, workers=1, sandbox_factory=toy_sandbox_factory,
              model_override=PerTaskLlm())
    run1 = Path(tempfile.mkdtemp())
    rollout.run_generation(tasks[:2], targets, cfg(), run1, attempts=1, **kw)
    with (run1 / "results.jsonl").open("a") as f:  # t2: the credit ran out mid-task
        f.write(json.dumps({"instance_id": "t2", "attempt": 1, "solved": False, "verifier": "swe_gym",
                            "repo": "toy/calc", "reason": "FATAL_API_ERROR: 402 Insufficient credits"}) + "\n")
    ledger = Path(tempfile.mkdtemp()) / "ledger"
    st = update_ledger(ledger, [run1 / "results.jsonl"])
    assert st["tasks_finished"] == 2 and st["tasks_solved"] == 1 and st["tasks_to_retry"] == 1, st
    assert (ledger / "traces" / "t0__a1.json").exists() and not (ledger / "traces" / "t1__a1.json").exists()

    run2 = Path(tempfile.mkdtemp())
    prior = [json.loads(x) for x in (ledger / "results.jsonl").read_text().splitlines()]
    res = rollout.run_generation(tasks, targets, cfg(), run2, attempts=2, prior=prior, **kw)
    assert sorted(r.instance_id for r in res) == ["t2", "t3"], [r.instance_id for r in res]
    st = update_ledger(ledger, [run2 / "results.jsonl"])
    assert st["tasks_finished"] == 4 and st["tasks_solved"] == 3 and st["tasks_to_retry"] == 0, st
    # the final merge works from the ledger alone (traces included)
    m = rollout.merge_generation(tasks, [ledger / "results.jsonl"], targets, Path(tempfile.mkdtemp()) / "g")
    assert m["accepted"] == 3, m
    print("ok test_ledger_resume_never_repeats_a_task")


def test_stops_when_the_api_keeps_failing():
    calls = []

    def failing_run_one(task, *a, attempt=1, **k):
        calls.append(task["instance_id"])
        return rollout.Attempt(task["instance_id"], verifier="swe_gym", repo="toy/calc",
                               reason="API_ERROR: Sandbox execution error: litellm.APIConnectionError: boom",
                               stats={"completion_tokens": 0})

    real, rollout.run_one = rollout.run_one, failing_run_one
    try:
        tasks = [{**TASK, "instance_id": f"t{i}", "selection_rank": i} for i in range(10)]
        targets = {"solved_target": 5, "max_per_repo": 100, "max_share": {}}
        rollout.run_generation(tasks, targets, cfg(), Path(tempfile.mkdtemp()), runner_factory=ApplyAndPytestRunner,
                               workers=1)
    finally:
        rollout.run_one = real
    assert calls == ["t0", "t1", "t2"], calls
    print("ok test_stops_when_the_api_keeps_failing")


def test_classify_and_shards():
    assert rollout.classify_agent_error("Sandbox execution error: litellm.AuthenticationError: bad key") == "FATAL_API_ERROR"
    assert rollout.classify_agent_error("Sandbox execution error: litellm.RateLimitError: 429") == "API_ERROR"
    assert rollout.classify_agent_error("Agent exceeded session timeout (45 min)") is None
    assert rollout.classify_agent_error('OpenrouterException - {"error": {"code": 403, "message": "Budget limit '
                                        'exceeded (key). $52.00 of $50.00 used"}}') == "FATAL_API_ERROR"
    t = rollout.shard_targets({"solved_target": 2400, "max_per_repo": 120, "max_share": {"a": .5}}, 20)
    assert t["solved_target"] == 126 and t["max_per_repo"] == 7
    tasks = [{"instance_id": str(i), "selection_rank": i} for i in range(10)]
    sel, n = rollout.select_shard_tasks(list(reversed(tasks)), "1/3")
    assert n == 3 and [x["instance_id"] for x in sel] == ["1", "4", "7"]
    print("ok test_classify_and_shards")


def test_prepare_script_in_container_layout():
    """prepare_script on a fake /testbed: moved to the workspace, future refs gone, bug folded in."""
    root = Path(tempfile.mkdtemp())
    src, ws = root / "testbed", root / "workspace"
    src.mkdir()
    sh = lambda c, cwd=src: subprocess.run(c, shell=True, cwd=cwd, check=True, capture_output=True, text=True).stdout  # noqa: E731
    (src / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    sh("git init -q -b main && git -c user.email=a@b -c user.name=a commit -qam x --allow-empty && "
       "git add -A && git -c user.email=a@b -c user.name=a commit -qm base")
    sh("git checkout -qb fix && echo '# future' >> calc.py && git -c user.email=a@b -c user.name=a commit -qam fix "
       "&& git checkout -q main")
    # a remote whose HEAD is a symref to a branch holding the fix (datalad's R2E image): git refuses
    # to delete a symref and its target in one batch, so prepare must handle symrefs first
    sh("git remote add origin https://example.invalid/repo.git && git update-ref refs/remotes/origin/maint fix "
       "&& git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/maint")
    (root / "setup.patch").write_text(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b\n")
    (src / "r2e_tests").mkdir()
    script = prepare_script(str(src), str(ws), has_setup_patch=True, setup_patch_path=str(root / "setup.patch"),
                            hidden_paths=[])
    script = script.replace("git config --global", "git config")  # don't touch the host's global config
    assert "PREPARE_OK" in subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
    assert (src.is_symlink() and (ws / "calc.py").read_text().endswith("a - b\n")
            and not (ws / "r2e_tests").exists())
    assert sh("git branch -a", ws).split() == ["*", "main"] and sh("git remote", ws) == ""
    assert sh("git status --porcelain", ws) == "" and "future" not in sh("git log --all -p", ws)
    print("ok test_prepare_script_in_container_layout")


if __name__ == "__main__":
    test_classify_and_shards()
    test_recall_of_project_history_is_flagged()
    test_stops_when_the_api_keeps_failing()
    test_second_attempt_and_verify_log()
    test_ledger_resume_never_repeats_a_task()
    test_retry_only_when_promising_and_flaky_rerun()
    test_prepare_script_in_container_layout()
    test_solves_and_records()
    test_wrong_fix_not_solved()
    test_no_patch()
    test_run_generation_target_limit_and_resume()
    print("all generate tests passed")
