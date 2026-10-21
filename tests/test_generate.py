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
    assert not a.solved and a.reason.startswith("GOLD_FAILS_F2P"), a.reason
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


def test_classify_and_shards():
    assert rollout.classify_agent_error("Sandbox execution error: litellm.AuthenticationError: bad key") == "FATAL_API_ERROR"
    assert rollout.classify_agent_error("Sandbox execution error: litellm.RateLimitError: 429") == "API_ERROR"
    assert rollout.classify_agent_error("Agent exceeded session timeout (45 min)") is None
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
    (root / "setup.patch").write_text(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return a - b\n")
    (src / "r2e_tests").mkdir()
    script = prepare_script(str(src), str(ws), has_setup_patch=True, setup_patch_path=str(root / "setup.patch"),
                            hidden_paths=[])
    script = script.replace("git config --global", "git config")  # don't touch the host's global config
    assert "PREPARE_OK" in subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
    assert (src.is_symlink() and (ws / "calc.py").read_text().endswith("a - b\n")
            and not (ws / "r2e_tests").exists())
    assert sh("git branch -a", ws).split() == ["*", "main"]
    assert sh("git status --porcelain", ws) == "" and "future" not in sh("git log --all -p", ws)
    print("ok test_prepare_script_in_container_layout")


if __name__ == "__main__":
    test_classify_and_shards()
    test_prepare_script_in_container_layout()
    test_solves_and_records()
    test_wrong_fix_not_solved()
    test_no_patch()
    test_run_generation_target_limit_and_resume()
    print("all generate tests passed")
