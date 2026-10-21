"""Stage 4 end-to-end on real Docker: ImageTaskSandbox + DockerRunner verification + the real
LiteLlm/OpenRouter wire path against a local fake server. Needs a local image `toy/calc:1`
(see tests/toy_image/) and a Docker daemon; skipped otherwise.

    PYTHONPATH=harness python3 tests/test_generate_docker.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

import test_openrouter_wire as wire  # noqa: E402
from test_generate import TASK  # noqa: E402

from codeagent_data.generate import rollout  # noqa: E402
from codeagent_data.validate.docker_runner import DockerRunner  # noqa: E402

PROBE = ("git log --all --oneline; git tag; git branch -a; ls /r2e_tests; readlink /testbed; "
         "pwd; git status --porcelain; git log -1 --format=%s; "
         "cd /tmp && python3 -c 'import calc; print(\"IMPORTED_FROM\", calc.__file__)'; "
         "cat $(python3 -c 'import site; print(site.getsitepackages()[0])')/toy_editable.pth")
SCRIPT = [("run_command", {"command": PROBE})] + wire.FIX_SCRIPT


def main():
    try:
        import docker
        docker.from_env().images.get("toy/calc:1")
    except Exception as e:  # noqa: BLE001
        print(f"skipped (no docker or no toy/calc:1 image): {e}")
        return
    wire.FIX_SCRIPT[:] = SCRIPT
    srv = HTTPServer(("127.0.0.1", 0), wire.FakeOpenRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["OPENROUTER_API_KEY"] = "sk-test"
    # the container has no network; the fake model server runs on the host, outside it
    cfg = rollout.GenConfig(time_minutes=5, tool_calls=20, turns=30, api_base=f"http://127.0.0.1:{srv.server_port}",
                            price_in=0.6, price_out=2.5)
    task = {**TASK, "env_image": "toy/calc:1",
            "setup_patch": ""}
    runner = DockerRunner(pull=False, network=False, platform=None, workers=1)
    out = Path(tempfile.mkdtemp())
    a = rollout.run_one(task, cfg, out, runner)
    srv.shutdown()
    entries = json.loads((out / a.trace_path).read_text())["entries"] if a.trace_path else []
    probe = next((e.get("result", "") for e in entries if e["type"] == "tool_response"), "")
    print("probe output seen by the agent:\n", json.loads(probe).get("stdout") if probe else a.agent_error)
    assert a.solved, (a.reason, a.agent_error)
    assert "FUTURE" not in probe and "future fix" not in probe and "v9.9" not in probe, probe
    assert "test_hidden" not in probe and "upstream" not in probe, probe
    out_txt = json.loads(probe)["stdout"]
    assert "IMPORTED_FROM /workspace/calc.py" in out_txt, out_txt  # tracebacks show /workspace
    assert "/opt/miniconda3/envs/testbed/lib/extra" in out_txt, out_txt  # env paths untouched
    print("ok test_generate_docker", a.reason, a.stats, f"{a.seconds:.1f}s")
    shutil.rmtree(out, ignore_errors=True)

    # SWE-smith style: the bug comes from setup_patch and is folded into the base commit
    wire.REQUESTS.clear()
    srv = HTTPServer(("127.0.0.1", 0), wire.FakeOpenRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg.api_base = f"http://127.0.0.1:{srv.server_port}"
    smith = {**task, "instance_id": "toy__calc-smith",
             "setup_patch": "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n"
                            "-    return a - b\n+    # smith bug\n+    return a - b\n"}
    wire.FIX_SCRIPT[:] = [SCRIPT[0], ("edit_file", {"filepath": "calc.py", "old_string": "    # smith bug\n    return a - b",
                                                    "new_string": "    return a + b"})] + SCRIPT[3:]
    out = Path(tempfile.mkdtemp())
    a = rollout.run_one(smith, cfg, out, runner)
    srv.shutdown()
    entries = json.loads((out / a.trace_path).read_text())["entries"]
    probe = json.loads(next(e["result"] for e in entries if e["type"] == "tool_response"))["stdout"]
    assert a.solved, (a.reason, a.agent_error, a.patch)
    log = [line.split(" ", 1)[1] for line in probe.splitlines()[:3]]
    assert log == ["baseline", "refactor (introduces bug)", "add calc"], probe  # bug folded in, no extra commit
    print("ok test_generate_docker (setup patch)", a.reason)


if __name__ == "__main__":
    main()
