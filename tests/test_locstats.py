"""loc-stats: localization profiles from traces. Run: python3 tests/test_locstats.py"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codeagent_data.generate.locstats import gold_locations, profile  # noqa: E402

GOLD = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -120,7 +120,7 @@ def compute(x):
-    return x - 1
+    return x + 1
diff --git a/doc/whatsnew/v1.rst b/doc/whatsnew/v1.rst
--- a/doc/whatsnew/v1.rst
+++ b/doc/whatsnew/v1.rst
@@ -1,1 +1,2 @@
+- fixed compute
"""


def call(tool, args, result=""):
    return [{"type": "tool_call", "tool": tool, "args": args},
            {"type": "tool_response", "tool": tool, "result": result}]


def trace(*steps):
    return {"entries": [e for s in steps for e in s]}


def test_paths():
    gold = gold_locations(GOLD)
    assert gold == {"pkg/core.py": [(120, 126)]}, gold  # docs are not a localization target
    direct = trace(call("run_command", {"command": "grep -rn 'def compute' /workspace/pkg"}, "pkg/core.py:118:def compute"),
                   call("read_file", {"filepath": "pkg/core.py", "start_line": 110, "end_line": 140},
                        json.dumps({"status": "ok", "start_line": 110, "end_line": 140})),
                   call("edit_file", {"filepath": "/workspace/pkg/core.py"}))
    p = profile(direct, gold, {"instance_id": "a", "solved": True})
    assert (p.path, p.first_gold_discover, p.first_gold_open, p.first_gold_lines, p.first_edit_in_gold) == \
        ("direct", 1, 2, 2, True), p
    wander = trace(*[call("run_command", {"command": f"sed -n 1,80p /workspace/pkg/mod{i}.py"}) for i in range(5)],
                   call("run_command", {"command": "grep -rn compute pkg"}),
                   call("run_command", {"command": "cat pkg/core.py"}),
                   call("edit_file", {"filepath": "pkg/core.py"}))
    p = profile(wander, gold, {"instance_id": "b"})
    assert p.path == "struggled" and len(p.detour_files) == 5 and p.first_gold_lines == 7 and p.searches_before_gold == 1, p
    lost = trace(call("read_file", {"filepath": "pkg/other.py"}, "{}"), call("edit_file", {"filepath": "pkg/other.py"}))
    p = profile(lost, gold, {"instance_id": "c"})
    assert p.path == "never_found" and p.non_gold_edits == 1 and p.first_edit_in_gold is False, p
    tests_only = trace(call("read_file", {"filepath": "pkg/tests/test_core.py"}, "{}"),
                       call("read_file", {"filepath": "pkg/core.py"}, json.dumps({"start_line": 1, "end_line": 150})))
    p = profile(tests_only, gold, {"instance_id": "d"})
    assert p.path == "direct" and p.detour_files == [] and p.first_gold_lines == 2, p  # reading tests is not a detour
    print("ok test_paths")


def test_cli():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "traces").mkdir()
    (tmp / "traces" / "a__a1.json").write_text(json.dumps(trace(call("read_file", {"filepath": "pkg/core.py"},
                                                                       json.dumps({"start_line": 100, "end_line": 130})))))
    (tmp / "results.jsonl").write_text(json.dumps({"instance_id": "a", "attempt": 1, "solved": True, "verifier": "swe_gym",
                                                   "trace_path": "traces/a__a1.json"}) + "\n"
                                       + json.dumps({"instance_id": "b", "attempt": 1, "solved": False,
                                                     "trace_path": None}) + "\n")
    (tmp / "tasks.jsonl").write_text(json.dumps({"instance_id": "a", "patch": GOLD}) + "\n")
    r = subprocess.run([sys.executable, "-m", "codeagent_data", "loc-stats", "--tasks", str(tmp / "tasks.jsonl"),
                        "--results", str(tmp / "results.jsonl"), "--out", str(tmp / "out")],
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in (tmp / "out" / "loc_profiles.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["path"] == "direct" and "trajectories: 1 (no trace or task row: 1)" in r.stdout
    print("ok test_cli")


if __name__ == "__main__":
    test_paths()
    test_cli()
