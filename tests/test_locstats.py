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
    g = profile(direct, gold, {"instance_id": "a"}).gold
    # a recursive grep hit is a discovery; reading the file is the first open
    assert (g.path, g.first_discover, g.first_open, g.first_lines, g.first_edit_inside) == \
        ("direct", 1, 2, 2, True), g
    wrong_first = trace(call("read_file", {"filepath": "pkg/core.py"}, json.dumps({"start_line": 110, "end_line": 140})),
                        call("edit_file", {"filepath": "pkg/other.py"}), call("edit_file", {"filepath": "pkg/core.py"}))
    assert profile(wrong_first, gold, {"instance_id": "w"}).gold.path == "struggled"
    wander = trace(*[call("run_command", {"command": f"sed -n 1,80p /workspace/pkg/mod{i}.py"}) for i in range(5)],
                   call("run_command", {"command": "ls pkg"}),
                   call("run_command", {"command": "cat pkg/core.py"}),
                   call("edit_file", {"filepath": "pkg/core.py"}))
    g = profile(wander, gold, {"instance_id": "b"}).gold
    assert g.path == "struggled" and len(g.detour_files) == 5 and g.first_lines == 7, g
    lost = trace(call("read_file", {"filepath": "pkg/other.py"}, "{}"), call("edit_file", {"filepath": "pkg/other.py"}))
    g = profile(lost, gold, {"instance_id": "c"}).gold
    assert g.path == "never_found" and g.edits_outside == 1 and g.first_edit_inside is False, g
    tests_only = trace(call("read_file", {"filepath": "pkg/tests/test_core.py"}, "{}"),
                       call("read_file", {"filepath": "pkg/core.py"}, json.dumps({"start_line": 1, "end_line": 150})))
    g = profile(tests_only, gold, {"instance_id": "d"}).gold
    assert g.path == "direct" and g.detour_files == [] and g.first_lines == 2, g  # reading tests is not a detour
    # grep -n -A on the file shows the changed lines; relative path after `cd`
    peek = trace(call("run_command", {"command": "grep -n 'return x' -A3 /workspace/pkg/core.py"},
                      json.dumps({"stdout": "121:    return x - 1\n122-\n"})),
                 call("run_command", {"command": "cd /workspace/pkg && sed -n 100,130p core.py"}))
    g = profile(peek, gold, {"instance_id": "e"}).gold
    assert g.first_open == 1 and g.first_lines == 1, g
    g = profile(trace(call("run_command", {"command": "cd /workspace/pkg && sed -n 100,130p core.py"})), gold,
                {"instance_id": "f"}).gold
    assert g.first_open == 1 and g.first_lines == 1, g
    # edited straight from search output: reached
    g = profile(trace(call("edit_file", {"filepath": "pkg/core.py"})), gold, {"instance_id": "g"}).gold
    assert g.path == "direct" and g.first_edit_inside, g
    # solved somewhere else: never found the reference fix, but direct to its own (verified) fix
    own_patch = GOLD.replace("pkg/core.py", "pkg/helpers.py")
    p = profile(lost := trace(call("read_file", {"filepath": "pkg/helpers.py"}, "{}"),
                              call("edit_file", {"filepath": "pkg/helpers.py"})),
                gold, {"instance_id": "h", "solved": True, "patch": own_patch})
    assert p.gold.path == "never_found" and p.own.path == "direct" and p.fixed_elsewhere and p.path == "direct", p
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
    assert len(rows) == 1 and rows[0]["path"] == "direct" and "skipped (no trace or task row): 1" in r.stdout, r.stdout
    print("ok test_cli")


if __name__ == "__main__":
    test_paths()
    test_cli()
