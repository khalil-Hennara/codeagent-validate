"""clean-recall: rewriting recalled project history out of solved trajectories.

Uses a real recall-heavy trace from the pilot (tests/data/recall_trace_pandas_48696.json), a
deterministic stand-in for the rewrite model, and once the real litellm path against a local fake
OpenRouter server.

    PYTHONPATH=harness python3 tests/test_clean.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

from codeagent_data.generate.clean import CleanConfig, clean_generated, clean_trace, foreign_mentions  # noqa: E402
from codeagent_data.generate.rollout import RECALL_RE  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "recall_trace_pandas_48696.json"


def drop_recall_sentences(text: str, issue: str) -> tuple[str, dict[str, int]]:
    """Stand-in for the rewrite model: removes every sentence that contains a recall phrase."""
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    kept = [p for p in parts if not RECALL_RE.search(p)]
    return "\n".join(kept).strip(), {"prompt_tokens": len(text) // 4, "completion_tokens": 10}


def echo(text: str, issue: str) -> tuple[str, dict[str, int]]:
    return text, {}


def make_generated(tmp: Path) -> Path:
    trace = json.loads(FIXTURE.read_text())
    issue = next(e["content"] for e in trace["entries"] if e["type"] == "task_prompt")
    gen = tmp / "generated"
    (gen / "traces").mkdir(parents=True)
    (gen / "traces" / "t__a1.json").write_text(json.dumps(trace))
    row = {"instance_id": "pandas-dev__pandas-48696", "problem_statement": issue,
           "teacher": {"trace": "traces/t__a1.json", "patch": "diff ..."}}
    (gen / "solved.jsonl").write_text(json.dumps(row) + "\n")
    return gen


def test_real_trace_gets_clean_and_actions_stay():
    tmp = Path(tempfile.mkdtemp())
    gen = make_generated(tmp)
    before = json.loads((gen / "traces" / "t__a1.json").read_text())
    issue = json.loads((gen / "solved.jsonl").read_text())["problem_statement"]
    assert sum(foreign_mentions(e.get("content", ""), issue) for e in before["entries"]) >= 10
    tot = clean_generated(gen, tmp / "clean", CleanConfig(workers=1), rewrite=drop_recall_sentences)
    after = json.loads((tmp / "clean" / "traces" / "t__a1.json").read_text())
    texts = [e.get("content", "") for e in after["entries"] if e["type"] in ("thinking", "text", "final")]
    assert sum(foreign_mentions(t, issue) for t in texts) == 0
    # tool calls untouched, in the same order
    calls = lambda tr: [json.dumps(e.get("args"), sort_keys=True) for e in tr["entries"] if e["type"] == "tool_call"]  # noqa: E731
    assert calls(after) == calls(before)
    assert tot["rewritten"] >= 3 and tot["dropped"] == 0, tot
    assert any("cleaned_from" in e for e in after["entries"])
    row = json.loads((tmp / "clean" / "solved.jsonl").read_text())
    assert row["teacher"]["cleaning"]["rewritten"] == tot["rewritten"]
    print("ok test_real_trace_gets_clean_and_actions_stay", tot)


def test_failed_rewrites_drop_the_text_and_lone_hits_stay():
    trace = {"entries": [
        {"type": "thinking", "content": "The actual upstream fix (PR #48745) added a bool branch. "
                                        "Let me recall commit 6d42d5e1a2e9."},
        {"type": "tool_call", "tool": "edit_file", "args": {"new_string": "# GH#55845\nx = 1"}},
        {"type": "thinking", "content": "Read the function first."}]}
    st = clean_trace(trace, "issue text", echo, CleanConfig())  # the model returns the text unchanged
    assert trace["entries"][0]["content"] == "" and st["dropped"] == 1 and st["tool_args_with_recall"] == 1
    assert trace["entries"][2]["content"] == "Read the function first."
    # a trajectory with a single innocent hit is not touched at all
    lone = {"entries": [{"type": "thinking", "content": "So the actual fix is to change line 3."}]}
    st = clean_trace(lone, "issue", echo, CleanConfig())
    assert st["entries"] == 0 and lone["entries"][0]["content"].startswith("So the actual fix")
    # references that are in the issue text itself are allowed
    assert foreign_mentions("as GH#48673 says", "Regression reported in GH#48673") == 0
    print("ok test_failed_rewrites_drop_the_text_and_lone_hits_stay")


class FakeOpenRouter(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][-1]["content"].split("<reasoning>\n", 1)[1].rsplit("\n</reasoning>", 1)[0]
        out, _ = drop_recall_sentences(text, "")
        resp = {"id": "x", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_cli_through_litellm():
    import subprocess

    srv = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp())
    gen = make_generated(tmp)
    env = {**os.environ, "OPENROUTER_API_KEY": "sk", "OPENROUTER_API_BASE": f"http://127.0.0.1:{srv.server_port}",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    r = subprocess.run([sys.executable, "-m", "codeagent_data", "clean-recall", "--generated", str(gen),
                        "--out", str(tmp / "c"), "--workers", "2"], env=env, capture_output=True, text=True)
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-2000:]
    assert "- dropped: 0" in (tmp / "c" / "cleaning_report.md").read_text(), r.stdout
    print("ok test_cli_through_litellm")


if __name__ == "__main__":
    test_failed_rewrites_drop_the_text_and_lone_hits_stay()
    test_real_trace_gets_clean_and_actions_stay()
    test_cli_through_litellm()
