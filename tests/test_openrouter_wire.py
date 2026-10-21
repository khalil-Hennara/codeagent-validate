"""The real model path (ADK LiteLlm -> litellm's openrouter provider -> HTTP) against a local fake
OpenRouter server: checks that reasoning comes back as thinking entries, that tool calls work, that
extra_body / reasoning_effort reach the request, and that reasoning is sent back on later turns.

    PYTHONPATH=harness python3 tests/test_openrouter_wire.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

from test_generate import FIX_SCRIPT, TASK, ApplyAndPytestRunner, toy_sandbox_factory  # noqa: E402

from codeagent_data.generate import rollout  # noqa: E402

REQUESTS: list[dict] = []


class FakeOpenRouter(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(body)
        step = sum(1 for m in body["messages"] if m["role"] == "tool")
        name, args = FIX_SCRIPT[min(step, len(FIX_SCRIPT) - 1)]
        msg = {"role": "assistant", "content": None, "reasoning": f"REASONING step {step}: I should call {name}."}
        if name == "final":
            msg["content"] = "Fixed add()."
        else:
            msg["tool_calls"] = [{"id": f"call_{step}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
        resp = {"id": f"gen-{step}", "object": "chat.completion", "created": 1, "model": "moonshotai/kimi-k3",
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "stop" if name == "final" else "tool_calls"}],
                "usage": {"prompt_tokens": 2000 + 100 * step, "completion_tokens": 300, "total_tokens": 2300,
                          "prompt_tokens_details": {"cached_tokens": 1500}, "cost": 0.004}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["OPENROUTER_API_KEY"] = "sk-test"
    cfg = rollout.GenConfig(time_minutes=5, tool_calls=20, turns=30, api_base=f"http://127.0.0.1:{srv.server_port}",
                            reasoning_effort="high", extra_body={"provider": {"sort": "price"}},
                            price_in=0.6, price_out=2.5, price_cached=0.15)
    out = Path(tempfile.mkdtemp())
    a = rollout.run_one(TASK, cfg, out, ApplyAndPytestRunner(), sandbox_factory=toy_sandbox_factory)
    srv.shutdown()
    assert a.solved, (a.reason, a.agent_error)
    first = REQUESTS[0]
    print("request keys:", sorted(first))
    assert first["model"] == "moonshotai/kimi-k3", first["model"]
    assert {t["function"]["name"] for t in first["tools"]} == {
        "run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status"}, first["tools"]
    entries = json.loads((out / a.trace_path).read_text())["entries"]
    thinking = [e["content"] for e in entries if e["type"] == "thinking"]
    assert len(thinking) == 5 and thinking[0].startswith("REASONING step 0"), thinking
    sent_back = [m for m in REQUESTS[-1]["messages"] if m["role"] == "assistant"]
    print("assistant msg keys on the last request:", sorted(sent_back[0]))
    print("reasoning sent back:", any("REASONING" in json.dumps(m) for m in sent_back))
    assert first.get("reasoning") == {"effort": "high"} and first.get("provider") == {"sort": "price"}, first
    assert any("REASONING step 0" in json.dumps(m) for m in sent_back), "reasoning must be sent back"
    print("stats:", a.stats, " est $", a.est_cost_usd)
    print("ok test_openrouter_wire")


if __name__ == "__main__":
    main()
