"""Remove recalled project history from solved trajectories.

Kimi often "remembers" how the project really fixed an issue (commits, PR numbers, "pandas main
today") inside its reasoning, even when told not to. The student cannot do that. Only the
reasoning text of the affected turns is changed; tool calls, tool results and the patch stay
exactly as they were, so the trajectory still matches what happened in the container.

For each `thinking` / `text` / `final` entry with recall phrases (see rollout.RECALL_RE):
  1. a cheap rewrite request deletes the recalled claims and keeps the technical idea as the
     agent's own proposal ("One way to fix this is ..."), adding nothing new;
  2. the result is checked: no recall phrases left (references that appear in the issue text
     are allowed) and not longer than the original;
  3. after two failed rewrites the entry's text is dropped (the tool call of that turn stays).

    python -m codeagent_data clean-recall --generated out/generated --out out/cleaned
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..jsonl import iter_jsonl
from .rollout import RECALL_LIMIT, RECALL_RE

TEXT_TYPES = ("thinking", "text", "final")

REWRITE_SYSTEM = """You edit the private reasoning of an AI coding agent that is fixing an issue in a
repository. The reasoning must rely only on what the agent could observe in its session: the issue
text, the code and files it read, and the output of the commands it ran.

Remove every part that comes from remembering the project's real history instead: claims about the
upstream / actual / real / official fix, commits and hashes, pull request, issue or ticket numbers
that do not appear in the issue text, release notes, versions in which something was fixed, what
"current main" or later versions contain, and phrases such as "let me recall" or "I remember".

When a removed passage carries a concrete technical idea (for example the code of a fix), keep the
idea but state it as the agent's own proposal ("One way to fix this is ...", "A fix could ...")
without saying that the project did it this way. Do not add new facts, code, file names or
conclusions. Keep everything else word for word, in the same style. If nothing is left, output an
empty line.

Output only the edited reasoning text."""


@dataclass
class CleanConfig:
    model: str = "moonshotai/kimi-k3"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_base: str | None = None
    extra_body: dict[str, Any] | None = None  # e.g. {"reasoning": {"effort": "low"}}
    max_tries: int = 2
    max_growth: float = 1.10  # the rewrite may not be longer than this x the original
    workers: int = 8


def foreign_mentions(text: str, issue: str) -> int:
    """Recall phrases in `text`, not counting references that also appear in the issue text."""
    issue_l = (issue or "").lower()
    return sum(1 for m in RECALL_RE.finditer(text or "") if m.group(0).lower() not in issue_l)


def litellm_rewriter(cfg: CleanConfig) -> Callable[[str, str], tuple[str, dict[str, int]]]:
    import litellm

    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise SystemExit(f"set {cfg.api_key_env} (the OpenRouter API key)")

    def rewrite(text: str, issue: str) -> tuple[str, dict[str, int]]:
        kwargs: dict[str, Any] = {"api_key": key, "num_retries": 3, "temperature": 0.2,
                                  "max_tokens": max(1024, len(text) // 2)}
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if cfg.extra_body:
            kwargs["extra_body"] = cfg.extra_body
        r = litellm.completion(model=f"openrouter/{cfg.model}", messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Issue text (references that appear here are allowed):\n"
                                        f"<issue>\n{issue[:6000]}\n</issue>\n\nReasoning to edit:\n"
                                        f"<reasoning>\n{text}\n</reasoning>"}], **kwargs)
        out = (r.choices[0].message.content or "").strip()
        out = re.sub(r"^<reasoning>\s*|\s*</reasoning>$", "", out).strip()
        u = getattr(r, "usage", None)
        return out, {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                     "completion_tokens": getattr(u, "completion_tokens", 0) or 0}

    return rewrite


def clean_trace(trace: dict[str, Any], issue: str, rewrite: Callable[[str, str], tuple[str, dict[str, int]]],
                cfg: CleanConfig) -> dict[str, Any]:
    """Cleans `trace` (legacy format) in place; returns what was done."""
    st = {"entries": 0, "rewritten": 0, "kept_as_is": 0, "dropped": 0, "prompt_tokens": 0,
          "completion_tokens": 0, "tool_args_with_recall": 0}
    entries = trace.get("entries", [])
    # A lone hit ("the actual fix is to ...", an issue number quoted from a code comment) is usually
    # innocent: entries are only rewritten in trajectories that recall history (RECALL_LIMIT hits
    # in total), or when one entry has several hits.
    total = sum(foreign_mentions(e.get("content", ""), issue) for e in entries if e.get("type") in TEXT_TYPES)
    for e in entries:
        if e.get("type") == "tool_call" and foreign_mentions(json.dumps(e.get("args") or {}), issue):
            st["tool_args_with_recall"] += 1  # e.g. "# GH#55845" in an edit: reported, not changed
        hits = foreign_mentions(e.get("content", ""), issue) if e.get("type") in TEXT_TYPES else 0
        if not hits or (total < RECALL_LIMIT and hits < 2):
            continue
        st["entries"] += 1
        original, done = e["content"], False
        for _ in range(cfg.max_tries):
            try:
                new, usage = rewrite(original, issue)
            except Exception as err:  # noqa: BLE001
                e.setdefault("cleaning_errors", []).append(str(err)[:200])
                continue
            st["prompt_tokens"] += usage.get("prompt_tokens", 0)
            st["completion_tokens"] += usage.get("completion_tokens", 0)
            if new.strip() == original.strip() and hits == 1:  # the editor saw nothing to remove
                done = True
                st["kept_as_is"] += 1
                break
            if foreign_mentions(new, issue) == 0 and len(new) <= cfg.max_growth * len(original):
                e["content"], done = new, True
                st["rewritten"] += 1
                break
        if not done:
            e["content"] = ""
            st["dropped"] += 1
        if e["content"] != original:
            e["cleaned_from"] = original  # kept for audit; the SFT parser reads `content`
    return st


def clean_generated(generated: Path, out: Path, cfg: CleanConfig,
                    rewrite: Callable[[str, str], tuple[str, dict[str, int]]] | None = None) -> dict[str, int]:
    """Read generate-merge's output (solved.jsonl + traces/), write a cleaned copy to `out`."""
    rewrite = rewrite or litellm_rewriter(cfg)
    rows = list(iter_jsonl(generated / "solved.jsonl"))
    (out / "traces").mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {"tasks": len(rows), "tasks_cleaned": 0}
    lock = threading.Lock()

    def one(row: dict[str, Any]) -> dict[str, Any]:
        rel = (row.get("teacher") or {}).get("trace")
        if not rel or not (generated / rel).exists():
            return row
        trace = json.loads((generated / rel).read_text(encoding="utf-8"))
        st = clean_trace(trace, row.get("problem_statement") or "", rewrite, cfg)
        (out / rel).write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
        with lock:
            totals["tasks_cleaned"] += bool(st["entries"])
            for k, v in st.items():
                totals[k] = totals.get(k, 0) + v
        return {**row, "teacher": {**row["teacher"], "cleaning": st}}

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        cleaned = list(pool.map(one, rows))
    with (out / "solved.jsonl").open("w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("results_merged.jsonl", "generation_report.md"):
        if (generated / name).exists():
            shutil.copy2(generated / name, out / name)
    report = ["# Recall cleaning", ""] + [f"- {k}: {v}" for k, v in totals.items()]
    (out / "cleaning_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return totals
