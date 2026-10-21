"""The generation ledger: one permanent record of every task the teacher has been run on.

The workflow keeps it on the `gen-ledger` branch of this repository and updates it at the end of
every run (also when jobs were cancelled, timed out or ran out of credit, because each shard
writes results.jsonl as it goes):

    results.jsonl   one row per attempt: task, try number, outcome, tokens, cost estimate,
                    patch, test progress, localization; `trace_path` for solved attempts
    traces/         the trace of every solved attempt (unsolved traces stay in the run's artifacts)
    README.md       counts

Every run reads it first (`generate --prior <ledger>/results.jsonl`): a task with a finished
attempt is never started again, solved or not. Only attempts that ended in an infrastructure or
API error (the model did not finish: IMAGE_UNAVAILABLE, INFRA_ERROR, PREPARE_FAILED, API_ERROR,
FATAL_API_ERROR such as running out of credit) are tried again.

    python -m codeagent_data ledger-update --ledger ledger --results shards/*/results.jsonl
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from ..jsonl import iter_jsonl
from .rollout import is_final


def update_ledger(ledger: Path, result_files: list[Path]) -> dict[str, int]:
    ledger.mkdir(parents=True, exist_ok=True)
    (ledger / "traces").mkdir(exist_ok=True)
    fp = ledger / "results.jsonl"
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if fp.exists():
        for r in iter_jsonl(fp):
            rows[(r["instance_id"], int(r.get("attempt") or 1))] = r
    added = 0
    for src in result_files:
        if not src.exists():
            continue
        for r in iter_jsonl(src):
            key = (r["instance_id"], int(r.get("attempt") or 1))
            old = rows.get(key)
            if old is not None and is_final(old) and not is_final(r):
                continue  # a finished attempt is never replaced by a later infrastructure failure
            if old is not None and is_final(old) and is_final(r) and old.get("solved") and not r.get("solved"):
                continue
            r = dict(r)
            trace = None
            if r.get("solved") and r.get("trace_path") and (src.parent / r["trace_path"]).exists():
                name = Path(r["trace_path"]).name
                shutil.copy2(src.parent / r["trace_path"], ledger / "traces" / name)
                trace = f"traces/{name}"
            elif old is not None and old.get("trace_path") and (ledger / old["trace_path"]).exists():
                trace = old["trace_path"]
            r["trace_path"] = trace
            added += old is None or old != r
            rows[key] = r
    with fp.open("w", encoding="utf-8") as f:
        for key in sorted(rows):
            f.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    final = [r for r in rows.values() if is_final(r)]
    tasks = {r["instance_id"] for r in final}
    solved = {r["instance_id"] for r in final if r.get("solved")}
    retry = {r["instance_id"] for r in rows.values() if not is_final(r)} - tasks
    by_src = Counter(r.get("verifier") or "?" for r in final if r.get("solved"))
    cost = sum(float(r.get("est_cost_usd") or 0) for r in rows.values())
    st = {"attempts": len(rows), "new_or_changed": added, "tasks_finished": len(tasks),
          "tasks_solved": len(solved), "tasks_to_retry": len(retry), "est_cost_usd": round(cost, 2)}
    lines = ["# Generation ledger", "",
             "Every task the teacher was run on. `generate --prior results.jsonl` skips the finished ones.", ""]
    lines += [f"- {k}: {v}" for k, v in st.items()]
    lines += ["", "solved tasks by source:", ""] + [f"- {k}: {v}" for k, v in sorted(by_src.items())]
    (ledger / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return st
