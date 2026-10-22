"""How the teacher found the code to change: a localization profile for every trajectory.

The "gold" locations come from the task's reference fix: its source files (docs and changelogs
excluded) and the line ranges of its hunks. Each tool call of a trace is turned into events:

  search    grep / rg / find / git grep / ack in run_command
  discover  a gold file's path appears in a tool's output (e.g. a grep hit)
  open      the agent reads a file: read_file, or cat / sed -n / head / tail / nl in run_command
  hit_lines the read range overlaps a gold hunk (it looked at the exact lines that change)
  edit      edit_file / write_file (or sed -i / cat > on a repo file)

and summarized per trajectory:

  first_gold_open      tool-call index (1-based) of the first read of a gold file (None = never)
  first_gold_lines     ... of the first read that covers a gold hunk
  detour_files         distinct non-gold source files opened before the first gold open
  searches_before_gold search commands before the first gold open
  calls_before_gold    tool calls before the first gold open
  first_edit_in_gold   the first edit went to a gold file
  non_gold_edits       edits to source files outside the reference fix
  path                 direct | searched | struggled | never_found  (thresholds below, adjustable)

    python -m codeagent_data loc-stats --tasks data/gen/*.jsonl --results ledger/results.jsonl \\
        [--trace-dirs shards/*] --out out/loc
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..jsonl import iter_jsonl
from ..patches import is_test_path

SEARCH_RE = re.compile(r"(?:^|[\s;&|(])(?:grep|egrep|rg|ag|ack|find|git\s+grep|git\s+log\s+-S)\b")
PATH_RE = re.compile(r"(?:/workspace/|\./)?((?:[\w.-]+/)*[\w.-]+\.(?:py|pyx|pxd|pyi|c|h|cpp|cc|js|ts|cfg|toml|ini))\b")
SED_RANGE_RE = re.compile(r"sed\s+-n\s+['\"]?(\d+)\s*,\s*(\d+)\s*p['\"]?\s+((?:/workspace/)?[\w./-]+)")
HEAD_RE = re.compile(r"\bhead\s+(?:-n\s*|-)(\d+)\s+((?:/workspace/)?[\w./-]+)")
OPEN_CMD_RE = re.compile(r"\b(?:cat|nl|less|more|head|tail)\s+(?:-\S+\s+)*((?:/workspace/)?[\w./-]+\.\w+)")
WRITE_CMD_RE = re.compile(r"(?:sed\s+-i\S*\s+.*?\s|>\s*)((?:/workspace/)?[\w./-]+\.py)\b")
DOC_RE = re.compile(r"(?i)(^|/)(docs?|changelog|changes|news|whatsnew|release[-_]?notes?)(/|\.|$)|\.(rst|md|txt)$")


def norm(path: str) -> str:
    p = (path or "").strip().strip("'\"")
    for prefix in ("/workspace/", "/testbed/", "./"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p


def is_source(path: str) -> bool:
    return bool(path) and not path.startswith(("/tmp", "tmp/", "/")) and not is_test_path(path) \
        and not DOC_RE.search(path)


def gold_locations(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Source files of a unified diff -> line ranges (old side) of its hunks."""
    out: dict[str, list[tuple[int, int]]] = {}
    cur = None
    for line in (patch or "").split("\n"):
        if line.startswith("+++ "):
            p = line[4:].split("\t")[0].strip()
            p = p[2:] if p.startswith("b/") else p
            cur = norm(p) if p != "/dev/null" and is_source(norm(p)) else None
            if cur:
                out.setdefault(cur, [])
        elif line.startswith("@@") and cur:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? ", line)
            if m:
                start, n = int(m.group(1)), int(m.group(2) or 1)
                out[cur].append((start, start + max(n, 1) - 1))
    return out


@dataclass
class LocProfile:
    instance_id: str
    attempt: int = 1
    solved: bool = False
    verifier: str = ""
    gold_files: list[str] = field(default_factory=list)
    n_calls: int = 0
    first_gold_discover: int | None = None
    first_gold_open: int | None = None
    first_gold_lines: int | None = None
    first_edit: int | None = None
    first_edit_in_gold: bool | None = None
    calls_before_gold: int | None = None
    searches_before_gold: int = 0
    detour_files: list[str] = field(default_factory=list)
    non_gold_edits: int = 0
    gold_files_opened: int = 0
    path: str = ""


def _overlaps(rng: tuple[int, int], hunks: list[tuple[int, int]]) -> bool:
    return any(rng[0] <= b and a <= rng[1] for a, b in hunks)


def tool_events(call: dict[str, Any], result: str) -> list[tuple[str, str, tuple[int, int] | None]]:
    """(kind, path, line range or None) for one tool call."""
    tool, args = call.get("tool") or "", call.get("args") or {}
    ev: list[tuple[str, str, tuple[int, int] | None]] = []
    if tool == "read_file":
        path = norm(args.get("filepath", ""))
        rng = None
        try:
            r = json.loads(result or "{}")
            if r.get("start_line"):
                rng = (int(r["start_line"]), int(r.get("end_line") or r["start_line"]))
        except (ValueError, TypeError):
            pass
        if rng is None:
            s, e = args.get("start_line"), args.get("end_line")
            rng = (int(s or 1), int(e or (int(s or 1) + 149)))
        ev.append(("open", path, rng))
    elif tool in ("edit_file", "write_file"):
        ev.append(("edit", norm(args.get("filepath", "")), None))
    elif tool == "run_command":
        cmd = args.get("command", "") or ""
        if SEARCH_RE.search(cmd):
            ev.append(("search", "", None))
        for m in SED_RANGE_RE.finditer(cmd):
            ev.append(("open", norm(m.group(3)), (int(m.group(1)), int(m.group(2)))))
        for m in HEAD_RE.finditer(cmd):
            ev.append(("open", norm(m.group(2)), (1, int(m.group(1)))))
        for m in OPEN_CMD_RE.finditer(cmd):
            p = norm(m.group(1))
            if not any(e[1] == p for e in ev):
                ev.append(("open", p, None))
        for m in WRITE_CMD_RE.finditer(cmd):
            p = norm(m.group(1))
            if is_source(p):
                ev.append(("edit", p, None))
    return ev


def profile(trace: dict[str, Any], gold: dict[str, list[tuple[int, int]]], meta: dict[str, Any],
            direct_max_calls: int = 6, direct_max_detours: int = 1, struggle_min_detours: int = 4,
            struggle_min_calls: int = 15) -> LocProfile:
    p = LocProfile(meta["instance_id"], int(meta.get("attempt") or 1), bool(meta.get("solved")),
                   meta.get("verifier") or "", sorted(gold))
    entries = trace.get("entries", [])
    calls = [e for e in entries if e.get("type") == "tool_call"]
    results = [e for e in entries if e.get("type") == "tool_response"]
    p.n_calls = len(calls)
    opened_gold: set[str] = set()
    detours: list[str] = []
    for i, call in enumerate(calls, start=1):
        result = (results[i - 1].get("result") or "") if i - 1 < len(results) else ""
        if p.first_gold_discover is None and any(g in result for g in gold):
            p.first_gold_discover = i
        for kind, path, rng in tool_events(call, result):
            if kind == "search" and p.first_gold_open is None:
                p.searches_before_gold += 1
            elif kind == "open":
                if path in gold:
                    opened_gold.add(path)
                    if p.first_gold_open is None:
                        p.first_gold_open = i
                        p.calls_before_gold = i - 1
                        p.first_gold_discover = p.first_gold_discover or i
                    # no range = the whole file was printed (cat / nl)
                    if p.first_gold_lines is None and (rng is None or _overlaps(rng, gold[path])):
                        p.first_gold_lines = i
                elif p.first_gold_open is None and is_source(path) and path not in detours:
                    detours.append(path)
            elif kind == "edit":
                if p.first_edit is None:
                    p.first_edit = i
                    p.first_edit_in_gold = path in gold
                if is_source(path) and path not in gold:
                    p.non_gold_edits += 1
    p.detour_files = detours
    p.gold_files_opened = len(opened_gold)
    if not gold:
        p.path = "no_gold_source"
    elif p.first_gold_open is None:
        p.path = "never_found"
    elif (p.calls_before_gold or 0) < direct_max_calls and len(detours) <= direct_max_detours:
        p.path = "direct"
    elif len(detours) >= struggle_min_detours or (p.calls_before_gold or 0) >= struggle_min_calls \
            or p.first_edit_in_gold is False:
        p.path = "struggled"
    else:
        p.path = "searched"
    return p


def find_trace(row: dict[str, Any], dirs: Iterable[Path]) -> Path | None:
    rel = row.get("trace_path")
    if not rel:
        return None
    for d in dirs:
        if (d / rel).exists():
            return d / rel
        if (d / "traces" / Path(rel).name).exists():
            return d / "traces" / Path(rel).name
    return None


def run_loc_stats(tasks: list[dict[str, Any]], result_files: list[Path], trace_dirs: list[Path],
                  out: Path) -> dict[str, Any]:
    by_id = {t["instance_id"]: t for t in tasks}
    rows: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    for fp in result_files:
        for r in iter_jsonl(fp):
            rows[(r["instance_id"], int(r.get("attempt") or 1))] = (r, fp.parent)
    out.mkdir(parents=True, exist_ok=True)
    profiles: list[LocProfile] = []
    missing = 0
    for (iid, _k), (r, base) in sorted(rows.items()):
        t = by_id.get(iid)
        tr = find_trace(r, [base] + trace_dirs)
        if t is None or tr is None:
            missing += 1
            continue
        trace = json.loads(tr.read_text(encoding="utf-8"))
        profiles.append(profile(trace, gold_locations(t.get("patch") or ""), r))
    with (out / "loc_profiles.jsonl").open("w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    report = render_loc_report(profiles, missing)
    (out / "loc_report.md").write_text(report, encoding="utf-8")
    print(report)
    return {"profiles": len(profiles), "missing_trace_or_task": missing}


def render_loc_report(profiles: list[LocProfile], missing: int = 0) -> str:
    lines = ["# Localization profiles", "",
             f"trajectories: {len(profiles)} (no trace or task row: {missing})", ""]
    groups = {"solved": [p for p in profiles if p.solved], "unsolved": [p for p in profiles if not p.solved]}
    order = ["direct", "searched", "struggled", "never_found", "no_gold_source"]
    lines += ["| path | " + " | ".join(groups) + " |", "| --- |" + " --- |" * len(groups)]
    for k in order:
        lines.append(f"| {k} | " + " | ".join(str(sum(p.path == k for p in g)) for g in groups.values()) + " |")

    def med(xs: list[float]) -> str:
        return f"{statistics.median(xs):.0f}" if xs else "-"

    for name, g in groups.items():
        found = [p for p in g if p.first_gold_open]
        if not g:
            continue
        lines += ["", f"## {name}", "",
                  f"- calls before opening a gold file: median {med([p.calls_before_gold or 0 for p in found])}",
                  f"- detour files before it: median {med([len(p.detour_files) for p in found])}, "
                  f"4 or more: {sum(len(p.detour_files) >= 4 for p in found)}",
                  f"- looked at the exact changed lines: {sum(p.first_gold_lines is not None for p in g)} of {len(g)}",
                  f"- first edit outside the reference fix: {sum(p.first_edit_in_gold is False for p in g)}"]
    by_src: dict[str, Counter] = {}
    for p in profiles:
        by_src.setdefault(p.verifier or "?", Counter())[p.path] += 1
    lines += ["", "## By source", "", "| source | " + " | ".join(order[:4]) + " |", "| --- |" + " --- |" * 4]
    for s, c in sorted(by_src.items()):
        lines.append(f"| {s} | " + " | ".join(str(c[k]) for k in order[:4]) + " |")
    return "\n".join(lines) + "\n"
