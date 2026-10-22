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

Two references are profiled: the task's reference fix (`gold`) and, for solved attempts, the
teacher's own verified patch (`own`). A solved attempt can be correct while fixing a different
place than the reference fix, so its `path` is measured against its own patch. A grep on a file
that prints its lines (grep -n, -A/-B/-C) counts as looking at those lines.

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
GREP_LINE_RE = re.compile(r"^(?:((?:/workspace/)?[\w./-]+\.\w+)[:-])?(\d+)[:-]", re.M)
CD_RE = re.compile(r"\bcd\s+(/workspace/[\w./-]+|/testbed/[\w./-]+)")
SED_PATTERN_RE = re.compile(r"sed\s+-n\s+['\"]/.*?p['\"]\s+((?:/workspace/)?[\w./-]+)")
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
class Reach:
    """How the agent reached one reference location set (the reference fix, or its own final patch)."""
    files: list[str] = field(default_factory=list)
    first_discover: int | None = None
    first_open: int | None = None
    first_lines: int | None = None
    calls_before: int | None = None
    searches_before: int = 0
    detour_files: list[str] = field(default_factory=list)
    first_edit_inside: bool | None = None
    edits_outside: int = 0
    path: str = ""


@dataclass
class LocProfile:
    instance_id: str
    attempt: int = 1
    solved: bool = False
    verifier: str = ""
    n_calls: int = 0
    # against the task's reference fix
    gold: Reach = field(default_factory=Reach)
    # against the teacher's own verified patch (solved attempts): the place it actually fixed
    own: Reach | None = None
    fixed_elsewhere: bool | None = None  # solved, but no file in common with the reference fix
    path: str = ""  # own.path for solved attempts, gold.path otherwise


def _overlaps(rng: tuple[int, int], hunks: list[tuple[int, int]]) -> bool:
    return any(rng[0] <= b and a <= rng[1] for a, b in hunks)


def tool_events(call: dict[str, Any], result: str) -> list[tuple[str, str, tuple[int, int] | None]]:
    """(kind, path, line range or None = whole file) for one tool call."""
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
        cd = CD_RE.search(cmd)
        base = norm(cd.group(1)).rstrip("/") + "/" if cd and norm(cd.group(1)) else ""

        def rel(path: str) -> str:
            p = norm(path)
            return p if (not base or path.startswith(("/workspace/", "/testbed/")) or p.startswith(base)) else base + p

        is_search = bool(SEARCH_RE.search(cmd))
        if is_search:
            ev.append(("search", "", None))
            # grep on named files, or grep -rn hits: the matched lines (and -A/-B/-C context) were shown
            try:
                out = json.loads(result or "{}").get("stdout", "") or ""
            except (ValueError, TypeError, AttributeError):
                out = result or ""
            # only files named in the command count (grep -n pat file.py); files that merely show up
            # among recursive hits (grep -rn pat dir) were discovered, not looked at
            named = {rel(m.group(1)) for m in PATH_RE.finditer(cmd)}
            for m in GREP_LINE_RE.finditer(out[:20000]):
                path = rel(m.group(1)) if m.group(1) else (next(iter(named)) if len(named) == 1 else "")
                if path and path in named:
                    n = int(m.group(2))
                    ev.append(("peek", path, (n, n)))
        for m in SED_RANGE_RE.finditer(cmd):
            ev.append(("open", rel(m.group(3)), (int(m.group(1)), int(m.group(2)))))
        for m in SED_PATTERN_RE.finditer(cmd):
            ev.append(("open", rel(m.group(1)), None))
        for m in HEAD_RE.finditer(cmd):
            ev.append(("open", rel(m.group(2)), (1, int(m.group(1)))))
        for m in OPEN_CMD_RE.finditer(cmd):
            p = rel(m.group(1))
            if not any(e[1] == p for e in ev):
                ev.append(("open", p, None))
        for m in WRITE_CMD_RE.finditer(cmd):
            p = rel(m.group(1))
            if is_source(p):
                ev.append(("edit", p, None))
    return ev


def _match(path: str, targets: dict[str, list[tuple[int, int]]]) -> str | None:
    """The target file `path` refers to (exact, or a relative path that ends it)."""
    if path in targets:
        return path
    if path and "/" not in path.lstrip("./"):
        return None  # a bare file name is too ambiguous
    return next((t for t in targets if t.endswith("/" + path)), None)


def reach(calls: list[dict[str, Any]], results: list[dict[str, Any]], target: dict[str, list[tuple[int, int]]],
          direct_max_calls: int, direct_max_detours: int, struggle_min_detours: int, struggle_min_calls: int) -> Reach:
    r = Reach(files=sorted(target))
    detours: list[str] = []
    first_edit_seen = False
    for i, call in enumerate(calls, start=1):
        result = (results[i - 1].get("result") or "") if i - 1 < len(results) else ""
        if r.first_discover is None and any(t in result for t in target):
            r.first_discover = i
        for kind, path, rng in tool_events(call, result):
            hit = _match(path, target) if kind != "search" else None
            if kind == "search" and r.first_open is None:
                r.searches_before += 1
            elif kind in ("open", "peek") and hit:
                if r.first_open is None:
                    r.first_open, r.calls_before = i, i - 1
                    r.first_discover = r.first_discover or i
                # no range = the whole file was printed (cat / nl / sed pattern)
                if r.first_lines is None and (rng is None or _overlaps(rng, target[hit])):
                    r.first_lines = i
            elif kind == "open" and r.first_open is None and is_source(path) and path not in detours:
                detours.append(path)
            elif kind == "edit":
                if hit and r.first_open is None:  # edited it straight from search output
                    r.first_open, r.calls_before = i, i - 1
                if not first_edit_seen:
                    first_edit_seen = True
                    r.first_edit_inside = bool(hit)
                if is_source(path) and not hit:
                    r.edits_outside += 1
    r.detour_files = detours
    if not target:
        r.path = "no_target"
    elif r.first_open is None:
        r.path = "never_found"
    elif len(detours) >= struggle_min_detours or (r.calls_before or 0) >= struggle_min_calls \
            or r.first_edit_inside is False:
        r.path = "struggled"
    elif (r.calls_before or 0) < direct_max_calls and len(detours) <= direct_max_detours:
        r.path = "direct"
    else:
        r.path = "searched"
    return r


def profile(trace: dict[str, Any], gold: dict[str, list[tuple[int, int]]], meta: dict[str, Any],
            direct_max_calls: int = 6, direct_max_detours: int = 1, struggle_min_detours: int = 4,
            struggle_min_calls: int = 15) -> LocProfile:
    entries = trace.get("entries", [])
    calls = [e for e in entries if e.get("type") == "tool_call"]
    results = [e for e in entries if e.get("type") == "tool_response"]
    th = (direct_max_calls, direct_max_detours, struggle_min_detours, struggle_min_calls)
    p = LocProfile(meta["instance_id"], int(meta.get("attempt") or 1), bool(meta.get("solved")),
                   meta.get("verifier") or "", len(calls), reach(calls, results, gold, *th))
    if p.solved and meta.get("patch"):
        own = gold_locations(meta["patch"])
        p.own = reach(calls, results, own, *th)
        p.fixed_elsewhere = bool(own) and bool(gold) and not (set(own) & set(gold))
    p.path = p.own.path if p.own else p.gold.path
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
    def med(xs: list[float]) -> str:
        return f"{statistics.median(xs):.0f}" if xs else "-"

    order = ["direct", "searched", "struggled", "never_found", "no_target"]
    solved = [p for p in profiles if p.solved]
    unsolved = [p for p in profiles if not p.solved]
    lines = ["# Localization profiles", "",
             f"trajectories: {len(profiles)} ({len(solved)} solved, {len(unsolved)} unsolved); "
             f"skipped (no trace or task row): {missing}", "",
             "`path` for a solved attempt is measured against its own verified patch (the place it really "
             "fixed); for an unsolved one against the reference fix.", "",
             "| path | solved (own patch) | solved (reference fix) | unsolved (reference fix) |",
             "| --- | --- | --- | --- |"]
    for k in order:
        lines.append(f"| {k} | {sum(p.own is not None and p.own.path == k for p in solved)} | "
                     f"{sum(p.gold.path == k for p in solved)} | {sum(p.gold.path == k for p in unsolved)} |")
    if solved:
        lines += ["", f"solved somewhere else than the reference fix (no file in common): "
                      f"{sum(bool(p.fixed_elsewhere) for p in solved)} of {len(solved)}"]
    for name, g, key in (("solved, own patch", solved, "own"), ("unsolved, reference fix", unsolved, "gold")):
        rs = [getattr(p, key) for p in g if getattr(p, key) is not None]
        found = [r for r in rs if r.first_open]
        if not rs:
            continue
        lines += ["", f"## {name}", "",
                  f"- calls before reaching it: median {med([r.calls_before or 0 for r in found])}",
                  f"- detour files before it: median {med([len(r.detour_files) for r in found])}, "
                  f"4 or more: {sum(len(r.detour_files) >= 4 for r in found)}",
                  f"- saw the exact changed lines: {sum(r.first_lines is not None for r in rs)} of {len(rs)}",
                  f"- first edit outside it: {sum(r.first_edit_inside is False for r in rs)}"]
    by_src: dict[str, Counter] = {}
    for p in profiles:
        by_src.setdefault(p.verifier or "?", Counter())[p.path] += 1
    lines += ["", "## By source (path as above)", "", "| source | " + " | ".join(order[:4]) + " |",
              "| --- |" + " --- |" * 4]
    for s_, c in sorted(by_src.items()):
        lines.append(f"| {s_} | " + " | ".join(str(c[k]) for k in order[:4]) + " |")
    return "\n".join(lines) + "\n"
