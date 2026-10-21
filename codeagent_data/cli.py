"""Stage 0 CLI: ingest -> normalize -> check -> route to env tier -> write task files + report.

    python -m codeagent_data ingest data/*.jsonl --out out/stage0 \
        --repo-cache ~/.cache/repocache --apply-check \
        --snapshots-dir /path/to/competition/snapshots
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .jsonl import iter_jsonl, read_jsonl
from .checks import harness_checks, repo_checks, static_checks, tier_checks
from .env import DEFAULT_EVAL_RESERVED, norm_source, resolve_env
from .ingest import apply_official, iter_raw_rows, load_official_meta, to_record
from .repocache import RepoCache
from .schema import Severity, TaskRecord

# When the same task appears in several sources, keep the one with the best environment story.
SOURCE_PRIORITY = ["competition", "swe_bench_verified", "swe_bench", "swe_bench_lite", "swe_rebench", "swe_gym", "swe_smith"]


def _prio(src: str) -> int:
    s = norm_source(src)
    return SOURCE_PRIORITY.index(s) if s in SOURCE_PRIORITY else len(SOURCE_PRIORITY)


def run_ingest(args: argparse.Namespace) -> int:
    out = Path(args.out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    cache = RepoCache(Path(args.repo_cache).expanduser()) if args.repo_cache else None
    reserved = set() if args.allow_eval_sources else set(DEFAULT_EVAL_RESERVED)
    snaps = Path(args.snapshots_dir) if args.snapshots_dir else None
    meta = load_official_meta(Path(args.official_meta)) if args.official_meta else {}

    recs: list[TaskRecord] = []
    for p in args.inputs:
        for i, row in enumerate(iter_raw_rows(Path(p))):
            if args.limit and len(recs) >= args.limit:
                break
            rec = to_record(row)
            if meta:
                apply_official(rec, meta, repair=args.repair_from_official)
            static_checks(rec, allow_non_python=args.allow_non_python)
            rec.env = resolve_env(rec, snapshots_dir=snaps)
            tier_checks(rec)
            harness_checks(rec)
            if norm_source(rec.source) in reserved:
                rec.role = "heldout_eval"
            recs.append(rec)

    # de-duplicate on (repo, instance_id), preferring higher-priority sources
    groups: dict[tuple[str, str], list[TaskRecord]] = defaultdict(list)
    for r in recs:
        groups[(r.repo.lower(), r.instance_id)].append(r)
    for (_, iid), rs in groups.items():
        if len(rs) > 1:
            rs.sort(key=lambda r: _prio(r.source))
            for dup in rs[1:]:
                dup.add("E_DUPLICATE", Severity.ERROR, f"duplicate of {rs[0].uid} ({iid})")

    if cache:
        # repo checks are independent per row: run them in parallel (clones are locked per repo).
        # Path checks run on rejected rows too (diagnostic); the apply check only on rows still ok.
        from concurrent.futures import ThreadPoolExecutor

        todo = [r for r in recs if not any(i.code == "E_DUPLICATE" for i in r.issues)]
        n_done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(lambda r: repo_checks(r, cache, apply_check=args.apply_check), todo):
                n_done += 1
                if n_done % 200 == 0 or n_done == len(todo):
                    print(f"repo checks: {n_done}/{len(todo)}", flush=True)

    # outputs
    with (out / "normalized.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    buckets: dict[str, list[TaskRecord]] = defaultdict(list)
    for r in recs:
        if r.ok:
            prefix = "heldout_" if r.role == "heldout_eval" else ""
            # one file per test runner family, e.g. tasks/swebench.jsonl, tasks/swe_smith.jsonl
            family = r.verifier or norm_source(r.source) or r.env.tier.value
            buckets[f"{prefix}{family}"].append(r)
    for name, rs in buckets.items():
        with (out / "tasks" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r.to_swegemma_task(), ensure_ascii=False) + "\n")

    report = build_report(recs, buckets)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(report, recs), encoding="utf-8")
    print(render_report(report, recs))
    return 0


def build_report(recs: list[TaskRecord], buckets: dict[str, list[TaskRecord]]) -> dict:
    by_code: Counter[str] = Counter()
    rejected_by_first_error: Counter[str] = Counter()
    for r in recs:
        for i in r.issues:
            by_code[i.code] += 1
        errs = [i.code for i in r.issues if i.severity == Severity.ERROR]
        if errs:
            rejected_by_first_error[errs[0]] += 1
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for r in recs:
        by_source[r.source]["total"] += 1
        by_source[r.source]["ok" if r.ok else "rejected"] += 1
        if r.ok:
            by_source[r.source][f"tier:{r.env.tier.value}"] += 1
    return {
        "total": len(recs),
        "ok": sum(r.ok for r in recs),
        "buckets": {k: len(v) for k, v in sorted(buckets.items())},
        "issues": dict(by_code.most_common()),
        "rejected_by_first_error": dict(rejected_by_first_error.most_common()),
        "by_source": {k: dict(v) for k, v in by_source.items()},
    }


def render_report(rep: dict, recs: list[TaskRecord]) -> str:
    lines = ["# Stage 0 report", "", f"rows: {rep['total']}  ok: {rep['ok']}  rejected: {rep['total'] - rep['ok']}", ""]
    lines += ["## Output buckets (tasks/<bucket>.jsonl)", ""] + [f"- {k}: {v}" for k, v in rep["buckets"].items()]
    lines += ["", "## Rejections (first error)", ""] + [f"- {k}: {v}" for k, v in rep["rejected_by_first_error"].items()]
    lines += ["", "## All issue codes", ""] + [f"- {k}: {v}" for k, v in rep["issues"].items()]
    lines += ["", "## By source", ""] + [f"- {k}: {v}" for k, v in rep["by_source"].items()]
    bad = [r for r in recs if not r.ok][:20]
    if bad:
        lines += ["", "## Sample rejected rows", ""]
        for r in bad:
            lines.append(f"### {r.uid} ({r.instance_id})")
            for i in r.issues:
                lines.append(f"- [{i.severity.value}] {i.code}: {i.message}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="codeagent_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="normalize + validate + route raw task rows")
    ing.add_argument("inputs", nargs="+")
    ing.add_argument("--out", required=True)
    ing.add_argument("--repo-cache", help="dir for blobless clones; enables repo-tree checks")
    ing.add_argument("--apply-check", action="store_true", help="also git-apply gold + test patches (needs --repo-cache)")
    ing.add_argument("--snapshots-dir", help="competition snapshots dir; rows found there become tier=native")
    ing.add_argument("--allow-eval-sources", action="store_true", help="do not hold out SWE-bench Verified")
    ing.add_argument("--allow-non-python", action="store_true")
    ing.add_argument("--limit", type=int, default=0)
    ing.add_argument("--workers", type=int, default=16, help="rows checked in parallel against the repo cache")
    ing.add_argument("--official-meta", help="jsonl from scripts/export_official_meta.py (adds version, detects misaligned rows)")
    ing.add_argument("--repair-from-official", action="store_true", help="replace misaligned text fields with the official row")

    val = sub.add_parser("validate", help="stage 1: base/gold test runs in each task's image (no model)")
    val.add_argument("tasks", nargs="+", help="tasks/*.jsonl from ingest (image tier)")
    val.add_argument("--out", required=True)
    val.add_argument("--workers", type=int, default=4, help="tasks validated in parallel")
    val.add_argument("--repeat", type=int, default=1, help="gold runs per task; 2 detects flaky tests but doubles the cost")
    val.add_argument("--shard", help="K/N: this machine takes shard K of N (0-based); images are not split across shards")
    val.add_argument("--timeout", type=int, default=1800, help="seconds per test run")
    val.add_argument("--mem", default="8g")
    val.add_argument("--cpus", type=float, default=2.0)
    val.add_argument("--offline", action="store_true", help="disable network in validation containers")
    val.add_argument("--no-pull", action="store_true")
    val.add_argument("--prune-images", action="store_true", help="delete each image after its task (bounded disk)")
    val.add_argument("--limit", type=int, default=0)
    val.add_argument("--rate-limit-wait", type=float, default=90.0,
                     help="minutes a task may wait for Docker Hub's pull limit before it is marked IMAGE_UNAVAILABLE")
    val.add_argument("--max-minutes", type=float, default=None,
                     help="stop starting new tasks after this many minutes (for time-limited machines/CI)")
    val.add_argument("--skip-done", nargs="*", default=[], help="verdicts.jsonl files from earlier runs: skip those tasks")
    val.add_argument("--retry", nargs="*", default=[],
                     help="verdict reasons to run again instead of skipping, e.g. --retry NO_TEST_OUTPUT GOLD_FAILS_F2P")

    sel = sub.add_parser("select", help="rank tasks so the first N solved ones are the most reliable mix")
    sel.add_argument("--stage0", required=True, help="ingest output folder (normalized.jsonl + tasks/)")
    sel.add_argument("--out", required=True)
    sel.add_argument("--n", type=int, default=2400, help="number of SOLVED tasks wanted")
    sel.add_argument("--shares", default="swebench=0.40,swe_gym=0.35,r2e_gym=0.15,swe_smith=0.10",
                     help="interleaving weights per source (the mix of attempts)")
    sel.add_argument("--max-share", default="swebench=0.50,swe_gym=0.50,r2e_gym=0.35,swe_smith=0.20",
                     help="ceiling on each source's share of the SOLVED tasks")
    sel.add_argument("--max-per-repo", type=int, default=120, help="ceiling on solved tasks per repo")
    sel.add_argument("--gate", action="append", default=[],
                     help="override a gate, e.g. --gate max_leak_ratio=0.6 --gate max_f2p=60 (see select.Gates)")

    mrg = sub.add_parser("merge", help="combine verdicts.jsonl from several runs/machines into one report")
    mrg.add_argument("--tasks", nargs="+", required=True, help="tasks/*.jsonl that were validated")
    mrg.add_argument("--verdicts", nargs="+", required=True, help="verdicts.jsonl files (later ones win)")
    mrg.add_argument("--out", required=True)
    gen = sub.add_parser("generate", help="stage 4: teacher rollouts (swegemma loop + OpenRouter model) + verification")
    gen.add_argument("tasks", nargs="+", help="valid_tasks.jsonl from validate/merge")
    gen.add_argument("--out", required=True)
    gen.add_argument("--targets", help="targets.json from select (solved target, caps); default: --n and select's defaults")
    gen.add_argument("--n", type=int, default=2400, help="solved tasks wanted when --targets is not given")
    gen.add_argument("--shard", help="K/N: this machine takes every N-th task (by rank) and 1/N of the target")
    gen.add_argument("--prior", nargs="*", default=[], help="results_merged.jsonl of earlier runs (skip + count them)")
    gen.add_argument("--only", nargs="*", default=[], help="run just these instance ids (debugging)")
    gen.add_argument("--workers", type=int, default=4, help="tasks in parallel (mostly waiting on the API)")
    gen.add_argument("--limit", type=int, default=0, help="max new attempts (e.g. 10 for a first test)")
    gen.add_argument("--max-minutes", type=float, default=None, help="stop starting new tasks after N minutes")
    gen.add_argument("--prune-images", action="store_true")
    gen.add_argument("--no-pull", action="store_true")
    gen.add_argument("--rate-limit-wait", type=float, default=90.0)
    gen.add_argument("--model", default="moonshotai/kimi-k3", help="OpenRouter model id")
    gen.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    gen.add_argument("--extra-body", default="", help='JSON added to each request, e.g. \'{"provider": {"sort": "price"}}\'')
    gen.add_argument("--teacher-dir", default=None, help="agent.yaml + prompts (default: teacher/)")
    gen.add_argument("--time-minutes", type=float, default=45.0, help="agent time budget per task")
    gen.add_argument("--tool-calls", type=int, default=100, help="agent tool-call budget per task")
    gen.add_argument("--turns", type=int, default=300, help="agent LLM-call budget per task")
    gen.add_argument("--command-timeout", type=int, default=300, help="seconds per run_command")
    gen.add_argument("--mem", default="6g")
    gen.add_argument("--cpus", type=float, default=2.0)
    gen.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens (cost estimate only)")
    gen.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens")
    gen.add_argument("--price-cached", type=float, default=None, help="USD per 1M cached prompt tokens")

    gm = sub.add_parser("generate-merge", help="combine generate results (shards + earlier runs) under the global limits")
    gm.add_argument("--tasks", nargs="+", required=True, help="the valid_tasks.jsonl given to generate")
    gm.add_argument("--results", nargs="+", required=True, help="results.jsonl / results_merged.jsonl (later win)")
    gm.add_argument("--targets", help="targets.json from select")
    gm.add_argument("--n", type=int, default=2400)
    gm.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        return run_generate(args)
    if args.cmd == "generate-merge":
        from .generate.rollout import merge_generation
        merge_generation(_read_jsonl(args.tasks), [Path(p) for p in args.results],
                         _load_targets(args.targets, args.n), Path(args.out))
        return 0
    if args.cmd == "ingest":
        return run_ingest(args)
    if args.cmd == "validate":
        return run_validate(args)
    if args.cmd == "merge":
        return run_merge(args)
    if args.cmd == "select":
        from dataclasses import fields as _fields
        from .select import Gates, run_select

        def _kv(text: str) -> dict[str, float]:
            return {k.strip(): float(v) for k, v in (kv.split("=") for kv in text.split(",") if kv.strip())}

        gates = Gates()
        types = {f.name: f.type for f in _fields(Gates)}
        for kv in args.gate:
            k, v = kv.split("=", 1)
            if k not in types:
                raise SystemExit(f"unknown gate {k!r}; choose from {sorted(types)}")
            setattr(gates, k, float(v) if "float" in str(types[k]) else int(v))
        run_select(Path(args.stage0), Path(args.out), args.n, _kv(args.shares), args.max_per_repo,
                   _kv(args.max_share), gates)
        return 0
    return 1


def _read_jsonl(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        rows += read_jsonl(p)
    return rows


def write_stage1_outputs(tasks: list[dict], verdict_files: list[Path], out: Path) -> None:
    """valid_tasks.jsonl + report.md from one or more verdicts.jsonl (later files win)."""
    verdicts: dict[str, dict] = {}
    for vf in verdict_files:
        if vf.exists():
            for v in iter_jsonl(vf):
                verdicts[v["instance_id"]] = v
    out.mkdir(parents=True, exist_ok=True)
    with (out / "verdicts_merged.jsonl").open("w") as f:  # pass to --skip-done in the next run
        for v in verdicts.values():
            f.write(json.dumps(v) + "\n")
    judged = [verdicts[t["instance_id"]] for t in tasks if t["instance_id"] in verdicts]
    reasons = Counter(v["reason"].split(":")[0].split(" (")[0] for v in judged)
    n_valid = 0
    with (out / "valid_tasks.jsonl").open("w") as f:
        for t in tasks:
            v = verdicts.get(t["instance_id"])
            if v and v["valid"]:
                n_valid += 1
                if v.get("mode") in ("expected", "derive"):  # store the lists stage 1 derived
                    t = {**t, "FAIL_TO_PASS": v["derived_f2p"], "PASS_TO_PASS": v["derived_p2p"]}
                t = {**t, "validation": {k: v.get(k) for k in ("route", "mode", "strict_file_pass", "flaky", "seconds")}}
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    by_source: dict[str, Counter] = {}
    reason_src: dict[str, Counter] = {}
    src_of = {t["instance_id"]: t.get("verifier") or t.get("source") or "?" for t in tasks}
    for v in judged:
        src = src_of.get(v["instance_id"], "?")
        by_source.setdefault(src, Counter())["valid" if v["valid"] else "invalid"] += 1
        reason_src.setdefault(v["reason"].split(":")[0].split(" (")[0], Counter())[src] += 1
    lines = ["# Stage 1 report", "",
             f"tasks: {len(tasks)}  judged: {len(judged)}  valid: {n_valid}  not yet run: {len(tasks) - len(judged)}", ""]
    lines += [f"- {k}: {c}" for k, c in reasons.most_common()]
    lines += ["", "## By source", ""] + [f"- {k}: {dict(c)}" for k, c in sorted(by_source.items())]
    srcs = sorted(by_source)
    lines += ["", "## Reasons by source", "", "| reason | " + " | ".join(srcs) + " |",
              "| --- |" + " --- |" * len(srcs)]
    for reason, c in sorted(reason_src.items(), key=lambda kv: -sum(kv[1].values())):
        lines.append(f"| {reason} | " + " | ".join(str(c.get(s, 0)) for s in srcs) + " |")
    strict = sum(1 for v in judged if v["valid"] and v.get("strict_file_pass"))
    lines += ["", f"valid and every test that ran passes (competition-style rule): {strict}"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_validate(args: argparse.Namespace) -> int:
    from .validate.validator import run_validation, select_shard

    tasks = select_shard(_read_jsonl(args.tasks), args.shard)
    if args.skip_done:  # verdicts from earlier runs / other machines
        from .validate.validator import is_final
        done = {v["instance_id"] for v in _read_jsonl([p for p in args.skip_done if Path(p).exists()]) if is_final(v, tuple(args.retry))}
        tasks = [t for t in tasks if t["instance_id"] not in done]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"validating {len(tasks)} tasks with {args.workers} workers"
          + (f" (shard {args.shard})" if args.shard else ""), flush=True)

    def factory():
        from .validate.docker_runner import DockerRunner
        return DockerRunner(mem_limit=args.mem, cpus=args.cpus, network=not args.offline,
                            pull=not args.no_pull, workers=args.workers,
                            rate_limit_wait_minutes=args.rate_limit_wait)

    out = Path(args.out)
    run_validation(tasks, factory, out, workers=args.workers, repeat=args.repeat,
                   timeout_s=args.timeout, prune_images=args.prune_images, max_minutes=args.max_minutes,
                   retry=tuple(args.retry))
    write_stage1_outputs(tasks, [out / "verdicts.jsonl"], out)
    return 0


def _load_targets(path: str | None, n: int) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    from .select import DEFAULT_MAX_SHARE, DEFAULT_SHARES
    return {"solved_target": n, "shares": dict(DEFAULT_SHARES), "max_share": dict(DEFAULT_MAX_SHARE),
            "max_per_repo": 120}


def run_generate(args: argparse.Namespace) -> int:
    from .generate.rollout import GenConfig, run_generation, select_shard_tasks, shard_targets

    tasks, n_shards = select_shard_tasks(_read_jsonl(args.tasks), args.shard)
    if args.only:
        tasks = [t for t in tasks if t["instance_id"] in set(args.only)]
    targets = _load_targets(args.targets, args.n)
    if n_shards > 1:
        targets = shard_targets(targets, n_shards)
    cfg = GenConfig(model=args.model, reasoning_effort=args.reasoning_effort,
                    extra_body=json.loads(args.extra_body) if args.extra_body else {},
                    time_minutes=args.time_minutes, tool_calls=args.tool_calls, turns=args.turns,
                    command_timeout=args.command_timeout, mem=args.mem, cpus=args.cpus,
                    price_in=args.price_in, price_out=args.price_out, price_cached=args.price_cached)
    if args.teacher_dir:
        cfg.teacher_dir = Path(args.teacher_dir)
    prior = _read_jsonl([p for p in args.prior if Path(p).exists()])
    print(f"generate: {len(tasks)} candidate tasks, target {targets['solved_target']} solved"
          + (f" (shard {args.shard})" if args.shard else "") + f", model {cfg.model}", flush=True)

    def factory():
        from .validate.docker_runner import DockerRunner
        # verification containers get network like stage 1; the agent's container never does
        return DockerRunner(mem_limit=args.mem, cpus=args.cpus, pull=not args.no_pull, workers=args.workers,
                            rate_limit_wait_minutes=args.rate_limit_wait)

    out = Path(args.out)
    run_generation(tasks, targets, cfg, out, runner_factory=factory, workers=args.workers, limit=args.limit,
                   max_minutes=args.max_minutes, prune_images=args.prune_images, prior=prior)
    from .generate.rollout import render_generation_report
    rows = _read_jsonl([str(out / "results.jsonl")]) if (out / "results.jsonl").exists() else []
    (out / "generation_report.md").write_text(render_generation_report(rows, targets), encoding="utf-8")
    print((out / "generation_report.md").read_text())
    return 0


def run_merge(args: argparse.Namespace) -> int:
    write_stage1_outputs(_read_jsonl(args.tasks), [Path(p) for p in args.verdicts], Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
