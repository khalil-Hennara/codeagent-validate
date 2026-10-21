"""Teacher rollouts: swegemma's own agent loop, on each task's image, with an OpenRouter model.

What is reused unchanged from swegemma (so trajectories match what Gemma sees at inference):
the 6 tools and their output limits, the task prompt (`build_agent_prompt`), continuation
nudges, budgets, patch extraction (`submit_patch` / working-tree fallback), stripping of edits
to test/config files, and the SessionTrace format (thinking, text, tool calls, tool results,
token usage).

What is replaced: swegemma prepares its universal py3.13 image from a snapshot + wheels. Here
the task's own image is used, so those steps become no-ops and `ImageTaskSandbox` prepares the
container instead (see prepare.py). Only git-exclude + the baseline commit run as in swegemma.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..jsonl import iter_jsonl
from ..patches import is_protected_config, is_test_path, parse_patch
from ..validate.specs import UnsupportedVerifier, build_plan
from ..validate.validator import judge
from .prepare import SETUP_PATCH_PATH, prepare_script

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_DIR = ROOT / "teacher"
AGENT_PATH = ("/workspace/.venv/bin:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
              "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
# errors worth a later retry (not the task's fault) / errors that must stop the whole run
RETRYABLE = ("IMAGE_UNAVAILABLE", "INFRA_ERROR", "PREPARE_FAILED", "API_ERROR")
_API_ERRORS = ("RateLimitError", "APIConnectionError", "ServiceUnavailable", "InternalServerError",
               "Timeout", "APIError", "BadGateway", "overloaded")
_FATAL_ERRORS = ("AuthenticationError", "401", "402", "insufficient", "Insufficient credits")


@dataclass
class GenConfig:
    model: str = "moonshotai/kimi-k3"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_base: str | None = None  # default: OpenRouter (tests point this at a local fake server)
    reasoning_effort: str | None = None  # "low" | "medium" | "high"; None = the model's default
    extra_body: dict[str, Any] = field(default_factory=dict)  # e.g. {"provider": {"sort": "price"}}
    teacher_dir: Path = DEFAULT_TEACHER_DIR
    time_minutes: float = 45.0
    tool_calls: int = 100
    turns: int = 300
    command_timeout: int = 300
    mem: str = "8g"
    cpus: float = 2.0
    # USD per 1M tokens, only for the cost estimate in the report
    price_in: float = 0.0
    price_out: float = 0.0
    price_cached: float | None = None


# ---------------------------------------------------------------------------------------------
# swegemma glue


def _patched_agent_runner():
    """swegemma's agent runner with its snapshot/wheels/setup.py preparation switched off."""
    import swegemma.harness.agent_runner as ar

    def noop(*_a: Any, **_k: Any) -> None:
        return None

    for name in ("setup_container_wheels", "extract_snapshot", "install_editable_package",
                 "install_test_dependencies", "setup_workspace_test_config"):
        setattr(ar, name, noop)
    return ar


def build_eval_config(cfg: GenConfig, out: Path, model_override: Any = None):
    from adk_submission import GenerationConstraints, ModelRegistry
    from swegemma.config import EvalConfig

    models = ModelRegistry()
    if model_override is not None:  # tests
        models.register("teacher", model_override)
    else:
        from google.adk.models.lite_llm import LiteLlm

        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise SystemExit(f"set {cfg.api_key_env} (the OpenRouter API key)")
        # litellm's openrouter provider maps OpenRouter's `reasoning` field to reasoning_content,
        # which ADK turns into thought parts -> "thinking" entries in the trace
        kwargs: dict[str, Any] = {"api_key": key, "num_retries": 5}
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        body = dict(cfg.extra_body)
        if cfg.reasoning_effort:  # OpenRouter's unified reasoning parameter
            body.setdefault("reasoning", {})["effort"] = cfg.reasoning_effort
        if body:
            kwargs["extra_body"] = body
        models.register("teacher", LiteLlm(model=f"openrouter/{cfg.model}", **kwargs))
    # Budgets go through the flat kwargs, exactly as in the organizers' notebook: the wheel's
    # EvalConfig ignores `budget=` (the first run got 10 minutes instead of 45).
    econf = EvalConfig(
        tasks_path=out / "tasks.jsonl", snapshots_dir=out, results_dir=out,
        submission_dir=Path(cfg.teacher_dir), models=models,
        timeout_seconds=cfg.command_timeout, max_time_minutes=float(cfg.time_minutes),
        max_tool_calls=cfg.tool_calls, max_turns=cfg.turns,
        generation_constraints=GenerationConstraints(defaults={}),  # no injected thinking budget
        display_mode="quiet", enable_sandbox_testing=True,
    )
    got = (econf.budget.time_minutes, econf.budget.tool_calls, econf.budget.turns)
    if got != (float(cfg.time_minutes), cfg.tool_calls, cfg.turns):
        raise SystemExit(f"the harness did not take the budget: got {got}, wanted "
                         f"{(cfg.time_minutes, cfg.tool_calls, cfg.turns)}")
    return econf


def swegemma_task(t: dict[str, Any]):
    from swegemma.models import Task

    return Task(instance_id=t["instance_id"], repo=t["repo"], base_commit=t.get("base_commit") or "",
                problem_statement=t["problem_statement"], patch="", test_patch=t.get("test_patch") or "",
                hints_text="")  # never: the student does not see hints, so neither does the teacher


def make_image_sandbox(task: dict[str, Any], cfg: GenConfig, put_file: Callable[..., None]):
    """A swegemma ContainerManager for this task's image that prepares the workspace on start."""
    from swegemma.harness.container_setup import ContainerSetupError
    from swegemma.sandbox import ContainerConfig, ContainerManager

    class ImageTaskSandbox(ContainerManager):
        def start(self) -> str:
            cid = super().start()
            setup = (task.get("setup_patch") or "").strip()
            if setup:
                put_file(self.client.containers.get(cid), SETUP_PATCH_PATH, task["setup_patch"])
            script = prepare_script(task.get("env_workdir") or "/testbed", "/workspace", has_setup_patch=bool(setup))
            res = self.exec(cid, script, timeout=900)
            if res.exit_code != 0 or "PREPARE_OK" not in (res.stdout or ""):
                self.stop(cid)
                raise ContainerSetupError(f"PREPARE_FAILED: {(res.stderr or res.stdout)[-800:]}")
            return cid

    sb = ImageTaskSandbox(ContainerConfig(
        image=task["env_image"], network_mode="none", mem_limit=cfg.mem,
        cpu_quota=int(cfg.cpus * 100_000), working_dir="/workspace", timeout_seconds=cfg.command_timeout,
        environment={"TEST_TMPDIR": "/tmp", "PATH": AGENT_PATH},
    ))
    sb._image_resolved = True  # the image was pulled already (rate-limit aware), don't re-resolve
    return sb


def trace_stats(trace: Any) -> dict[str, int]:
    st = Counter()
    for e in trace.entries if trace is not None else []:
        u = getattr(e, "usage", None) or {}
        st["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        # largest single request = the context length the trajectory reaches (matters for the student)
        st["max_context_tokens"] = max(st["max_context_tokens"], int(u.get("prompt_tokens") or 0))
        st["completion_tokens"] += int(u.get("completion_tokens") or 0)
        st["cached_tokens"] += int(u.get("cached_tokens") or 0)
        et = getattr(e, "event_type", "")
        st["tool_calls"] += et == "tool_call"
        st["thinking_entries"] += et == "thinking"
    return dict(st)


def save_trace(trace: Any, out: Path, instance_id: str, attempt: int = 1) -> str | None:
    """Raw entry list ("legacy" format): keeps thinking / text / tool_call / tool_response / usage
    apart, which the SFT parser needs (the ATIF export merges thinking into the message text)."""
    if trace is None:
        return None
    path = out / "traces" / f"{instance_id.replace('/', '__')}__a{attempt}.json"
    trace.save(path, format="legacy")
    return str(path.relative_to(out))  # relative to the run folder, so shard artifacts can move


def estimate_cost(st: dict[str, int], cfg: GenConfig) -> float:
    cached = st.get("cached_tokens", 0)
    price_cached = cfg.price_in if cfg.price_cached is None else cfg.price_cached
    uncached = max(0, st.get("prompt_tokens", 0) - cached)
    return (uncached * cfg.price_in + cached * price_cached + st.get("completion_tokens", 0) * cfg.price_out) / 1e6


def preflight(cfg: GenConfig, model_override: Any = None) -> str:
    """One small request through exactly the agent's path (ADK LiteLlm -> litellm -> OpenRouter, with
    a tool declared) before any container starts: a wrong key, model id or missing dependency fails
    here in seconds instead of after every task's setup."""
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    model = model_override or build_eval_config(cfg, Path("/tmp")).models.get("teacher")
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="get_status", description="Return the session status.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={}))])
    req = LlmRequest(model=getattr(model, "model", None),
                     contents=[types.Content(role="user", parts=[types.Part(text="Reply with the word OK.")])],
                     config=types.GenerateContentConfig(tools=[tool], max_output_tokens=512))

    async def call() -> list[Any]:
        return [r async for r in model.generate_content_async(req, stream=False)]

    try:
        responses = asyncio.run(call())
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"preflight: the model call failed, nothing was started:\n  {type(e).__name__}: {e}")
    err = next((r.error_message for r in responses if getattr(r, "error_message", None)), None)
    if err:
        raise SystemExit(f"preflight: the model returned an error, nothing was started:\n  {err}")
    parts = [p for r in responses if r.content for p in (r.content.parts or [])]
    thought = any(getattr(p, "thought", False) for p in parts)
    text = "".join(p.text or "" for p in parts if p.text and not getattr(p, "thought", False))
    return (f"preflight OK: {cfg.model} answered {text[:40]!r}; "
            f"reasoning returned: {'yes' if thought else 'NO (check the model / provider)'}")


def classify_agent_error(err: str | None) -> str | None:
    if not err:
        return None
    if any(k in err for k in _FATAL_ERRORS):
        return "FATAL_API_ERROR"
    if "PREPARE_FAILED" in err:
        return "PREPARE_FAILED"
    if any(k in err for k in _API_ERRORS):
        return "API_ERROR"
    return None


# ---------------------------------------------------------------------------------------------
# one task


@dataclass
class Attempt:
    instance_id: str
    uid: str = ""
    verifier: str = ""
    repo: str = ""
    fold: str = ""
    selection_rank: int = -1
    solved: bool = False
    reason: str = ""
    agent_error: str | None = None
    patch: str = ""
    seconds: float = 0.0
    est_cost_usd: float = 0.0
    stats: dict[str, int] = field(default_factory=dict)
    trace_path: str | None = None
    model: str = ""
    attempt: int = 1
    # how close an unsolved patch came: target tests failing, F2P passing fraction
    tests: dict[str, Any] = field(default_factory=dict)
    # localization: the gold patch's source files that the teacher's patch also edits
    loc: dict[str, Any] = field(default_factory=dict)
    gold: dict[str, int] = field(default_factory=dict)  # size of the reference fix


def source_files(patch: str) -> list[str]:
    return sorted({f.path for f in parse_patch(patch or "").files
                   if f.path and not is_test_path(f.path) and not is_protected_config(f.path)})


def gold_size(task: dict[str, Any]) -> dict[str, int]:
    """Size of the reference fix (source files only): the best cheap predictor of difficulty."""
    files = [f for f in parse_patch(task.get("patch") or "").files
             if f.path and not is_test_path(f.path) and not is_protected_config(f.path)]
    return {"files": len(files), "lines": sum(f.added + f.removed for f in files),
            "hunks": sum(f.hunks for f in files)}


def size_bucket(lines: int) -> str:
    return "1-10" if lines <= 10 else "11-30" if lines <= 30 else "31-80" if lines <= 80 else ">80"


def localization(task: dict[str, Any], patch: str) -> dict[str, Any]:
    gold, mine = source_files(task.get("patch") or ""), source_files(patch)
    hit = sorted(set(gold) & set(mine))
    return {"gold_files": gold, "teacher_files": mine, "hit": len(hit),
            "all_gold_files": bool(gold) and len(hit) == len(gold)}


def _test_progress(task: dict[str, Any], v: Any) -> dict[str, Any]:
    f2p = set(task.get("FAIL_TO_PASS") or [])
    failing = set(v.f2p_fail_on_gold or []) | set(v.p2p_fail_on_gold or [])
    f2p_failing = len(failing & f2p) if f2p else len(v.f2p_fail_on_gold or [])
    n_f2p = len(f2p) or max(1, v.f2p_total)
    return {"failing": len(failing), "f2p_failing": f2p_failing, "f2p_total": n_f2p,
            "f2p_pass_frac": round(1 - f2p_failing / n_f2p, 3), "failing_names": sorted(failing)[:20]}


def verify_patch(task: dict[str, Any], patch: str, runner: Any, timeout_s: int = 1800,
                 log_path: Path | None = None, flaky_rerun_max: int = 2) -> tuple[bool, str, dict[str, Any]]:
    """Run the task's tests on base + (setup) + teacher patch in a fresh container.

    When only 1-2 target tests fail, the tests are run once more: stage 3 ran the gold patch only
    once, so a flaky test could otherwise reject a correct patch. The test log of an unsolved
    attempt is kept at `log_path` for review. Returns (solved, reason, test progress)."""
    try:
        plan = build_plan(task)
    except UnsupportedVerifier as e:
        return False, f"UNSUPPORTED: {e}", {}
    setup = [("setup", task["setup_patch"])] if (task.get("setup_patch") or "").strip() else []
    progress: dict[str, Any] = {}
    logs: list[str] = []
    for run in range(2):
        res = runner.run(task["env_image"], plan.script, workdir=plan.workdir,
                         patches=setup + [("agent", patch)], timeout_s=timeout_s)
        logs.append(res.log)
        if res.stage == "agent_apply":
            ok, reason = False, "AGENT_PATCH_APPLY_FAIL"
        elif res.stage == "timeout":
            ok, reason = False, "TIMEOUT (verification)"
        elif res.stage != "done":
            ok, reason = False, f"INFRA_ERROR: verification stopped at {res.stage}"
        else:
            v = judge(task, plan, "", [res.log])
            # stage 3's verdict names talk about "the gold patch"; here it is the teacher's patch
            ok = v.valid
            reason = "SOLVED" if ok else (v.reason.replace("GOLD_", "", 1)
                                          .replace("the gold patch", "the teacher's patch"))
            if ok and run == 1:
                reason = "SOLVED (on a re-run: a target test is flaky)"
            if not ok:
                progress = _test_progress(task, v)
                if progress["failing_names"]:
                    reason += f" | failing: {', '.join(progress['failing_names'][:8])}"
            if not ok and run == 0 and 0 < progress.get("failing", 0) <= flaky_rerun_max:
                continue  # one more run before rejecting a near-miss
        break
    if not ok and log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{reason}\n\n" + "\n\n>>>>> RE-RUN\n\n".join(logs), encoding="utf-8",
                            errors="replace")
    return ok, reason, progress


def retry_worthy(a: "Attempt") -> bool:
    """Spend a second try only where the first one got somewhere: half or more of the F2P tests
    pass, or the patch edits the right files, or the budget ran out before a patch existed."""
    if a.solved:
        return False
    if a.reason.startswith("NO_PATCH"):
        return True
    return (a.tests.get("f2p_pass_frac", 0) >= 0.5) or bool(a.loc.get("hit"))


def run_one(task: dict[str, Any], cfg: GenConfig, out: Path, runner: Any, *, attempt: int = 1,
            sandbox_factory: Callable[[dict[str, Any]], Any] | None = None,
            model_override: Any = None) -> Attempt:
    t0 = time.time()
    a = Attempt(task["instance_id"], task.get("uid", ""), task.get("verifier", ""), task.get("repo", ""),
                task.get("fold", ""), int(task.get("selection_rank", -1)), model=cfg.model, attempt=attempt,
                gold=gold_size(task))
    if sandbox_factory is None:
        err = runner.ensure_image(task["env_image"])
        if err:
            a.reason, a.seconds = f"IMAGE_UNAVAILABLE: {err[:300]}", time.time() - t0
            return a
        sandbox = make_image_sandbox(task, cfg, runner._put)
    else:
        sandbox = sandbox_factory(task)
    ar = _patched_agent_runner()
    econf = build_eval_config(cfg, out, model_override)
    try:
        patch, agent_error, trace = asyncio.run(
            ar.run_agent_sandbox(sandbox, econf, swegemma_task(task), Path("/dev/null")))
    except Exception as e:  # noqa: BLE001  (ContainerSetupError, docker errors, ...)
        a.reason = "PREPARE_FAILED" if "PREPARE_FAILED" in str(e) else f"INFRA_ERROR: {str(e)[:300]}"
        a.agent_error, a.seconds = str(e)[:2000], time.time() - t0
        return a
    a.agent_error, a.patch = agent_error, patch or ""
    a.stats = trace_stats(trace)
    a.est_cost_usd = round(estimate_cost(a.stats, cfg), 4)
    a.trace_path = save_trace(trace, out, task["instance_id"], attempt)
    kind = classify_agent_error(agent_error)
    if kind:
        a.reason = f"{kind}: {agent_error[:300]}"
    elif not a.patch.strip():
        a.reason = "NO_PATCH" + (f": {agent_error[:200]}" if agent_error else "")
    else:
        log = out / "logs" / f"{task['instance_id'].replace('/', '__')}__a{attempt}.verify.log"
        a.loc = localization(task, a.patch)
        a.solved, a.reason, a.tests = verify_patch(task, a.patch, runner, log_path=log)
    a.seconds = time.time() - t0
    return a


# ---------------------------------------------------------------------------------------------
# many tasks, in parallel, until the solved target is reached


def shard_targets(targets: dict[str, Any], num_shards: int, margin: float = 0.05) -> dict[str, Any]:
    """Each shard aims for its slice of the target (+margin); merge applies the global limits."""
    t = dict(targets)
    t["solved_target"] = math.ceil(targets["solved_target"] * (1 + margin) / num_shards)
    t["max_per_repo"] = max(1, math.ceil(targets["max_per_repo"] * (1 + margin) / num_shards))
    t["max_share"] = dict(targets["max_share"])
    return t


def select_shard_tasks(tasks: list[dict[str, Any]], shard: str | None) -> tuple[list[dict[str, Any]], int]:
    """Rank-ordered tasks; shard K/N takes every N-th task so every shard gets the same mix."""
    tasks = sorted(tasks, key=lambda t: int(t.get("selection_rank", 1 << 30)))
    if not shard:
        return tasks, 1
    k, n = (int(x) for x in shard.split("/"))
    return tasks[k::n], n


def run_generation(tasks: list[dict[str, Any]], targets: dict[str, Any], cfg: GenConfig, out: Path, *,
                   runner_factory: Callable[[], Any], workers: int = 4, limit: int = 0,
                   max_minutes: float | None = None, prune_images: bool = False,
                   attempts: int = 1, prior: list[dict[str, Any]] | None = None,
                   sandbox_factory: Callable[[dict[str, Any]], Any] | None = None,
                   model_override: Any = None) -> list[Attempt]:
    """`prior`: results of earlier runs (e.g. results_merged.jsonl). Those of these `tasks` that
    finished are skipped and their solved ones count toward the target.

    `attempts`: an unsolved task is tried again (a new sample; temperature 1.0) up to this many
    times in total. Retries go to the end of the queue, so new tasks come first."""
    from ..select import should_attempt

    out.mkdir(parents=True, exist_ok=True)
    results_fp = out / "results.jsonl"
    mine = {t["instance_id"] for t in tasks}
    earlier = [r for r in prior or [] if r["instance_id"] in mine]
    if results_fp.exists():  # resumable: keep finished attempts, retry infrastructure failures
        earlier += list(iter_jsonl(results_fp))
    finals: dict[str, set[int]] = {}
    solved_ids: dict[str, dict[str, Any]] = {}
    for r in earlier:
        if is_final(r):
            finals.setdefault(r["instance_id"], set()).add(int(r.get("attempt") or 1))
            if r.get("solved"):
                solved_ids.setdefault(r["instance_id"], r)
    solved_src = Counter(r["verifier"] for r in solved_ids.values())
    solved_repo = Counter((r.get("repo") or "").lower() for r in solved_ids.values())

    def next_attempt(iid: str) -> int:
        return max(finals.get(iid, {0})) + 1

    queue = [(t, next_attempt(t["instance_id"])) for t in tasks
             if t["instance_id"] not in solved_ids and next_attempt(t["instance_id"]) <= attempts]
    runner = runner_factory()
    lock = threading.Lock()
    state = {"next": 0, "started": 0, "stop": None, "api_fail_streak": 0}
    t0 = time.time()
    results: list[Attempt] = []

    def next_task() -> tuple[dict[str, Any], int] | None:
        with lock:
            while True:
                if state["stop"]:
                    return None
                if sum(solved_src.values()) >= targets["solved_target"]:
                    state["stop"] = "solved target reached"
                    return None
                if max_minutes and time.time() - t0 > max_minutes * 60:
                    state["stop"] = "--max-minutes reached"
                    return None
                if limit and state["started"] >= limit:
                    state["stop"] = "--limit reached"
                    return None
                if state["next"] >= len(queue):
                    state["stop"] = "no tasks left"
                    return None
                t, k = queue[state["next"]]
                state["next"] += 1
                if should_attempt(t, solved_src, solved_repo, targets):
                    state["started"] += 1
                    return t, k

    def worker() -> None:
        while True:
            nxt = next_task()
            if nxt is None:
                return
            t, k = nxt
            try:
                a = run_one(t, cfg, out, runner, attempt=k, sandbox_factory=sandbox_factory,
                            model_override=model_override)
            except Exception as e:  # noqa: BLE001
                a = Attempt(t["instance_id"], t.get("uid", ""), t.get("verifier", ""), t.get("repo", ""),
                            reason=f"INFRA_ERROR: {str(e)[:300]}", attempt=k)
            if prune_images and t.get("env_image") and sandbox_factory is None:
                runner.remove_image(t["env_image"])
            with lock:
                if a.solved:
                    solved_src[a.verifier] += 1
                    solved_repo[a.repo.lower()] += 1
                elif is_final(asdict(a)) and k < attempts and retry_worthy(a):
                    queue.append((t, k + 1))  # another sample later
                if a.reason.startswith("FATAL_API_ERROR"):
                    state["stop"] = f"fatal API error, stopping: {a.reason[:200]}"
                # the model never answering (bad setup, provider down) is not the tasks' fault:
                # stop instead of burning through the queue
                if a.reason.startswith("API_ERROR") and not a.stats.get("completion_tokens"):
                    state["api_fail_streak"] += 1
                    if state["api_fail_streak"] >= max(3, workers):
                        state["stop"] = f"the model API failed {state['api_fail_streak']} times in a row: {a.reason[:200]}"
                elif a.stats.get("completion_tokens"):
                    state["api_fail_streak"] = 0
                with results_fp.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
                results.append(a)
                n_solved = sum(solved_src.values())
                el = (time.time() - t0) / 60
                print(f"[{el:.0f} min] {a.instance_id} (try {k}): {a.reason[:120]}  "
                      f"(solved {n_solved}/{targets['solved_target']}, ~${a.est_cost_usd:.2f})", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"stopped: {state['stop']}", flush=True)
    return results


def is_final(r: dict[str, Any]) -> bool:
    return not str(r.get("reason", "")).startswith(RETRYABLE + ("FATAL_API_ERROR",))


def render_generation_report(rows: list[dict[str, Any]], targets: dict[str, Any] | None = None) -> str:
    by_src: dict[str, Counter] = {}
    reasons: Counter[str] = Counter()
    cost = tokens_in = tokens_out = cached = secs = 0.0
    n_calls = n_think = 0
    ctx: list[int] = []
    final = [r for r in rows if is_final(r)]
    for r in final:
        c = by_src.setdefault(r.get("verifier") or "?", Counter())
        c["attempted"] += 1
        c["solved"] += bool(r.get("solved"))
        reasons[str(r.get("reason", "")).split(":")[0].split(" (")[0]] += 1
        cost += float(r.get("est_cost_usd") or 0)
        st = r.get("stats") or {}
        tokens_in += st.get("prompt_tokens", 0)
        tokens_out += st.get("completion_tokens", 0)
        cached += st.get("cached_tokens", 0)
        n_calls += st.get("tool_calls", 0)
        n_think += st.get("thinking_entries", 0)
        ctx.append(st.get("max_context_tokens", 0))
        secs += float(r.get("seconds") or 0)
    n, solved = len(final), sum(bool(r.get("solved")) for r in final)
    task_ids = {r["instance_id"] for r in final}
    solved_tasks = {r["instance_id"] for r in final if r.get("solved")}
    first = [r for r in final if int(r.get("attempt") or 1) == 1]
    lines = ["# Generation report", "", f"attempts: {n}  solved: {solved}"
             + (f"  ({solved / n:.0%})" if n else ""),
             f"tasks: {len(task_ids)}  solved: {len(solved_tasks)}"
             + (f"  ({len(solved_tasks) / len(task_ids):.0%}; first try "
                f"{sum(bool(r.get('solved')) for r in first) / max(1, len(first)):.0%})" if task_ids else ""), ""]
    lines += ["| source | attempted | solved | rate |", "| --- | --- | --- | --- |"]
    for s, c in sorted(by_src.items()):
        lines.append(f"| {s} | {c['attempted']} | {c['solved']} | {c['solved'] / max(1, c['attempted']):.0%} |")
    lines += ["", "## Outcomes", ""] + [f"- {k}: {v}" for k, v in reasons.most_common()]
    unsolved = [r for r in final if not r.get("solved") and r.get("patch")]
    if unsolved:
        near = sum((r.get("tests") or {}).get("f2p_pass_frac", 0) >= 0.5 for r in unsolved)
        right = sum(bool((r.get("loc") or {}).get("hit")) for r in unsolved)
        lines += ["", f"unsolved with a patch: {len(unsolved)}; half or more of F2P passing: {near}; "
                      f"editing at least one file of the reference fix: {right}"]
    by_size: dict[str, Counter] = {}
    for r in final:
        if r.get("gold"):
            c = by_size.setdefault(size_bucket(r["gold"].get("lines", 0)), Counter())
            c["n"] += 1
            c["solved"] += bool(r.get("solved"))
    if by_size:
        order = ["1-10", "11-30", "31-80", ">80"]
        lines += ["", "## Solve rate by size of the reference fix (changed source lines)", "",
                  "| lines | attempts | solved | rate |", "| --- | --- | --- | --- |"]
        lines += [f"| {k} | {by_size[k]['n']} | {by_size[k]['solved']} | {by_size[k]['solved'] / by_size[k]['n']:.0%} |"
                  for k in order if k in by_size]
    if n:
        lines += ["", "## Cost and time (per attempt)", "",
                  f"- prompt tokens: {tokens_in / n:,.0f} (cached {cached / max(1, tokens_in):.0%})   "
                  f"completion tokens: {tokens_out / n:,.0f}",
                  f"- longest context per attempt: median {sorted(ctx)[len(ctx) // 2]:,} tokens, max {max(ctx):,}",
                  f"- tool calls: {n_calls / n:.1f}   reasoning blocks captured: {n_think / n:.1f}"
                  + ("   <-- NO REASONING CAPTURED: check the model/provider" if n_calls and not n_think else ""),
                  f"- estimated cost: ${cost / n:.3f} per attempt, ${cost / max(1, solved):.3f} per solved task"
                  + ("" if cost else "  (pass --price-in/--price-out to estimate)"),
                  f"- wall time: {secs / n / 60:.1f} min per attempt"]
        if targets and solved:
            need = targets["solved_target"]
            lines.append(f"- projection for {need} solved: ~{need / (solved / n):,.0f} attempts, "
                         f"~${need * cost / solved:,.2f}")
    retry = [r for r in rows if r not in final]
    if retry:
        lines += ["", f"{len(retry)} attempts hit infrastructure/API errors and will be retried on the next run."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# merge shards / runs


def merge_generation(tasks: list[dict[str, Any]], result_files: list[Path], targets: dict[str, Any],
                     out: Path) -> dict[str, int]:
    """Combine results from shards and earlier runs (later files win) and apply the GLOBAL limits:
    solved tasks are accepted in selection-rank order while their source / repo is under its cap.

    Writes results_merged.jsonl (pass as --prior next time), solved.jsonl (task + teacher patch +
    trace path; the SFT input), traces/ for the accepted tasks, and generation_report.md."""
    import shutil

    from ..select import should_attempt

    rows: dict[tuple[str, int], dict[str, Any]] = {}
    trace_src: dict[tuple[str, int], Path] = {}
    for fp in result_files:
        if not fp.exists():
            continue
        for r in iter_jsonl(fp):
            key = (r["instance_id"], int(r.get("attempt") or 1))
            old = rows.get(key)
            if old is not None and is_final(old) and not is_final(r):
                continue  # never replace a finished attempt by a later infrastructure failure
            rows[key] = r
            if r.get("trace_path") and (fp.parent / r["trace_path"]).exists():
                trace_src[key] = fp.parent / r["trace_path"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results_merged.jsonl").open("w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_id = {t["instance_id"]: t for t in tasks}
    first_solved: dict[str, dict[str, Any]] = {}
    for key in sorted(rows, key=lambda k: k[1]):
        r = rows[key]
        if r.get("solved") and r["instance_id"] in by_id:
            first_solved.setdefault(r["instance_id"], r)
    solved = sorted(first_solved.values(), key=lambda r: int(by_id[r["instance_id"]].get("selection_rank", 1 << 30)))
    src_c: Counter[str] = Counter()
    repo_c: Counter[str] = Counter()
    accepted = 0
    (out / "traces").mkdir(exist_ok=True)
    with (out / "solved.jsonl").open("w", encoding="utf-8") as f:
        for r in solved:
            t = by_id[r["instance_id"]]
            if not should_attempt(t, src_c, repo_c, targets):
                continue
            src_c[t.get("verifier") or ""] += 1
            repo_c[(t.get("repo") or "").lower()] += 1
            accepted += 1
            trace = None
            key = (r["instance_id"], int(r.get("attempt") or 1))
            if key in trace_src:
                trace = f"traces/{trace_src[key].name}"
                shutil.copy2(trace_src[key], out / trace)
            teacher = {k: r.get(k) for k in ("model", "patch", "stats", "est_cost_usd", "seconds", "agent_error",
                                             "attempt")}
            f.write(json.dumps({**t, "teacher": {**teacher, "trace": trace}}, ensure_ascii=False) + "\n")
    report = render_generation_report(list(rows.values()), targets)
    report += f"\naccepted under the global limits (solved.jsonl): {accepted} of {targets['solved_target']}\n"
    report += "".join(f"- {s}: {c}\n" for s, c in sorted(src_c.items()))
    (out / "generation_report.md").write_text(report, encoding="utf-8")
    print(report)
    return {"rows": len(rows), "solved": len(solved), "accepted": accepted}
