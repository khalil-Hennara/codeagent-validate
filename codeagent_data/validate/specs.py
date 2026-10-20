"""Build the eval script + log parser for a task.

Two routes:
  swebench-harness : repo/version listed in SWE-bench's per-version specs (swebench==4.1.0).
                     We reuse its eval script (conda env, test directives, Django runtests,
                     sympy bin/test, ...) and its per-repo log parser, so our verdicts match
                     the official harness exactly.
  generic          : anything else with an image (SWE-Gym, SWE-rebench, custom): a script with
                     the same shape, the row's `test_cmd` (or `pytest -rA <test files>`), and
                     the row's `log_parser` (or the pytest parser).

Pinned to swebench 4.1.0: from 5.0 the specs moved into the HF dataset rows.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

try:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as SWEBENCH_SPECS
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.log_parsers import python as _py_parsers
    from swebench.harness.test_spec.test_spec import make_test_spec

    HAVE_SWEBENCH = True
except Exception:  # pragma: no cover - optional dependency
    SWEBENCH_SPECS, MAP_REPO_TO_PARSER, _py_parsers, make_test_spec = {}, {}, None, None
    HAVE_SWEBENCH = False

from ..patches import is_test_path, parse_patch

START = ">>>>> Start Test Output"
END = ">>>>> End Test Output"
_HEREDOC = "EOF_CODEAGENT_7731"


@dataclass
class EvalPlan:
    route: str  # "swebench-harness" | "generic"
    script: str
    parse: Callable[[str], dict[str, str]]
    workdir: str
    base_commit: str


def row_test_cmd(extra: dict[str, Any]) -> str | None:
    if extra.get("test_cmd"):
        return str(extra["test_cmd"])
    cfg = extra.get("install_config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except json.JSONDecodeError:
            cfg = None
    if isinstance(cfg, dict) and cfg.get("test_cmd"):
        tc = cfg["test_cmd"]
        return " ".join(tc) if isinstance(tc, list) else str(tc)
    return None


def _slice(log: str) -> str:
    if START in log and END in log:
        return log.split(START, 1)[1].split(END, 1)[0]
    return log


class UnsupportedVerifier(Exception):
    pass


UNSUPPORTED = {"swe_bench_pro"}


def swebench_repo_key(repo: str) -> str | None:
    """Case-insensitive lookup into SWE-bench's specs (the notebook lower-cases repo names)."""
    if not repo:
        return None
    if repo in SWEBENCH_SPECS:
        return repo
    low = repo.lower()
    return next((k for k in SWEBENCH_SPECS if k.lower() == low), None)


def _parser_by_name(name: str | None, repo: str):
    fn = getattr(_py_parsers, str(name), None) if (_py_parsers and name) else None
    if fn is None and repo in MAP_REPO_TO_PARSER:
        fn = MAP_REPO_TO_PARSER[repo]
    return fn or _fallback_pytest_parser


def build_plan(task: dict[str, Any]) -> EvalPlan:
    """Pick the test script + parser for a task, in order of trust:

    1. row-eval-script : the dataset row ships the official eval.sh + log parser name (SWE-bench on HF).
    2. swebench-harness: SWE-bench 4.1 per-repo/version spec.
    3. generic         : pytest on the test files (row test_cmd if given).
    """
    verifier = task.get("verifier") or ""
    if verifier in UNSUPPORTED:
        raise UnsupportedVerifier(f"no stage-1 runner for verifier={verifier} yet")
    repo, version = task["repo"], str(task.get("version") or "")
    stub = SimpleNamespace(instance_id=task["instance_id"], repo=repo, version=version)

    if verifier == "r2e_gym":
        return _r2e_plan(task)

    if task.get("eval_script") and task.get("log_parser"):
        parser = _parser_by_name(task["log_parser"], repo)
        return EvalPlan("row-eval-script", task["eval_script"], lambda log: parser(_slice(log), stub),
                        task.get("env_workdir") or "/testbed", task["base_commit"])

    key = swebench_repo_key(repo) if HAVE_SWEBENCH else None
    if key and version in SWEBENCH_SPECS[key] and task.get("test_patch"):
        inst = {
            "instance_id": task["instance_id"], "repo": key, "version": version,
            "base_commit": task["base_commit"], "patch": task.get("patch", ""),
            "test_patch": task["test_patch"],
            "FAIL_TO_PASS": json.dumps(task.get("FAIL_TO_PASS", [])),
            "PASS_TO_PASS": json.dumps(task.get("PASS_TO_PASS", [])),
            "environment_setup_commit": task.get("environment_setup_commit") or task["base_commit"],
            "problem_statement": "", "hints_text": "", "created_at": "",
        }
        spec = make_test_spec(inst)
        parser = MAP_REPO_TO_PARSER[key]
        return EvalPlan("swebench-harness", spec.eval_script, lambda log: parser(_slice(log), spec),
                        "/testbed", task["base_commit"])
    return _generic_plan(task)


def _test_files_from_ids(ids: list[str]) -> list[str]:
    out: list[str] = []
    for t in ids:
        f = str(t).split("::", 1)[0].strip()
        if f.endswith(".py") and f not in out:
            out.append(f)
    return out


def _generic_plan(task: dict[str, Any]) -> EvalPlan:
    workdir = task.get("env_workdir") or "/testbed"
    base = task["base_commit"]
    test_patch = task.get("test_patch") or ""
    tp = parse_patch(test_patch)
    existing = [f.old_path for f in tp.files if f.old_path and not f.is_new]
    cmd = task.get("test_cmd") or row_test_cmd(task.get("extra", {}) or {})
    if not cmd:
        files = [p for p in tp.paths if p.endswith(".py")]
        targets = [p for p in files if is_test_path(p)] or files
        ids = [str(t) for t in list(task.get("FAIL_TO_PASS") or []) + list(task.get("PASS_TO_PASS") or [])]
        if not targets and test_patch.strip() and ids and all("::" in t for t in ids):
            # test_patch only edits data files (e.g. mypy's test-data/*.test): the owning .py file
            # holds thousands of cases, so run exactly the listed tests instead
            targets = list(dict.fromkeys(ids))
        if not targets:  # no test_patch (SWE-smith): run the files the F2P/P2P tests live in
            targets = _test_files_from_ids(ids)
        if not targets:
            raise UnsupportedVerifier("cannot tell which tests to run (no test_patch, no pytest node ids, no test_cmd)")
        cmd = "pytest -rA " + " ".join(shlex.quote(p) for p in targets)
    reset = f"git checkout {base} -- {' '.join(shlex.quote(p) for p in existing)}" if existing else ":"
    lines = [
        "#!/bin/bash",
        "set -uxo pipefail",
        "if [ -f /opt/miniconda3/bin/activate ]; then source /opt/miniconda3/bin/activate; conda activate testbed || true; fi",
        f"cd {shlex.quote(workdir)}",
        f"git config --global --add safe.directory {shlex.quote(workdir)}",
    ]
    if test_patch.strip():
        lines += [reset, f"git apply -v - <<'{_HEREDOC}'", test_patch.rstrip("\n"), _HEREDOC]
    lines += [f": '{START}'", cmd, f": '{END}'"]
    if test_patch.strip():
        lines.append(reset)
    script = "\n".join(lines) + "\n"
    parser = _parser_by_name(task.get("log_parser") or (task.get("extra") or {}).get("log_parser") or "parse_log_pytest",
                             task["repo"])
    stub = SimpleNamespace(instance_id=task["instance_id"], repo=task["repo"], version=task.get("version", ""))
    return EvalPlan("generic", script, lambda log: parser(_slice(log), stub), workdir, base)


# ---- R2E-Gym --------------------------------------------------------------------------------
# Mirrors r2egym.agenthub.runtime.docker.DockerRuntime.setup_env / run_tests for R2E images:
# run_tests.sh and r2e_tests are moved from /testbed to /root (hidden from the agent), the hidden
# tests at /r2e_tests go to /root/r2e_tests with a symlink back into the repo, then
# `bash /root/run_tests.sh` runs them. The repo's venv lives in /testbed/.venv.
_R2E_SETUP = """cd /testbed
mkdir -p /root/.local/bin
ln -sf /testbed/.venv /root/.venv 2>/dev/null || true
ln -sf /testbed/.venv/bin/python /root/.local/bin/python 2>/dev/null || true
ln -sf /testbed/.venv/bin/python /root/.local/bin/python3 2>/dev/null || true
find /testbed/.venv/bin -type f -executable -exec ln -sf {} /root/.local/bin/ \\; 2>/dev/null || true
export PATH=/root/.local/bin:$PATH
find . -name '*.pyc' -delete 2>/dev/null; find . -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
for f in run_tests.sh r2e_tests; do [ -e /testbed/$f ] && [ ! -L /testbed/$f ] && mv /testbed/$f /root/$f; done
if [ -d /r2e_tests ]; then find /r2e_tests -name '*.pyc' -delete; rm -rf /root/r2e_tests; mv /r2e_tests /root/r2e_tests; fi
[ -e /testbed/r2e_tests ] || ln -s /root/r2e_tests /testbed/r2e_tests
"""


def _r2e_plan(task: dict[str, Any]) -> EvalPlan:
    script = "\n".join([
        "#!/bin/bash",
        "set -uxo pipefail",
        _R2E_SETUP.rstrip("\n"),
        f": '{START}'",
        "bash /root/run_tests.sh",
        f": '{END}'",
        "",
    ])
    return EvalPlan("r2e_gym", script, lambda log: r2e_normalize(parse_log_r2e(_slice(log))), "/testbed",
                    task.get("base_commit", ""))


def parse_log_r2e(log: str | None) -> dict[str, str]:
    """r2egym.repo_analysis.execution_log_parser.parse_log_pytest (used for every R2E repo)."""
    if not log or "short test summary info" not in log:
        return {}
    out: dict[str, str] = {}
    for line in log.split("short test summary info")[1].strip().split("\n"):
        if "PASSED" in line:
            out[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            out[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            out[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return out


def r2e_normalize(d: dict[str, str]) -> dict[str, str]:
    """R2E's comparison keys: ANSI colour codes removed, text after ' - ' dropped."""
    import re as _re
    clean = {_re.sub(r"\u001b\[\d+m", "", k): v for k, v in d.items()}
    return {k.split(" - ")[0]: clean[k] for k in sorted(clean)}


def _fallback_pytest_parser(log: str, _spec: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in log.split("\n"):
        for status in ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL"):
            if line.startswith(status + " "):
                case = line[len(status) + 1:].split(" - ", 1)[0].strip()
                out[case] = status
    return out
