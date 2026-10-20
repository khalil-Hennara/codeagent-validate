"""Normalized task schema shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    ERROR = "error"  # row is rejected
    WARN = "warn"  # row is kept but flagged
    INFO = "info"


class EnvTier(str, Enum):
    """How a task will be executed and verified.

    native  - competition-format task: snapshot exists in the swegemma snapshots dir;
              runs on the stock harness (py3.13 universal image + wheels + setup.py).
    image   - a per-task Docker image exists (given in the row or derived from the
              source's naming convention); runs with swegemma tools on an image backend
              and is verified with the dataset's own test runner.
    py313   - (unused: would need the competition wheels, which we do not use for training)
    none    - no execution route; drop.
    """

    NATIVE = "native"
    IMAGE = "image"
    PY313 = "py313"
    NONE = "none"


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity.value, "message": self.message}


@dataclass
class EnvSpec:
    tier: EnvTier = EnvTier.NONE
    image: str | None = None
    image_source: str | None = None  # "row" | "derived:<rule>" | None
    image_verified_convention: bool = False  # naming rule confirmed against a public registry
    workdir_in_image: str | None = None  # e.g. /testbed for SWE-bench images
    verify_with: str | None = None  # "swegemma-pytest" | "swebench-harness" | "pytest-nodeids"
    notes: list[str] = field(default_factory=list)


@dataclass
class TestRef:
    raw: str
    framework: str  # "pytest" | "django" | "bare"
    name: str
    file_hint: str | None = None


@dataclass
class TaskRecord:
    uid: str
    source: str
    split: str
    instance_id: str
    repo: str
    base_commit: str
    language: str
    problem_statement: str
    hints_text: str
    gold_patch: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    env_image: str | None
    env_available: bool | None
    created_at: str = ""
    environment_setup_commit: str = ""
    version: str = ""
    # from the selection notebook (v3)
    verifier: str = ""  # swebench | swe_gym | swe_smith | r2e_gym | swe_bench_pro | promax
    setup_patch: str = ""  # applied before the agent starts (SWE-smith: the bug)
    extra_context: str = ""  # SWE-bench Pro requirements/interface: part of the task text
    fix_commit: str = ""
    # derived
    gold_files: list[str] = field(default_factory=list)
    test_patch_files: list[str] = field(default_factory=list)
    env: EnvSpec = field(default_factory=EnvSpec)
    issues: list[Issue] = field(default_factory=list)
    role: str = "train"  # "train" | "heldout_eval"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    def add(self, code: str, severity: Severity, message: str) -> None:
        self.issues.append(Issue(code, severity, message))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["env"]["tier"] = self.env.tier.value
        d["issues"] = [i.to_dict() for i in self.issues]
        d["ok"] = self.ok
        return d

    def to_swegemma_task(self) -> dict[str, Any]:
        """Row loadable by swegemma.models.load_tasks (extra keys are ignored there).

        hints_text is deliberately empty: swegemma puts it in the agent prompt, and hints often
        describe the fix. The original hints stay in normalized.jsonl for reference.
        """
        ps = self.problem_statement
        if self.extra_context.strip():
            ps = ps.rstrip() + "\n\n" + self.extra_context.strip() + "\n"
        raw = self.extra.get("raw", {}) if isinstance(self.extra.get("raw"), dict) else {}
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": ps,
            "patch": self.gold_patch,
            "test_patch": self.test_patch,
            "hints_text": "",
            "created_at": self.created_at,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
            "environment_setup_commit": self.environment_setup_commit,
            # pipeline extras (ignored by swegemma, used by our runners)
            "uid": self.uid,
            "source": self.source,
            "env_tier": self.env.tier.value,
            "env_image": self.env.image,
            "env_workdir": self.env.workdir_in_image,
            "verify_with": self.env.verify_with,
            "verifier": self.verifier,
            "version": self.version,
            "setup_patch": self.setup_patch,
            "fix_commit": self.fix_commit,
            # official per-task test script + log parser when the dataset ships them (SWE-bench on HF)
            "eval_script": raw.get("eval_script") or "",
            "log_parser": raw.get("log_parser") or "",
            "test_cmd": self.extra.get("test_cmd") or "",
            "expected_output_json": raw.get("expected_output_json") or "",
        }
