"""Environment routing: decide how each task is executed and verified.

swegemma itself knows exactly one environment: the universal `swebench-sandbox` image
(Python 3.13, network off) + a repo snapshot + the competition wheels dir + sandbox/setup.py.
That works for the competition's own repos, but not for e.g. Django 3.0 or xarray 0.12
from SWE-bench, which need their original Python/conda environments. Those rows must run
in their per-task image, with swegemma's *tools* on top (same tool schema the Gemma agent
sees at inference), and must be verified with the dataset's own test command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .schema import EnvSpec, EnvTier, TaskRecord


def norm_source(source: str) -> str:
    s = (source or "").strip().lower()
    s = s.rsplit("/", 1)[-1]  # "princeton-nlp/SWE-bench_Verified" -> "swe-bench_verified"
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    aliases = {
        "swe_bench_verified": "swe_bench_verified",
        "swebench_verified": "swe_bench_verified",
        "swe_bench": "swe_bench",
        "swebench": "swe_bench",
        "swe_bench_lite": "swe_bench_lite",
        "swebench_lite": "swe_bench_lite",
        "swe_rebench": "swe_rebench",
        "swerebench": "swe_rebench",
        "swe_gym": "swe_gym",
        "swegym": "swe_gym",
        "swe_smith": "swe_smith",
        "swesmith": "swe_smith",
        "competition": "competition",
        "gemma4_agent": "competition",
    }
    return aliases.get(s, s)


@dataclass(frozen=True)
class ImageRule:
    namespace: str
    workdir: str
    verify_with: str
    convention_verified: bool  # confirmed on Docker Hub; others must be checked before use
    sep: str = "_1776_"  # replaces "__" in the instance id (SWE-Gym uses "_s_")

    def image_for(self, instance_id: str) -> str:
        key = instance_id.replace("__", self.sep).lower()
        return f"{self.namespace}/sweb.eval.x86_64.{key}:latest"


# Rules for sources whose rows may arrive with env_image = null but env_available = true.
IMAGE_RULES: dict[str, ImageRule] = {
    "swe_bench_verified": ImageRule("swebench", "/testbed", "swebench-harness", True),
    "swe_bench": ImageRule("swebench", "/testbed", "swebench-harness", True),
    "swe_bench_lite": ImageRule("swebench", "/testbed", "swebench-harness", True),
    "swe_rebench": ImageRule("swerebench", "/testbed", "pytest-nodeids", True),
    # e.g. xingyaoww/sweb.eval.x86_64.conan-io_s_conan-15422 (confirmed on Docker Hub)
    "swe_gym": ImageRule("xingyaoww", "/testbed", "generic", True, sep="_s_"),
}

DEFAULT_EVAL_RESERVED = frozenset({"swe_bench_verified"})

# Repo location inside images that come with the row (no naming rule needed).
ROW_IMAGE_WORKDIRS: dict[str, tuple[str, bool]] = {  # source -> (workdir, confirmed)
    "r2e_gym": ("/testbed", True),            # R2E-Gym runtime: repo_path="/testbed"
    "swe_smith": ("/testbed", True),          # SWE-smith images follow the SWE-bench layout
    "swe_bench_promax": ("/testbed", False),  # SWE-Factory images; confirm on the smoke test
    "swe_bench_pro": ("/app", False),
}


def snapshot_exists(snapshots_dir: Path | None, instance_id: str, repo: str) -> bool:
    """Replicates swegemma.deduplication.resolve_task_snapshot_paths (existence only)."""
    if snapshots_dir is None or not snapshots_dir.is_dir():
        return False
    s = snapshots_dir
    clean = re.sub(r"[^a-z0-9]+", "_", repo.lower()).strip("_")
    if (s / "fallbacks" / f"{instance_id}.tgz").exists():
        return True
    for base in (s / "base_snapshots" / f"base_{clean}.tgz", s / f"base_{clean}.tgz"):
        if base.exists() and any(
            p.exists() for p in (s / "patches" / f"{instance_id}.patch", s / f"{instance_id}.patch")
        ):
            return True
    return any((s / f"{instance_id}{ext}").exists() for ext in (".tgz", ".tar.gz"))


def resolve_env(rec: TaskRecord, *, snapshots_dir: Path | None = None) -> EnvSpec:
    src = norm_source(rec.source)
    spec = EnvSpec()

    # 1) competition-native (only used for evaluation runs, never for training data)
    if src == "competition" or snapshot_exists(snapshots_dir, rec.instance_id, rec.repo):
        spec.tier = EnvTier.NATIVE
        spec.workdir_in_image = "/workspace"
        spec.verify_with = "swegemma-pytest"
        return spec

    # 2) explicit image in the row
    if rec.env_image:
        spec.tier = EnvTier.IMAGE
        spec.image = rec.env_image
        spec.image_source = "row"
        rule = IMAGE_RULES.get(src)
        workdir, confirmed = ROW_IMAGE_WORKDIRS.get(src, (rule.workdir if rule else "/testbed", rule is not None))
        spec.workdir_in_image = workdir
        spec.verify_with = rule.verify_with if rule else "generic"
        spec.image_verified_convention = True
        if not confirmed:
            spec.notes.append(f"repo location in {src or 'unknown-source'} images assumed {workdir}; confirm on the smoke test")
        return spec

    # 3) image derivable from the source's naming convention
    rule = IMAGE_RULES.get(src)
    if rule and rec.env_available is not False:
        spec.tier = EnvTier.IMAGE
        spec.image = rule.image_for(rec.instance_id)
        spec.image_source = f"derived:{rule.namespace}"
        spec.image_verified_convention = rule.convention_verified
        spec.workdir_in_image = rule.workdir
        spec.verify_with = rule.verify_with
        if not rule.convention_verified:
            spec.notes.append("image naming rule not yet confirmed; run `docker manifest inspect`")
        return spec

    # 4) no image and no naming rule: we do not use the competition's wheels/universal
    #    sandbox for training data, so there is no environment to run this row in.
    spec.tier = EnvTier.NONE
    spec.notes.append(
        "no Docker image (env_image empty and no naming rule for this source); "
        "would need environment construction"
    )
    return spec
