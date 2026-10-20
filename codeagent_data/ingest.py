"""Read raw rows (jsonl / json / parquet) and map heterogeneous field names to TaskRecord."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import TaskRecord
from .testids import parse_list

_ALIASES: dict[str, tuple[str, ...]] = {
    "instance_id": ("instance_id", "id", "task_id"),
    "repo": ("repo_canon", "repo", "repository"),
    "base_commit": ("base_commit", "commit", "base_sha"),
    "problem_statement": ("problem_statement", "issue", "issue_text", "problem"),
    "hints_text": ("hints", "hints_text"),
    "gold_patch": ("patch", "gold_patch", "fix_patch"),
    "test_patch": ("test_patch",),
    "fail_to_pass": ("f2p", "FAIL_TO_PASS", "fail_to_pass"),
    "pass_to_pass": ("p2p", "PASS_TO_PASS", "pass_to_pass"),
    "env_image": ("env_image", "image_name", "docker_image", "image"),
    "env_available": ("env_available",),
    "source": ("source", "dataset"),
    "split": ("split",),
    "language": ("language", "lang"),
    "created_at": ("created_at",),
    "environment_setup_commit": ("environment_setup_commit",),
    "version": ("version",),
    "verifier": ("verifier",),
    "setup_patch": ("setup_patch",),
    "extra_context": ("extra_context",),
    "fix_commit": ("fix_commit",),
}


def _pick(row: dict[str, Any], key: str) -> Any:
    for k in _ALIASES[key]:
        if k in row and row[k] is not None and row[k] != "":
            return row[k]
    return None


_SMITH_COMMIT = re.compile(r"^[^.]+\.([0-9a-f]{7,40})(?:\.|$)")


def smith_base_commit(instance_id: str, image: str | None) -> str:
    """SWE-smith has no base_commit column: the snapshot commit is in the id
    ('oauthlib__oauthlib.1fd52536.pr_879') and in the image tag ('...oauthlib.1fd52536')."""
    m = _SMITH_COMMIT.match(instance_id or "")
    if m:
        return m.group(1)
    m = re.search(r"\.([0-9a-f]{7,40})(?::[\w.-]+)?$", image or "")
    return m.group(1) if m else ""


def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def iter_raw_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        from .jsonl import iter_jsonl
        yield from iter_jsonl(path)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else [data])
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pip install pyarrow to read parquet") from e
        table = pq.read_table(path)
        yield from table.to_pylist()
    else:
        raise SystemExit(f"unsupported input format: {path}")


def load_official_meta(path: Path) -> dict[str, dict[str, Any]]:
    """instance_id -> official row (from scripts/export_official_meta.py)."""
    return {r["instance_id"]: r for r in iter_raw_rows(path)}


def _fingerprint(text: str) -> str:
    return "".join(text.split())[:400]


def apply_official(rec: TaskRecord, meta: dict[str, dict[str, Any]], *, repair: bool) -> None:
    """Fill version/env commit from the official dataset and detect (or repair) misaligned text."""
    from .schema import Severity

    off = meta.get(rec.instance_id)
    if off is None:
        return
    rec.version = rec.version or str(off.get("version") or "")
    rec.environment_setup_commit = rec.environment_setup_commit or str(off.get("environment_setup_commit") or "")
    fields = {
        "problem_statement": "problem_statement", "gold_patch": "patch", "test_patch": "test_patch",
    }
    differs = [mine for mine, theirs in fields.items()
               if off.get(theirs) and _fingerprint(getattr(rec, mine)) != _fingerprint(str(off[theirs]))]
    if rec.base_commit and off.get("base_commit") and rec.base_commit != off["base_commit"]:
        differs.append("base_commit")
    if not differs:
        return
    if repair:
        for mine, theirs in fields.items():
            if off.get(theirs):
                setattr(rec, mine, str(off[theirs]))
        rec.base_commit = str(off.get("base_commit") or rec.base_commit)
        rec.fail_to_pass = parse_list(off.get("FAIL_TO_PASS")) or rec.fail_to_pass
        rec.pass_to_pass = parse_list(off.get("PASS_TO_PASS")) or rec.pass_to_pass
        rec.add("I_REPAIRED_FROM_OFFICIAL", Severity.INFO, f"replaced {differs} with the official row")
    else:
        rec.add("E_DIFFERS_FROM_OFFICIAL", Severity.ERROR,
                f"{differs} differ from the official {rec.instance_id} (use --repair-from-official)")


def to_record(row: dict[str, Any]) -> TaskRecord:
    """Map one raw row (selection-notebook output or a plain SWE-bench-style row) to a TaskRecord."""
    source = str(_pick(row, "source") or "unknown")
    split = str(_pick(row, "split") or "")
    iid = str(_pick(row, "instance_id") or "").strip()
    uid = str(row.get("uid") or (f"{source}::{split}::{row.get('row_idx')}" if row.get("row_idx") is not None else iid))
    known = {k for aliases in _ALIASES.values() for k in aliases} | {"uid", "row_idx", "raw_json"}
    extra = {k: v for k, v in row.items() if k not in known}
    if row.get("raw_json"):  # full original dataset row saved by the selection notebook
        try:
            extra["raw"] = json.loads(row["raw_json"]) if isinstance(row["raw_json"], str) else dict(row["raw_json"])
        except (TypeError, ValueError):
            extra["raw"] = {}
    rec = TaskRecord(
        uid=uid,
        source=source,
        split=split,
        instance_id=iid,
        repo=str(_pick(row, "repo") or "").strip(),
        base_commit=str(_pick(row, "base_commit") or "").strip(),
        language=str(_pick(row, "language") or "python").strip().lower(),
        problem_statement=str(_pick(row, "problem_statement") or ""),
        hints_text=str(_pick(row, "hints_text") or ""),
        gold_patch=str(_pick(row, "gold_patch") or ""),
        test_patch=str(_pick(row, "test_patch") or ""),
        fail_to_pass=parse_list(_pick(row, "fail_to_pass")),
        pass_to_pass=parse_list(_pick(row, "pass_to_pass")),
        env_image=(str(_pick(row, "env_image")).strip() or None) if _pick(row, "env_image") else None,
        env_available=_as_bool(_pick(row, "env_available")),
        created_at=str(_pick(row, "created_at") or ""),
        environment_setup_commit=str(_pick(row, "environment_setup_commit") or ""),
        version=str(_pick(row, "version") or ""),
        verifier=str(_pick(row, "verifier") or ""),
        setup_patch=str(_pick(row, "setup_patch") or ""),
        extra_context=str(_pick(row, "extra_context") or ""),
        fix_commit=str(_pick(row, "fix_commit") or ""),
        extra=extra,
    )
    if not rec.base_commit and (rec.verifier == "swe_smith" or rec.source == "swe_smith"):
        rec.base_commit = smith_base_commit(rec.instance_id, rec.env_image)
    return rec
