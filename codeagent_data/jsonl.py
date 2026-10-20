"""JSONL helpers.

Records are written with ensure_ascii=False, so strings may contain raw U+2028, U+0085, form
feeds, etc. str.splitlines() treats those as line breaks and cuts records in half, so JSONL is
always split on "\\n" only (which json.dumps never emits inside a record).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="\n") as f:  # newline="\n": split on \n only
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: not valid JSON ({e})") from e


def read_jsonl(*paths: str | Path) -> list[dict[str, Any]]:
    return [r for p in paths for r in iter_jsonl(p)]
