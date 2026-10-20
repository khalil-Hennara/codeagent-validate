"""Parse FAIL_TO_PASS / PASS_TO_PASS identifiers from the formats seen in SWE datasets."""

from __future__ import annotations

import json
import re
from typing import Any

from .schema import TestRef

# Django runner style: "test_name (package.module.ClassName)"
_DJANGO_RE = re.compile(r"^([A-Za-z_][\w]*) \(([\w.]+)\)\s*$")
# Django >= 5 style sometimes: "test_name (package.module.ClassName.test_name)"


def parse_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_test_id(raw: str, repo: str = "") -> TestRef:
    s = raw.strip()
    if "::" in s:
        file_part, rest = s.split("::", 1)
        name = rest.split("::")[-1]
        return TestRef(raw=raw, framework="pytest", name=name, file_hint=file_part)
    m = _DJANGO_RE.match(s)
    if m:
        name, dotted = m.group(1), m.group(2)
        parts = dotted.split(".")
        if parts and parts[-1] == name:
            parts = parts[:-1]
        # drop trailing CamelCase class name(s)
        while parts and parts[-1][:1].isupper():
            parts = parts[:-1]
        file_hint = None
        if parts:
            mod_path = "/".join(parts) + ".py"
            file_hint = ("tests/" + mod_path) if repo.lower() == "django/django" else mod_path
        return TestRef(raw=raw, framework="django", name=name, file_hint=file_hint)
    name = s.split("[", 1)[0].split()[0] if s else s
    return TestRef(raw=raw, framework="bare", name=name, file_hint=None)


def files_match(hint: str, candidates: list[str]) -> bool:
    h = hint.strip("/")
    for c in candidates:
        c = c.strip("/")
        if c == h or c.endswith("/" + h) or h.endswith("/" + c):
            return True
    return False
