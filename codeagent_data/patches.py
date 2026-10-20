"""Lenient unified-diff parsing with malformation detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_GIT_RE = re.compile(r'^diff --git "?a/(.+?)"? "?b/(.+?)"?$')


@dataclass
class FileChange:
    old_path: str | None
    new_path: str | None
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False
    added: int = 0
    removed: int = 0
    hunks: int = 0

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


@dataclass
class ParsedPatch:
    files: list[FileChange] = field(default_factory=list)
    malformed_lines: list[tuple[int, str]] = field(default_factory=list)
    hunk_count_mismatches: int = 0

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files if f.path]

    @property
    def modified_existing(self) -> list[str]:
        """Paths that must exist at base_commit for the patch to apply."""
        return [f.old_path for f in self.files if f.old_path and not f.is_new]


def _strip_prefix(p: str) -> str | None:
    p = p.strip().split("\t", 1)[0].strip().strip('"')
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse_patch(text: str) -> ParsedPatch:
    out = ParsedPatch()
    if not text or not text.strip():
        return out
    cur: FileChange | None = None
    rem_old = rem_new = 0
    in_hunk = False
    # split on "\n" only: str.splitlines() would also break at form feeds (\x0c, used as page
    # breaks in some Python files) and other Unicode separators inside code lines
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for lineno, line in enumerate((l[:-1] if l.endswith("\r") else l for l in lines), 1):
        if line.startswith("diff --git "):
            if in_hunk and (rem_old > 0 or rem_new > 0):
                out.hunk_count_mismatches += 1
            in_hunk = False
            m = _DIFF_GIT_RE.match(line)
            cur = FileChange(old_path=m.group(1) if m else None, new_path=m.group(2) if m else None)
            out.files.append(cur)
            continue
        if in_hunk:
            if line.startswith("\\"):
                continue
            if line == "" or line[:1] in (" ", "+", "-"):
                # an empty line is a blank context line (GNU diff --suppress-blank-empty; git accepts it)
                if line == "" or line[0] == " ":
                    rem_old -= 1
                    rem_new -= 1
                elif line[0] == "+":
                    rem_new -= 1
                    if cur:
                        cur.added += 1
                else:
                    rem_old -= 1
                    if cur:
                        cur.removed += 1
                if rem_old <= 0 and rem_new <= 0:
                    in_hunk = False
                continue
            if line.startswith("@@"):
                out.hunk_count_mismatches += 1
                in_hunk = False  # fall through to hunk header handling
            else:
                # A hunk line without a ' ', '+', '-' prefix: whitespace was stripped
                # (e.g. lost leading space on context lines) -> git apply will fail.
                out.malformed_lines.append((lineno, line[:120]))
                rem_old -= 1
                rem_new -= 1
                if rem_old <= 0 and rem_new <= 0:
                    in_hunk = False
                continue
        if line.startswith("--- "):
            if cur is None:  # diff without 'diff --git' header
                cur = FileChange(old_path=None, new_path=None)
                out.files.append(cur)
            cur.old_path = _strip_prefix(line[4:])
            if cur.old_path is None:
                cur.is_new = True
            continue
        if line.startswith("+++ "):
            if cur is not None:
                cur.new_path = _strip_prefix(line[4:])
                if cur.new_path is None:
                    cur.is_deleted = True
            continue
        if line.startswith("new file mode") and cur:
            cur.is_new = True
        elif line.startswith("deleted file mode") and cur:
            cur.is_deleted = True
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            if cur:
                cur.is_binary = True
        elif line.startswith("rename to ") and cur:
            cur.new_path = line[len("rename to "):].strip()
        m = _HUNK_RE.match(line)
        if m:
            in_hunk = True
            rem_old = int(m.group(2)) if m.group(2) is not None else 1
            rem_new = int(m.group(4)) if m.group(4) is not None else 1
            if cur:
                cur.hunks += 1
    if in_hunk and (rem_old > 0 or rem_new > 0):
        out.hunk_count_mismatches += 1
    # files whose new side is /dev/null were deleted; their old path still must exist
    return out


_PROTECTED_NAMES = {
    "conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg",
    ".pytest.ini", "sitecustomize.py", "usercustomize.py",
}


def is_test_path(path: str) -> bool:
    """Mirror of swegemma's protected-test heuristic (verification.py)."""
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    parts = p.split("/")
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    if name.endswith(".py") and any(d in ("tests", "test", "testing") for d in parts[:-1]):
        return True
    return False


def is_protected_config(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name in _PROTECTED_NAMES or name.endswith(".pth")


_OLD_EMPTY_HUNK = re.compile(r"^@@ -0,0 \+\d+(?:,\d+)? @@")


def fix_new_file_headers(patch: str, exists) -> tuple[str, list[str]]:
    """Rewrite file sections that create a file but are written as an edit.

    R2E-Gym patches are rebuilt with difflib from old/new file contents, so a file added by the
    commit comes out as `--- a/x` + `@@ -0,0 +1,N @@` instead of `--- /dev/null` + `new file mode`.
    `git apply` then fails with "No such file". `exists(path)` says whether the file is present at
    base_commit; only sections whose file is absent and whose single hunk starts at -0,0 change.
    """
    if not patch:
        return patch, []
    sections = re.split(r"(?m)^(?=diff --git )", patch)
    out, fixed = [], []
    for sec in sections:
        m = re.match(r"diff --git a/(\S+) b/(\S+)\n", sec)
        if not m or "\nnew file mode" in sec or "\n--- /dev/null" in sec:
            out.append(sec)
            continue
        path = m.group(1)
        hunks = re.findall(r"(?m)^@@ .*$", sec)
        if len(hunks) == 1 and _OLD_EMPTY_HUNK.match(hunks[0]) and not exists(path):
            sec = sec.replace(m.group(0), m.group(0) + "new file mode 100644\n", 1)
            sec = re.sub(r"(?m)^--- a/\S+$", "--- /dev/null", sec, count=1)
            fixed.append(path)
        out.append(sec)
    return "".join(out), fixed
