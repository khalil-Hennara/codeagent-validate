"""Blobless git clones used to check rows against the real repository at base_commit.

One `git clone --filter=blob:none` per repo: commits + trees only, so `ls-tree` is cheap
and `git show <commit>:<path>` fetches just the blobs we need for apply checks.

Thread-safe: rows are checked in parallel. Clones and explicit fetches take a per-repo lock;
reads (ls-tree, show, lazy blob fetches) run concurrently.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from pathlib import Path

HAVE_GNU_PATCH = shutil.which("patch") is not None


class RepoCache:
    def __init__(self, root: Path, url_template: str = "https://github.com/{repo}.git"):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.url_template = url_template
        self._failed: set[str] = set()
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()
        self._trees: dict[tuple[str, str], frozenset[str]] = {}

    def _lock(self, repo: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[repo.lower()]

    def _dir(self, repo: str) -> Path:
        return self.root / repo.lower().replace("/", "__")

    def _git(self, repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self._dir(repo)), *args], capture_output=True, check=check)

    def ensure(self, repo: str) -> bool:
        with self._lock(repo):
            if repo.lower() in self._failed:
                return False
            d = self._dir(repo)
            if (d / ".git").exists() or (d / "HEAD").exists():
                return True
            r = subprocess.run(
                ["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                 self.url_template.format(repo=repo), str(d)],
                capture_output=True,
            )
            if r.returncode != 0:
                self._failed.add(repo.lower())
                shutil.rmtree(d, ignore_errors=True)
                return False
            return True

    def has_commit(self, repo: str, commit: str) -> bool:
        if self._git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0:
            return True
        # commit may be unreachable from default branches (e.g. PR merge bases); try fetching it
        with self._lock(repo):
            self._git(repo, "fetch", "-q", "--no-write-fetch-head", "--filter=blob:none", "origin",
                      commit.rstrip("^"), check=False)
        return self._git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0

    def tree(self, repo: str, commit: str) -> frozenset[str]:
        key = (repo.lower(), commit)
        if key not in self._trees:
            out = self._git(repo, "ls-tree", "-r", "--name-only", commit).stdout.decode("utf-8", "replace")
            self._trees[key] = frozenset(x for x in out.split("\n") if x)
        return self._trees[key]

    def apply_check(self, repo: str, commit: str, patches: list[tuple[str, str, list[str]]]) -> list[tuple[str, str]]:
        """Apply patches in order on a minimal checkout of the touched files.

        patches: [(label, patch_text, paths_that_must_exist)]. Returns [(label, error)] for failures.
        Each patch is applied for real (so later patches see earlier ones), in the harness order:
        setup patch, gold patch, test patch. Mirrors the Docker validator: `git apply`, then GNU
        `patch --fuzz=5` as a fallback.
        """
        errors: list[tuple[str, str]] = []
        tree = self.tree(repo, commit)
        with tempfile.TemporaryDirectory(prefix="applychk_") as tmp:
            tmpd = Path(tmp)
            for _, _, paths in patches:
                for p in paths:
                    if p in tree and not (tmpd / p).exists():
                        blob = self._git(repo, "show", f"{commit}:{p}", check=False)
                        if blob.returncode == 0:
                            (tmpd / p).parent.mkdir(parents=True, exist_ok=True)
                            (tmpd / p).write_bytes(blob.stdout)
            subprocess.run(["git", "init", "-q"], cwd=tmpd, check=True)
            for label, text, _ in patches:
                pf = tmpd / f".{label}.patch"
                pf.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
                r = subprocess.run(["git", "apply", "--recount", str(pf)], cwd=tmpd, capture_output=True)
                if r.returncode != 0 and HAVE_GNU_PATCH:
                    dry = subprocess.run(["patch", "--batch", "--fuzz=5", "-p1", "--dry-run", "-i", str(pf)],
                                         cwd=tmpd, capture_output=True)
                    if dry.returncode == 0:
                        r = subprocess.run(["patch", "--batch", "--fuzz=5", "-p1", "-i", str(pf)],
                                           cwd=tmpd, capture_output=True)
                if r.returncode != 0:
                    errors.append((label, (r.stderr or r.stdout).decode("utf-8", "replace").strip()[:500]))
                    break  # later patches depend on this one
        return errors
