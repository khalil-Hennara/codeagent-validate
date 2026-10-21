"""Shell script that turns a task image into the workspace the agent sees.

Run once in the agent's container, before swegemma's own git-exclude + baseline commit:

1. The repo moves to /workspace (swegemma's tools are anchored there); the old path
   (/testbed) becomes a symlink, so editable installs and venvs that point at it keep working.
2. Verification material is removed: R2E-Gym's hidden tests and runner. The tests are run
   later in a separate, fresh container, never in this one.
3. Git history is cut back to the base commit: every ref that is not an ancestor of HEAD
   (upstream branches with the real fix, SWE-smith instance branches, future tags, stashes),
   the remotes and the reflogs are removed and unreachable objects are pruned. Past history
   stays, so versioneer/setuptools-scm version strings keep working.
4. SWE-smith: the bug (setup patch) is applied and folded into the base commit, so
   `git log -p` does not show it as a separate change.
"""

from __future__ import annotations

import shlex

SETUP_PATCH_PATH = "/tmp/codeagent_setup.patch"
HIDDEN_PATHS = ["/r2e_tests", "/root/r2e_tests", "/root/run_tests.sh"]
HIDDEN_REPO_PATHS = ["r2e_tests", "run_tests.sh", "expected_test_output.json"]


def prepare_script(src: str = "/testbed", ws: str = "/workspace", *, has_setup_patch: bool = False,
                   setup_patch_path: str = SETUP_PATCH_PATH, hidden_paths: list[str] | None = None) -> str:
    q = shlex.quote
    hidden = " ".join(q(p) for p in (HIDDEN_PATHS if hidden_paths is None else hidden_paths))
    hidden_repo = " ".join(q(p) for p in HIDDEN_REPO_PATHS)
    apply_setup = ""
    if has_setup_patch:
        p = q(setup_patch_path)
        apply_setup = f"""
# 4. SWE-smith: introduce the bug, folded into the base commit
git apply {p} || git apply --reject {p} || patch --batch --fuzz=5 -p1 -i {p}
rm -f {p}
git add -A
git commit -q --amend --no-edit --allow-empty
"""
    return f"""set -eo pipefail
WS={q(ws)}
SRC={q(src)}
# 1. repo at $WS, old path kept as a symlink
if [ "$SRC" != "$WS" ] && [ -d "$SRC" ] && [ ! -L "$SRC" ]; then
  rm -rf "$WS"
  mv "$SRC" "$WS"
  ln -s "$WS" "$SRC"
fi
cd "$WS"
# 1b. installed-package pointers (editable installs, .pth files) name the old path; point them at
# $WS so imports, tracebacks and test runners show /workspace, as in the competition sandbox
if [ "$SRC" != "$WS" ]; then
  for sp in $(find /opt /usr /root "$WS/.venv" -maxdepth 6 -type d \\( -name site-packages -o -name dist-packages \\) 2>/dev/null); do
    for f in "$sp"/*.pth "$sp"/*.egg-link "$sp"/__editable__* "$sp"/*.dist-info/direct_url.json; do
      if [ -f "$f" ] && grep -q -- "$SRC" "$f"; then
        sed -E -i "s#(^|[^A-Za-z0-9_.-])$SRC(/|\\$|[^A-Za-z0-9_.-])#\\1$WS\\2#g" "$f" || true
      fi
    done
  done
fi
# 2. hide verification material
rm -rf {hidden} {hidden_repo}
git config --global user.email agent@example.com
git config --global user.name agent
git config --global --add safe.directory '*'
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  git init -q && git add -A && git commit -qm base
fi
mkdir -p .git/info && printf '.venv/\\n' >> .git/info/exclude
# 3. keep only the base commit's own history
# symbolic refs first (e.g. refs/remotes/origin/HEAD -> origin/maint): deleting a symref and its
# target in one batch is refused by git
git for-each-ref --format='%(symref) %(refname)' | while read -r target ref; do
  if [ -n "$ref" ]; then git symbolic-ref --delete "$ref" || true; fi
done
# then every other ref that is not an ancestor of HEAD, in one pass (fast even with thousands of
# refs, e.g. SWE-smith's per-instance branches)
git for-each-ref --no-merged=HEAD --format='delete %(refname)' | git update-ref --stdin
rm -f .git/ORIG_HEAD .git/FETCH_HEAD .git/packed-refs.lock
git stash clear 2>/dev/null || true
for r in $(git remote); do git remote remove "$r" || git config --remove-section "remote.$r" || true; done
{apply_setup}
git reflog expire --expire=now --all
git gc --prune=now --quiet || true
echo PREPARE_OK
"""
