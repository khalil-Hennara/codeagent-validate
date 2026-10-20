"""Consistency checks. Every row must pass these before it costs a container or a teacher call."""

from __future__ import annotations

import re

from .patches import fix_new_file_headers, is_protected_config, is_test_path, parse_patch
from .repocache import RepoCache
from .schema import EnvTier, Severity, TaskRecord
from .testids import files_match, parse_test_id

E, W, I = Severity.ERROR, Severity.WARN, Severity.INFO
_IID_RE = re.compile(r"^(?P<org>[^_/]+(?:_[^_/]+)*?)__(?P<name>.+)-(?P<num>\d+)$")
_GENERIC_TOPDIRS = {"src", "lib", "libs", "python", "pkg", "packages", "tests", "test", "docs", "doc", "scripts"}


def _tok(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 3}


def static_checks(rec: TaskRecord, *, allow_non_python: bool = False) -> None:
    for fld in ("instance_id", "repo", "base_commit", "problem_statement", "gold_patch"):
        if not getattr(rec, fld).strip():
            rec.add("E_MISSING_FIELD", E, f"empty field: {fld}")

    if rec.language != "python" and not allow_non_python:
        rec.add("E_LANG_UNSUPPORTED", E, f"language={rec.language}; competition sandbox is Python-only")

    # instance_id <-> repo
    m = _IID_RE.match(rec.instance_id)
    if m and rec.repo:
        iid_repo = f"{m.group('org')}/{m.group('name')}".lower()
        if iid_repo != rec.repo.lower():
            rec.add("E_IID_REPO_MISMATCH", E, f"instance_id implies {iid_repo}, repo field is {rec.repo}")

    gold = parse_patch(rec.gold_patch)
    rec.gold_files = gold.paths
    if rec.gold_patch.strip() and not gold.files:
        rec.add("E_PATCH_UNPARSEABLE", E, "gold patch has no file headers")
    if gold.malformed_lines:
        ln, txt = gold.malformed_lines[0]
        rec.add("E_PATCH_MALFORMED", E,
                f"{len(gold.malformed_lines)} hunk line(s) lack ' '/+/- prefix (whitespace stripped?) e.g. L{ln}: {txt!r}")
    if gold.hunk_count_mismatches:
        rec.add("W_PATCH_HUNK_COUNTS", W, f"{gold.hunk_count_mismatches} hunk(s) with wrong line counts (git apply --recount may rescue)")

    tp = parse_patch(rec.test_patch)
    rec.test_patch_files = tp.paths
    if tp.malformed_lines:
        ln, txt = tp.malformed_lines[0]
        rec.add("E_TEST_PATCH_MALFORMED", E,
                f"{len(tp.malformed_lines)} hunk line(s) lack ' '/+/- prefix e.g. L{ln}: {txt!r}")

    raw = rec.extra.get("raw") or {}
    has_expected = bool(raw.get("expected_output_json"))  # R2E-Gym verifies against this instead of F2P
    if rec.verifier == "swe_smith" and not rec.setup_patch.strip():
        rec.add("E_NO_SETUP_PATCH", E, "SWE-smith row without setup_patch (the bug): re-run the v3 notebook")
    if not rec.fail_to_pass and not rec.test_patch.strip() and not has_expected:
        rec.add("E_NO_TESTS", E, "no FAIL_TO_PASS, no test_patch and no expected_output_json: cannot verify a fix")
    elif not rec.fail_to_pass and not has_expected:
        rec.add("W_NO_F2P", W, "no FAIL_TO_PASS: harness falls back to test names parsed from test_patch")

    protected = [p for p in rec.gold_files if is_test_path(p) or is_protected_config(p)]
    if protected:
        rec.add("W_GOLD_TOUCHES_PROTECTED", W,
                f"gold patch edits {protected}; swegemma resets agent edits to these files at "
                "evaluation, so a trajectory reproducing this fix teaches edits the competition discards")

    # FAIL_TO_PASS <-> test_patch linkage (catches rows whose columns were joined from different tasks)
    if rec.fail_to_pass and rec.test_patch.strip():
        refs = [parse_test_id(t, rec.repo) for t in rec.fail_to_pass]
        linked = 0
        for r in refs:
            by_file = r.file_hint is not None and files_match(r.file_hint, rec.test_patch_files)
            by_name = bool(r.name) and re.search(rf"\b{re.escape(r.name)}\b", rec.test_patch) is not None
            linked += by_file or by_name
        if linked == 0:
            hints = sorted({r.file_hint for r in refs if r.file_hint})[:3]
            # Only a warning: Django rows often name tests by their docstring and mypy tests live
            # in test-data files, so text matching has false positives. The repo-tree check and
            # stage 1 (F2P must fail on base, pass with gold) are the decisive checks.
            rec.add("W_F2P_NOT_IN_TEST_PATCH", W,
                    f"none of {len(refs)} FAIL_TO_PASS tests named in test_patch "
                    f"(tests point at {hints or [r.name for r in refs[:3]]}, test_patch edits {rec.test_patch_files[:3]})")
        elif linked < len(refs):
            rec.add("W_F2P_PARTIAL_LINK", W, f"{len(refs) - linked}/{len(refs)} FAIL_TO_PASS tests not found in test_patch")

    # gold patch paths vs repo name (heuristic; the repo-tree check below is definitive)
    if rec.gold_files and rec.repo:
        repo_toks = _tok(rec.repo)
        top = {p.split("/", 1)[0].lower() for p in rec.gold_files if "/" in p}
        unrelated = [d for d in top if d not in _GENERIC_TOPDIRS and not (_tok(d) & repo_toks)]
        if top and len(unrelated) == len(top - _GENERIC_TOPDIRS) and unrelated:
            rec.add("W_PATCH_REPO_SUSPECT", W, f"gold patch top-level dirs {sorted(top)} look unrelated to {rec.repo}")

    changed = sum(f.added + f.removed for f in gold.files)
    if changed > 400 or len(gold.files) > 10:
        rec.add("I_LARGE_PATCH", I, f"{len(gold.files)} files / {changed} changed lines")


def repo_checks(rec: TaskRecord, cache: RepoCache, *, apply_check: bool = False) -> None:
    if not rec.repo or not rec.base_commit:
        return
    if not cache.ensure(rec.repo):
        rec.add("W_REPO_UNREACHABLE", W, f"could not clone {rec.repo}; repo checks skipped")
        return
    if not cache.has_commit(rec.repo, rec.base_commit):
        rec.add("E_COMMIT_MISSING", E, f"base_commit {rec.base_commit[:12]} not found in {rec.repo}")
        return
    tree = cache.tree(rec.repo, rec.base_commit)
    # files created by the commit but written as edits (R2E-Gym's rebuilt diffs): fix the headers
    for attr, label in (("gold_patch", "gold"), ("test_patch", "test")):
        fixed, paths = fix_new_file_headers(getattr(rec, attr), lambda f: f in tree)
        if paths:
            setattr(rec, attr, fixed)
            rec.add("I_NEW_FILE_HEADERS_FIXED", I, f"{label} patch: marked {len(paths)} created file(s) as new, e.g. {paths[:2]}")
    gold = parse_patch(rec.gold_patch)
    missing = [p for p in gold.modified_existing if p not in tree]
    if missing:
        rec.add("E_PATCH_PATH_MISSING", E,
                f"gold patch modifies {len(missing)} file(s) absent at base_commit, e.g. {missing[:3]} "
                "(row text likely belongs to a different task)")
    elif gold.modified_existing:
        # the real tree confirms the patch belongs to this repo: drop the name-based guess
        rec.issues = [i for i in rec.issues if i.code != "W_PATCH_REPO_SUSPECT"]
    tp = parse_patch(rec.test_patch)
    tmissing = [p for p in tp.modified_existing if p not in tree]
    if tmissing:
        rec.add("E_TEST_PATCH_PATH_MISSING", E, f"test_patch modifies files absent at base_commit, e.g. {tmissing[:3]}")
    if apply_check and not missing and not tmissing and rec.ok:
        setup = parse_patch(rec.setup_patch)
        touched = setup.modified_existing + gold.modified_existing + tp.modified_existing
        steps = []
        if rec.setup_patch.strip():
            steps.append(("setup", rec.setup_patch, touched))  # SWE-smith: the bug goes in first
        steps.append(("gold", rec.gold_patch, [] if steps else touched))
        if rec.test_patch.strip():
            steps.append(("test", rec.test_patch, []))
        codes = {"setup": "E_SETUP_PATCH_APPLY", "gold": "E_PATCH_APPLY", "test": "E_TEST_PATCH_APPLY"}
        for label, err in cache.apply_check(rec.repo, rec.base_commit, steps):
            rec.add(codes[label], E, f"git apply and patch --fuzz=5 both failed: {err}")


UNSUPPORTED_VERIFIERS = {"swe_bench_pro"}


def harness_checks(rec: TaskRecord) -> None:
    """Decide which test runner stage 1 will use and whether the row has what that runner needs."""
    from .validate.specs import SWEBENCH_SPECS, row_test_cmd, swebench_repo_key

    if rec.env.tier.value != "image":
        return
    raw = rec.extra.get("raw") or {}
    verifier = rec.verifier or ("swebench" if swebench_repo_key(rec.repo) else "generic")
    if verifier in UNSUPPORTED_VERIFIERS:
        rec.env.verify_with = verifier
        rec.add("W_VALIDATE_UNSUPPORTED", W,
                f"stage 1 has no runner for {verifier} yet: kept in tasks/{verifier}.jsonl, not validated")
        return
    if verifier == "swebench":
        if raw.get("eval_script") and raw.get("log_parser"):
            rec.env.verify_with = "row-eval-script"  # the official script shipped with the dataset row
            return
        rec.env.verify_with = "swebench-harness"
        key = swebench_repo_key(rec.repo)
        if not key:
            rec.add("E_UNKNOWN_REPO", E, f"{rec.repo} is not in SWE-bench's specs and the row has no eval_script")
        elif not rec.version:
            rec.add("E_NO_VERSION", E,
                    f"{rec.repo} needs `version` to pick its SWE-bench test spec (row has no eval_script either)")
        elif rec.version not in SWEBENCH_SPECS[key]:
            rec.add("E_UNKNOWN_VERSION", E, f"version {rec.version!r} not in SWE-bench specs for {key}")
        return
    if verifier == "r2e_gym":
        rec.env.verify_with = "r2e_gym"
        if not raw.get("expected_output_json"):
            rec.add("E_NO_EXPECTED_OUTPUT", E, "R2E-Gym row without expected_output_json in raw_json: cannot verify")
        return
    rec.env.verify_with = "generic"
    if not rec.fail_to_pass:
        rec.add("I_DERIVE_F2P", I, "no FAIL_TO_PASS list: stage 1 derives it (fails on base, passes with gold)")
    if not row_test_cmd({**raw, **rec.extra}):
        rec.add("I_DEFAULT_TEST_CMD", I, "no test_cmd in row; stage 1 runs `pytest -rA <test files>`")


def tier_checks(rec: TaskRecord) -> None:
    if rec.env.tier == EnvTier.NONE:
        rec.add("E_NO_ENV", E, "; ".join(rec.env.notes) or "no environment route")
        return
    for n in rec.env.notes:
        rec.add("I_ENV_NOTE", I, n)
