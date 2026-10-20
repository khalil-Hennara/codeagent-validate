"""Stage 0 tests. Run: python3 tests/test_stage0.py  (or pytest tests/)"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codeagent_data.checks import repo_checks, static_checks  # noqa: E402
from codeagent_data.env import resolve_env  # noqa: E402
from codeagent_data.ingest import to_record  # noqa: E402
from codeagent_data.repocache import RepoCache  # noqa: E402
from codeagent_data.schema import EnvTier  # noqa: E402
from codeagent_data.testids import parse_test_id  # noqa: E402

GOLD = """diff --git a/widget/core.py b/widget/core.py
--- a/widget/core.py
+++ b/widget/core.py
@@ -1,3 +1,3 @@
 def quantile(x, keep_attrs=None):
-    return x
+    return (x, keep_attrs)

"""
TEST = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,5 @@
 from widget.core import quantile

+
+def test_keep_attrs():
+    assert quantile(1, True) == (1, True)
"""


def _make_remote(tmp: Path) -> tuple[str, str]:
    remote = tmp / "remotes" / "acme" / "widget"
    (remote / "widget").mkdir(parents=True)
    (remote / "tests").mkdir()
    (remote / "widget" / "core.py").write_text("def quantile(x, keep_attrs=None):\n    return x\n\n")
    (remote / "tests" / "test_core.py").write_text("from widget.core import quantile\n\n")
    g = ["git", "-C", str(remote), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*g[:3], "init", "-q"], check=True)
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "base"], check=True)
    sha = subprocess.run([*g[:3], "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return f"file://{tmp}/remotes/{{repo}}", sha


def _row(**kw):
    base = dict(uid="t::0", source="swe_gym", split="train", instance_id="acme__widget-1", repo="acme/widget",
                base_commit="", language="python", problem_statement="quantile drops attrs",
                patch=GOLD, test_patch=TEST, f2p=["tests/test_core.py::test_keep_attrs"], p2p=[],
                env_image=None, env_available=True)
    base.update(kw)
    return base


def test_good_row_passes_all_checks():
    with tempfile.TemporaryDirectory() as t:
        url, sha = _make_remote(Path(t))
        rec = to_record(_row(base_commit=sha))
        static_checks(rec)
        rec.env = resolve_env(rec)
        repo_checks(rec, RepoCache(Path(t) / "cache", url_template=url), apply_check=True)
        assert rec.ok, [i.to_dict() for i in rec.issues]
        assert rec.env.tier == EnvTier.IMAGE and rec.env.image == "xingyaoww/sweb.eval.x86_64.acme_s_widget-1:latest"


def test_apply_failure_detected():
    with tempfile.TemporaryDirectory() as t:
        url, sha = _make_remote(Path(t))
        rec = to_record(_row(base_commit=sha, patch=GOLD.replace("-    return x", "-    return y")))
        static_checks(rec)
        repo_checks(rec, RepoCache(Path(t) / "cache", url_template=url), apply_check=True)
        assert any(i.code == "E_PATCH_APPLY" for i in rec.issues)


def test_misaligned_sample_row_rejected_statically():
    row = json.loads((ROOT / "tests/fixtures/sample_row_as_pasted.json").read_text())
    rec = to_record(row)
    static_checks(rec)
    codes = {i.code for i in rec.issues}
    assert "W_F2P_NOT_IN_TEST_PATCH" in codes and not rec.ok  # rejected via E_PATCH_MALFORMED


def test_env_routing():
    def tier(**kw):
        r = to_record(_row(base_commit="abc", **kw))
        return resolve_env(r)
    v = tier(source="swe_bench_verified", instance_id="django__django-10554", repo="django/django")
    assert v.image == "swebench/sweb.eval.x86_64.django_1776_django-10554:latest" and v.verify_with == "swebench-harness"
    assert tier(env_image="my/img:1").image_source == "row"
    assert tier(source="my_scraped_set").tier == EnvTier.NONE
    assert tier(source="my_scraped_set", env_available=False).tier == EnvTier.NONE
    assert tier(source="competition").tier == EnvTier.NATIVE


def test_test_id_parsing():
    d = parse_test_id("test_union (queries.test_qs_combinators.QuerySetSetOperationTests)", "django/django")
    assert d.framework == "django" and d.file_hint == "tests/queries/test_qs_combinators.py"
    p = parse_test_id("tests/test_x.py::TestA::test_b[1-2]")
    assert p.framework == "pytest" and p.file_hint == "tests/test_x.py" and p.name == "test_b[1-2]"


def test_official_meta_detects_and_repairs():
    from codeagent_data.ingest import apply_official
    row = json.loads((ROOT / "tests/fixtures/sample_row_as_pasted.json").read_text())
    meta = {"django__django-10554": {"instance_id": "django__django-10554", "version": "3.0",
            "base_commit": row["base_commit"], "problem_statement": "Union queryset with ordering breaks",
            "patch": GOLD, "test_patch": TEST, "FAIL_TO_PASS": row["f2p"], "PASS_TO_PASS": []}}
    rec = to_record(row)
    apply_official(rec, meta, repair=False)
    assert rec.version == "3.0" and any(i.code == "E_DIFFERS_FROM_OFFICIAL" for i in rec.issues)
    rec = to_record(row)
    apply_official(rec, meta, repair=True)
    assert rec.problem_statement.startswith("Union") and rec.gold_patch == GOLD


def test_smith_base_commit_from_instance_id():
    rec = to_record(_row(source="swe_smith", verifier="swe_smith", base_commit=None,
                         instance_id="oauthlib__oauthlib.1fd52536.pr_879", setup_patch=GOLD, test_patch=""))
    assert rec.base_commit == "1fd52536"


def test_r2e_new_file_headers_fixed_in_repo_checks():
    import difflib
    with tempfile.TemporaryDirectory() as t:
        url, sha = _make_remote(Path(t))
        new = "diff --git a/docs/new.rst b/docs/new.rst\n" + "\n".join(
            difflib.unified_diff([], ["hello"], "a/docs/new.rst", "b/docs/new.rst", lineterm="")) + "\n"
        rec = to_record(_row(base_commit=sha, patch=GOLD + new, test_patch=TEST))
        static_checks(rec)
        repo_checks(rec, RepoCache(Path(t) / "cache", url_template=url), apply_check=True)
        codes = {i.code for i in rec.issues}
        assert rec.ok and "I_NEW_FILE_HEADERS_FIXED" in codes, [i.to_dict() for i in rec.issues]
        assert "--- /dev/null" in rec.gold_patch


def test_parallel_ingest_cli():
    from codeagent_data.cli import main
    with tempfile.TemporaryDirectory() as t:
        url, sha = _make_remote(Path(t))
        rows = [_row(base_commit=sha, instance_id=f"acme__widget-{i}", env_image="img:1") for i in range(12)]
        inp = Path(t) / "rows.jsonl"
        inp.write_text("\n".join(json.dumps(r) for r in rows))
        import codeagent_data.repocache as rc
        old = rc.RepoCache.__init__.__defaults__
        rc.RepoCache.__init__.__defaults__ = (url,)
        try:
            main(["ingest", str(inp), "--out", f"{t}/out", "--repo-cache", f"{t}/cache", "--apply-check", "--workers", "6"])
        finally:
            rc.RepoCache.__init__.__defaults__ = old
        lines = (Path(t) / "out/tasks/swe_gym.jsonl").read_text().splitlines()
        assert len(lines) == 12, (Path(t) / "out/report.md").read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
