"""scripts/fetch_harness.py + setup_harness.sh with a fake Kaggle API (no network).

    HARNESS_SRC=/path/with/swegemma,adk_eval_core,adk_submission python3 tests/test_fetch_harness.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_harness  # noqa: E402

PKGS = {"swegemma": "swegemma-0.3.1-py3-none-any.whl", "adk_eval_core": "adk_eval_core-0.2.0-py3-none-any.whl",
        "adk_submission": "adk_submission-0.2.0-py3-none-any.whl"}
OTHER = ["vllm-0.11.0+cu128-cp312-cp312-linux_x86_64.whl", "google_adk-2.12.0-py3-none-any.whl",
         "litellm-1.106.0-py3-none-any.whl"]


def find_pkg(src: Path, pkg: str) -> Path:
    return min((p.parent for p in src.rglob("__init__.py") if p.parent.name == pkg), key=lambda p: len(str(p)))


class FakeApi:
    def __init__(self, store: Path, page: int = 2, fail_list: bool = False):
        self.store, self.page, self.fail_list, self.downloaded = store, page, fail_list, []

    def dataset_list_files(self, dataset, page_token=None, page_size=20):
        if self.fail_list:
            raise RuntimeError("401")
        names = sorted(p.name for p in self.store.iterdir())
        i = int(page_token or 0)
        nxt = str(i + self.page) if i + self.page < len(names) else ""
        return SimpleNamespace(error_message="", next_page_token=nxt,
                               files=[SimpleNamespace(name=n) for n in names[i:i + self.page]])

    def dataset_download_file(self, dataset, name, path, force, quiet):
        self.downloaded.append(name)
        if name.startswith("adk_eval"):  # served zipped, like Kaggle does sometimes
            with zipfile.ZipFile(Path(path) / f"{name}.zip", "w") as zf:
                zf.write(self.store / name, name)
        else:
            shutil.copy(self.store / name, path)

    def dataset_download_files(self, dataset, path, force, quiet, unzip):
        for p in self.store.iterdir():
            shutil.copy(p, path)


def main() -> None:
    src = Path(os.environ.get("HARNESS_SRC", "/home/claude/src"))
    store = Path(tempfile.mkdtemp())
    for pkg, whl in PKGS.items():
        with zipfile.ZipFile(store / whl, "w") as zf:  # a wheel: the package at the archive root
            base = find_pkg(src, pkg)
            for f in base.rglob("*.py"):
                zf.write(f, f"{pkg}/{f.relative_to(base)}")
            zf.writestr(f"{whl.split('-py3')[0]}.dist-info/METADATA", "Name: x\n")
    for n in OTHER:
        with zipfile.ZipFile(store / n, "w") as zf:
            zf.writestr("dummy/__init__.py", "")
    for fail in (False, True):
        dest = Path(tempfile.mkdtemp())
        api = FakeApi(store, fail_list=fail)
        got = fetch_harness.fetch(dest, api=api)
        names = sorted(p.name for p in got)
        assert "vllm" not in " ".join(names), names  # GPU stack never kept
        if not fail:
            assert names == sorted(PKGS.values()), names
            assert sorted(api.downloaded) == sorted(PKGS.values()), api.downloaded  # no vllm download
        out = Path(tempfile.mkdtemp()) / "harness"
        r = subprocess.run(["bash", str(ROOT / "scripts/setup_harness.sh"), str(dest), str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "harness OK" in r.stdout, r.stdout + r.stderr
    # wheels under other distribution names: everything small is fetched, packages found by content
    store2 = Path(tempfile.mkdtemp())
    for i, whl in enumerate(PKGS.values()):
        shutil.copy(store / whl, store2 / f"kaggle_harness_part{i}-1.0-py3-none-any.whl")
    shutil.copy(store / OTHER[0], store2 / OTHER[0])
    dest = Path(tempfile.mkdtemp())
    api = FakeApi(store2)
    fetch_harness.fetch(dest, api=api)
    assert not any("vllm" in n for n in api.downloaded), api.downloaded
    r = subprocess.run(["bash", str(ROOT / "scripts/setup_harness.sh"), str(dest), str(Path(tempfile.mkdtemp()) / "h")],
                       capture_output=True, text=True)
    assert "harness OK" in r.stdout, r.stdout + r.stderr
    try:
        fetch_harness.fetch(Path(tempfile.mkdtemp()), api=FakeApi(Path(tempfile.mkdtemp())))
        raise AssertionError("an empty download must fail")
    except SystemExit as e:
        assert "no wheels" in str(e)
    print("ok test_fetch_harness")


if __name__ == "__main__":
    main()
