"""Download the harness wheels (swegemma, adk_eval_core, adk_submission) from the organizers'
public Kaggle dataset, so the harness code never has to live in this repository.

    KAGGLE_USERNAME=... KAGGLE_KEY=... python scripts/fetch_harness.py harness_src   # or KAGGLE_API_TOKEN=...
    bash scripts/setup_harness.sh harness_src harness      # unpacks the wheels into harness/

Only the three harness wheels are downloaded (the dataset also holds vLLM/torch wheels for the
GPU inference side, which generation does not need). If the file listing fails, the whole
dataset is downloaded instead.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

DATASET = "metric/gemma-4-developer-agent-wheelhouse"
HARNESS = re.compile(r"(?i)^(swegemma|adk[_-]eval[_-]core|adk[_-]submission)-.+\.whl$")
# the GPU inference stack is not needed for generation and is large
BIG = re.compile(r"(?i)^(torch|vllm|nvidia|triton|xformers|flash|flashinfer|cutlass|cuda|cupy|bitsandbytes|jax)")
# the dependency pins we compare against (requirements-generate.txt)
PINNED = re.compile(r"(?i)^(google[_-]adk|google[_-]genai|litellm)-([^-]+)-")


def list_files(api, dataset: str) -> list[str]:
    names, token = [], None
    while True:
        res = api.dataset_list_files(dataset, page_token=token, page_size=200)
        if getattr(res, "error_message", None):
            raise RuntimeError(res.error_message)
        names += [f.name for f in (res.files or [])]
        token = getattr(res, "next_page_token", None)
        if not token:
            return names


def unpack_zips(dest: Path) -> None:
    """Kaggle sometimes serves a single file as <name>.zip."""
    for z in list(dest.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()


def fetch(dest: Path, dataset: str = DATASET, api=None) -> list[Path]:
    if api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()  # KAGGLE_USERNAME / KAGGLE_KEY, or ~/.kaggle/kaggle.json
    dest.mkdir(parents=True, exist_ok=True)
    try:
        names = list_files(api, dataset)
        print(f"{dataset}: {len(names)} files")
        for n in names:
            m = PINNED.match(n)
            if m:
                print(f"  competition pin: {m.group(1)}=={m.group(2)}")
        wanted = [n for n in names if HARNESS.match(n)]
        if len({HARNESS.match(n).group(1).lower().replace("-", "_") for n in wanted}) < 3:
            # unexpected wheel names: take every small wheel, setup_harness.sh finds the packages
            print("harness wheels not recognised by name; downloading all non-GPU wheels")
            wanted = [n for n in names if n.endswith(".whl") and not BIG.match(n)]
        for n in wanted:
            api.dataset_download_file(dataset, n, path=str(dest), force=True, quiet=True)
    except Exception as e:  # noqa: BLE001  (old CLI versions, API changes): take everything
        print(f"listing failed ({e}); downloading the whole dataset", flush=True)
        api.dataset_download_files(dataset, path=str(dest), force=True, quiet=True, unzip=True)
    unpack_zips(dest)
    for p in dest.rglob("*"):  # drop the GPU stack (the full download includes it) and non-wheels
        if p.is_file() and (BIG.match(p.name) or p.suffix != ".whl"):
            p.unlink()
    got = sorted(dest.rglob("*.whl"))
    if not got:
        raise SystemExit(f"no wheels downloaded from {dataset} "
                         "(did you accept the competition rules with this Kaggle account?)")
    for p in got:
        print(f"  {p.name}")
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()
    fetch(Path(args.dest), args.dataset)


if __name__ == "__main__":
    sys.exit(main())
