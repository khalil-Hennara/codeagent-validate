"""Export official task metadata (version, text fields, test lists) from Hugging Face.

Run on a machine with HF access:
    pip install datasets
    python scripts/export_official_meta.py --out official_meta.jsonl

Then: python -m codeagent_data ingest ... --official-meta official_meta.jsonl [--repair-from-official]
"""

from __future__ import annotations

import argparse
import json

DEFAULT = [
    ("princeton-nlp/SWE-bench", "test"),
    ("princeton-nlp/SWE-bench", "dev"),
    ("princeton-nlp/SWE-bench", "train"),
    ("princeton-nlp/SWE-bench_Verified", "test"),
]
KEEP = ["instance_id", "repo", "base_commit", "version", "environment_setup_commit",
        "problem_statement", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "created_at"]


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", action="append", help="name:split, repeatable (default: SWE-bench full + Verified)")
    args = ap.parse_args()
    pairs = [tuple(d.rsplit(":", 1)) for d in args.dataset] if args.dataset else DEFAULT
    seen: set[str] = set()
    with open(args.out, "w", encoding="utf-8") as f:
        for name, split in pairs:
            try:
                ds = load_dataset(name, split=split)
            except Exception as e:  # noqa: BLE001
                print(f"skip {name}:{split}: {e}")
                continue
            n = 0
            for row in ds:
                if row["instance_id"] in seen:
                    continue
                seen.add(row["instance_id"])
                f.write(json.dumps({k: row.get(k) for k in KEEP}, ensure_ascii=False) + "\n")
                n += 1
            print(f"{name}:{split}: {n} rows")


if __name__ == "__main__":
    main()
