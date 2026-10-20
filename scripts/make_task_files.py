"""Write the first N ranked tasks into size-limited JSONL parts for the GitHub workflow.

GitHub rejects files over 100 MB (and warns above 50 MB), so the tasks are split into
data/<name>/part-000.jsonl, part-001.jsonl, ... in rank order.

    python scripts/make_task_files.py out/select/ranked.jsonl --head 200  --out data/pilot
    python scripts/make_task_files.py out/select/ranked.jsonl --head 4500 --out data/full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codeagent_data.jsonl import iter_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ranked", help="out/select/ranked.jsonl")
    ap.add_argument("--head", type=int, required=True, help="take the first N tasks in rank order")
    ap.add_argument("--out", required=True, help="folder to write part-*.jsonl into (emptied first)")
    ap.add_argument("--max-mb", type=float, default=40.0, help="max size per part file")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part-*.jsonl"):
        old.unlink()
    limit = int(args.max_mb * 1024 * 1024)
    part, size, n, f = 0, 0, 0, None
    mix: Counter[str] = Counter()
    for t in iter_jsonl(args.ranked):
        if n >= args.head:
            break
        line = (json.dumps(t, ensure_ascii=False) + "\n").encode("utf-8")
        if f is None or size + len(line) > limit:
            if f:
                f.close()
                part += 1
            f, size = (out / f"part-{part:03d}.jsonl").open("wb"), 0
        f.write(line)
        size += len(line)
        n += 1
        mix[t.get("verifier") or t.get("source") or "?"] += 1
    if f:
        f.close()
    for p in sorted(out.glob("part-*.jsonl")):
        print(f"{p}  {p.stat().st_size / 1e6:.1f} MB")
    print(f"{n} tasks:", dict(mix.most_common()))


if __name__ == "__main__":
    main()
