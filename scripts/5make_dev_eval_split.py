#!/usr/bin/env python3
"""
make_dev_eval_split.py

Create a user-level development/evaluation split for FairGap.

Input
-----
- pairs.jsonl

Outputs
-------
- split.jsonl
- split_summary.json

Behavior
--------
- Splits counterfactual pairs at the user_id level.
- Default ratio is 20% dev and 80% eval.
- Split is deterministic under a fixed random seed.
- user_id is treated as a string to support anonymized identifiers.

Each row in split.jsonl:
{
  "user_id": "<user_id>",
  "split": "dev" | "eval"
}

Example
-------
python scripts/5make_dev_eval_split.py \
  --pairs data/movielens_smoke/gender/pairs_sample.jsonl \
  --out data/movielens_smoke/gender/split_sample.jsonl \
  --summary_out data/movielens_smoke/gender/split_summary.json \
  --dev_ratio 0.2 \
  --seed 1234
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterator, List, Set


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def collect_user_ids(pairs_path: str) -> List[str]:
    user_ids: Set[str] = set()

    for row in iter_jsonl(pairs_path):
        if "user_id" not in row:
            continue

        uid = str(row["user_id"]).strip()
        if uid:
            user_ids.add(uid)

    return sorted(user_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Input pairs.jsonl")
    ap.add_argument("--out", required=True, help="Output split.jsonl")
    ap.add_argument("--summary_out", required=True, help="Output split_summary.json")
    ap.add_argument("--dev_ratio", type=float, default=0.2, help="Development split ratio")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible split")
    args = ap.parse_args()

    if not (0.0 < args.dev_ratio < 1.0):
        raise ValueError("--dev_ratio must be between 0 and 1")

    ensure_dir(os.path.dirname(args.out))
    ensure_dir(os.path.dirname(args.summary_out))

    user_ids = collect_user_ids(args.pairs)
    n_total = len(user_ids)

    if n_total == 0:
        raise RuntimeError(f"No valid user_id found in: {args.pairs}")

    if n_total < 2:
        raise RuntimeError("Need at least two users to create dev/eval split")

    rng = random.Random(args.seed)
    shuffled = list(user_ids)
    rng.shuffle(shuffled)

    n_dev = int(round(n_total * args.dev_ratio))
    n_dev = max(1, min(n_total - 1, n_dev))

    dev_ids = set(shuffled[:n_dev])
    eval_ids = set(shuffled[n_dev:])

    if dev_ids & eval_ids:
        raise RuntimeError("Split error: dev and eval overlap")

    if len(dev_ids) + len(eval_ids) != n_total:
        raise RuntimeError("Split error: dev + eval does not cover all users")

    with open(args.out, "w", encoding="utf-8") as f_out:
        for uid in sorted(dev_ids):
            rec = {"user_id": uid, "split": "dev"}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for uid in sorted(eval_ids):
            rec = {"user_id": uid, "split": "eval"}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "seed": int(args.seed),
        "dev_ratio_requested": float(args.dev_ratio),
        "n_total_users": int(n_total),
        "n_dev_users": int(len(dev_ids)),
        "n_eval_users": int(len(eval_ids)),
        "actual_dev_ratio": round(len(dev_ids) / n_total, 6),
        "actual_eval_ratio": round(len(eval_ids) / n_total, 6),
        "note": "File paths are intentionally omitted from this summary for anonymized artifact release.",
    }

    with open(args.summary_out, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote split file: {args.out}")
    print(
        f"[OK] Users total={n_total} dev={len(dev_ids)} eval={len(eval_ids)} "
        f"(dev_ratio={summary['actual_dev_ratio']:.6f})"
    )
    print(f"[OK] Wrote summary: {args.summary_out}")


if __name__ == "__main__":
    main()
