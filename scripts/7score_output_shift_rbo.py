#!/usr/bin/env python3
"""
score_output_shift_rbo.py

Compute output-side shift (OBS) for counterfactual recommendation pairs.

Definition
----------
For each user u, let Y_a and Y_b be the top-K ranked recommendation lists
generated under two counterfactual prompt variants.

OBS(u) = 1 - RBO@K(Y_a, Y_b)

where RBO@K is rank-biased overlap at depth K, normalized so that two identical
lists yield RBO = 1 and OBS = 0.

Inputs
------
- ranked_lists.jsonl
- split.jsonl

Outputs
-------
- output_distance.dev.jsonl
- output_distance.eval.jsonl
- output_distance_summary.json

Example
-------
python scripts/7score_output_shift_rbo.py \
  --ranked_lists data/movielens_smoke/gender/ranked_lists_sample.jsonl \
  --split data/movielens_smoke/gender/split_sample.jsonl \
  --out_dev data/movielens_smoke/gender/output_distance_sample.dev.jsonl \
  --out_eval data/movielens_smoke/gender/output_distance_sample.eval.jsonl \
  --out_summary data/movielens_smoke/gender/output_distance_summary.json \
  --variant_a female \
  --variant_b male \
  --k 10 \
  --p 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple


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


def load_split_map(path: str) -> Dict[str, str]:
    split_map: Dict[str, str] = {}
    for row in iter_jsonl(path):
        if "user_id" not in row or "split" not in row:
            continue
        uid = str(row["user_id"])
        split = str(row["split"]).strip().lower()
        if split not in ("dev", "eval"):
            continue
        split_map[uid] = split
    return split_map


def load_ranked_lists(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns:
      rows_by_user[user_id][variant] = row

    Variant names are supplied by --variant_a and --variant_b.
    """
    rows_by_user: Dict[str, Dict[str, Dict[str, Any]]] = {}
    seen: Dict[Tuple[str, str], int] = defaultdict(int)

    for row in iter_jsonl(path):
        if "user_id" not in row or "variant" not in row:
            continue

        uid = str(row["user_id"])
        variant = str(row["variant"]).strip().lower()

        seen[(uid, variant)] += 1
        if seen[(uid, variant)] > 1:
            warnings.warn(
                f"Duplicate row for user_id={uid} variant={variant} "
                f"(occurrence {seen[(uid, variant)]}); later row overwrites earlier.",
                stacklevel=2,
            )

        rows_by_user.setdefault(uid, {})
        rows_by_user[uid][variant] = row

    return rows_by_user


def rbo_at_k(list_a: List[str], list_b: List[str], p: float = 0.9, k: int = 10) -> float:
    """
    Normalized finite-depth RBO@K.

    raw = (1-p) * sum_{d=1..K} p^(d-1) * A_d
    RBO@K = raw / (1 - p^K)

    where A_d is prefix overlap at depth d.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if k <= 0:
        raise ValueError("k must be positive")

    a = list_a[:k]
    b = list_b[:k]

    prefix_a: set[str] = set()
    prefix_b: set[str] = set()

    raw = 0.0
    for d in range(1, k + 1):
        if d <= len(a):
            prefix_a.add(a[d - 1])
        if d <= len(b):
            prefix_b.add(b[d - 1])

        overlap = len(prefix_a.intersection(prefix_b))
        a_d = overlap / d
        raw += (p ** (d - 1)) * a_d

    raw = (1.0 - p) * raw
    max_rbo = 1.0 - p ** k
    return raw / max_rbo if max_rbo > 0.0 else 0.0


def summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}

    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = var ** 0.5
    median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": vals[0],
        "median": median,
        "max": vals[-1],
    }


def normalize_variant_name(x: str) -> str:
    return str(x).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_lists", required=True, help="Input ranked_lists JSONL")
    ap.add_argument("--split", required=True, help="Input split JSONL")
    ap.add_argument("--out_dev", required=True, help="Output dev JSONL")
    ap.add_argument("--out_eval", required=True, help="Output eval JSONL")
    ap.add_argument("--out_summary", required=True, help="Output summary JSON")
    ap.add_argument("--variant_a", default="female", help="First counterfactual variant name")
    ap.add_argument("--variant_b", default="male", help="Second counterfactual variant name")
    ap.add_argument("--k", type=int, default=10, help="Top-K cutoff for RBO")
    ap.add_argument("--p", type=float, default=0.9, help="RBO persistence parameter")
    args = ap.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")
    if not (0.0 < args.p < 1.0):
        raise ValueError("--p must be in (0, 1)")

    variant_a = normalize_variant_name(args.variant_a)
    variant_b = normalize_variant_name(args.variant_b)
    if variant_a == variant_b:
        raise ValueError("--variant_a and --variant_b must be different")

    ensure_dir(os.path.dirname(args.out_dev))
    ensure_dir(os.path.dirname(args.out_eval))
    ensure_dir(os.path.dirname(args.out_summary))

    split_map = load_split_map(args.split)
    rows_by_user = load_ranked_lists(args.ranked_lists)

    n_users_seen = 0
    n_pairs_scored = 0
    n_missing_split = 0
    n_missing_pair = 0
    n_dev = 0
    n_eval = 0

    dev_d_out_vals: List[float] = []
    eval_d_out_vals: List[float] = []

    with (
        open(args.out_dev, "w", encoding="utf-8") as f_dev,
        open(args.out_eval, "w", encoding="utf-8") as f_eval,
    ):
        for uid in sorted(rows_by_user.keys()):
            n_users_seen += 1

            if uid not in split_map:
                n_missing_split += 1
                continue

            variants = rows_by_user[uid]
            if variant_a not in variants or variant_b not in variants:
                n_missing_pair += 1
                continue

            row_a = variants[variant_a]
            row_b = variants[variant_b]

            titles_a = row_a.get("ranked_titles", []) or []
            titles_b = row_b.get("ranked_titles", []) or []

            if not isinstance(titles_a, list) or not isinstance(titles_b, list):
                n_missing_pair += 1
                continue

            titles_a = [str(x).strip() for x in titles_a if str(x).strip()]
            titles_b = [str(x).strip() for x in titles_b if str(x).strip()]

            rbo = rbo_at_k(titles_a, titles_b, p=args.p, k=args.k)
            d_out = 1.0 - rbo

            split = split_map[uid]
            rec = {
                "user_id": uid,
                "split": split,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "titles_a": titles_a[: args.k],
                "titles_b": titles_b[: args.k],
                "n_titles_a": len(titles_a[: args.k]),
                "n_titles_b": len(titles_b[: args.k]),
                "parse_ok_a": bool(row_a.get("parse_ok", False)),
                "parse_ok_b": bool(row_b.get("parse_ok", False)),
                "rbo_at_k": rbo,
                "d_out": d_out,
                "k": int(args.k),
                "p": float(args.p),
                "pair_metadata": row_a.get("pair_metadata") or row_b.get("pair_metadata") or {},
            }

            if split == "dev":
                f_dev.write(json.dumps(rec, ensure_ascii=False) + "\n")
                dev_d_out_vals.append(d_out)
                n_dev += 1
            else:
                f_eval.write(json.dumps(rec, ensure_ascii=False) + "\n")
                eval_d_out_vals.append(d_out)
                n_eval += 1

            n_pairs_scored += 1

    summary = {
        "k": int(args.k),
        "p": float(args.p),
        "variant_a": variant_a,
        "variant_b": variant_b,
        "rbo_normalization": {
            "method": "divide_by_max_rbo",
            "max_rbo": round(1.0 - args.p ** args.k, 6),
            "note": "identical lists -> RBO=1.0, d_out=0.0",
        },
        "users_seen": int(n_users_seen),
        "pairs_scored": int(n_pairs_scored),
        "missing_split": int(n_missing_split),
        "missing_pair": int(n_missing_pair),
        "rows_written_dev": int(n_dev),
        "rows_written_eval": int(n_eval),
        "dev": {"d_out": summarize(dev_d_out_vals)},
        "eval": {"d_out": summarize(eval_d_out_vals)},
    }

    with open(args.out_summary, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"[OK] users_seen={n_users_seen}")
    print(f"[OK] pairs_scored={n_pairs_scored}")
    print(f"[OK] missing_split={n_missing_split}")
    print(f"[OK] missing_pair={n_missing_pair}")
    print(f"[OK] wrote dev rows={n_dev} -> {args.out_dev}")
    print(f"[OK] wrote eval rows={n_eval} -> {args.out_eval}")
    print(f"[OK] wrote summary -> {args.out_summary}")
    print(f"[OK] rbo_max={1.0 - args.p ** args.k:.4f} (normalization factor)")


if __name__ == "__main__":
    main()
