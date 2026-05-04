"""
python3 fairgap/7score_output_shift_rbo.py \
  --ranked_lists examples/toy_out/ranked_lists.jsonl \
  --split examples/toy_out/split.jsonl \
  --out_dev examples/toy_out/output_distance.dev.jsonl \
  --out_eval examples/toy_out/output_distance.eval.jsonl \
  --out_summary examples/toy_out/output_distance_summary.json \
  --k 10 \
  --p 0.9

"""


import argparse
import json
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple


RANKED_LISTS_DEFAULT = "examples/toy_out/ranked_lists.jsonl"
SPLIT_DEFAULT = "examples/toy_out/split.jsonl"

OUT_DEV_DEFAULT = "examples/toy_out/output_distance.dev.jsonl"
OUT_EVAL_DEFAULT = "examples/toy_out/output_distance.eval.jsonl"
OUT_SUMMARY_DEFAULT = "examples/toy_out/output_distance_summary.json"

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


def load_split_map(path: str) -> Dict[int, str]:
    split_map: Dict[int, str] = {}
    for row in iter_jsonl(path):
        if "user_id" not in row or "split" not in row:
            continue
        try:
            uid = int(row["user_id"])
        except Exception:
            continue
        split = str(row["split"]).strip().lower()
        if split not in ("dev", "eval"):
            continue
        split_map[uid] = split
    return split_map


def load_ranked_lists(path: str) -> Dict[int, Dict[str, Dict[str, Any]]]:
    rows_by_user: Dict[int, Dict[str, Dict[str, Any]]] = {}
    seen: Dict[Tuple[int, str], int] = defaultdict(int)

    for row in iter_jsonl(path):
        if "user_id" not in row or "variant" not in row:
            continue
        try:
            uid = int(row["user_id"])
        except Exception:
            continue

        variant = str(row["variant"]).strip().lower()
        if variant not in ("age_a", "age_b"):
            continue

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
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if k <= 0:
        raise ValueError("k must be positive")

    a = list_a[:k]
    b = list_b[:k]

    prefix_a: set = set()
    prefix_b: set = set()

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_lists", default=RANKED_LISTS_DEFAULT)
    ap.add_argument("--split", default=SPLIT_DEFAULT)
    ap.add_argument("--out_dev", default=OUT_DEV_DEFAULT)
    ap.add_argument("--out_eval", default=OUT_EVAL_DEFAULT)
    ap.add_argument("--out_summary", default=OUT_SUMMARY_DEFAULT)
    ap.add_argument("--k", type=int, default=10, help="Top-K cutoff for RBO")
    ap.add_argument("--p", type=float, default=0.9, help="RBO persistence parameter")
    args = ap.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")
    if not (0.0 < args.p < 1.0):
        raise ValueError("--p must be in (0, 1)")

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
            if "age_a" not in variants or "age_b" not in variants:
                n_missing_pair += 1
                continue

            row_a = variants["age_a"]
            row_b = variants["age_b"]

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
                "titles_age_a": titles_a[: args.k],
                "titles_age_b": titles_b[: args.k],
                "n_titles_age_a": len(titles_a[: args.k]),
                "n_titles_age_b": len(titles_b[: args.k]),
                "parse_ok_age_a": bool(row_a.get("parse_ok", False)),
                "parse_ok_age_b": bool(row_b.get("parse_ok", False)),
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

