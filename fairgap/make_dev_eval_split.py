#!/usr/bin/env python3
"""
fairgap/5make_dev_eval_split.py

Create a user-level development/evaluation split for FairGap.

Modes
-----
1) Full-pairs mode:
   Split all users found in pairs.jsonl.

2) Generated-complete mode:
   Split only users that have BOTH age_a and age_b variants present
   in generations.jsonl. This is useful when generation is only partially
   completed and you want split.jsonl to reflect the currently usable subset.
"""

import argparse
import json
import os
import random
from typing import Dict, Iterator, List, Set

PAIRS_DEFAULT = "examples/toy_out/pairs.jsonl"
GENERATIONS_DEFAULT = "examples/toy_out/generations.jsonl"
OUT_SPLIT_DEFAULT = "examples/toy_out/split.jsonl"
OUT_SUMMARY_DEFAULT = "examples/toy_out/split_summary.json"

VARIANT_A = "age_a"
VARIANT_B = "age_b"


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict]:
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


def normalize_user_id(x) -> str:
    return str(x).strip()


def collect_user_ids_from_pairs(pairs_path: str) -> List[str]:
    user_ids: Set[str] = set()
    for row in iter_jsonl(pairs_path):
        if "user_id" not in row:
            continue
        uid = normalize_user_id(row["user_id"])
        if uid:
            user_ids.add(uid)
    return sorted(user_ids, key=lambda x: int(x) if x.isdigit() else x)


def collect_variant_counts_from_generations(generations_path: str) -> Dict[str, Set[str]]:
    """
    Returns:
      dict[user_id] = set of available variants among {"age_a", "age_b"}.
    """
    seen: Dict[str, Set[str]] = {}

    for row in iter_jsonl(generations_path):
        uid = normalize_user_id(row.get("user_id"))
        variant = str(row.get("variant", "")).strip()

        if not uid or variant not in (VARIANT_A, VARIANT_B):
            continue

        prompt = row.get("prompt", None)
        output_text = row.get("output_text", None)

        if not isinstance(prompt, str) or not prompt.strip():
            continue

        if not isinstance(output_text, str) or not output_text.strip():
            continue

        if uid not in seen:
            seen[uid] = set()

        seen[uid].add(variant)

    return seen


def collect_complete_user_ids_from_generations(generations_path: str) -> List[str]:
    seen = collect_variant_counts_from_generations(generations_path)
    complete_ids = [
        uid for uid, variants in seen.items()
        if {VARIANT_A, VARIANT_B}.issubset(variants)
    ]
    return sorted(complete_ids, key=lambda x: int(x) if x.isdigit() else x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=PAIRS_DEFAULT, help="Input pairs.jsonl")
    ap.add_argument("--generations", default=GENERATIONS_DEFAULT, help="Input generations.jsonl")
    ap.add_argument("--out", default=OUT_SPLIT_DEFAULT, help="Output split.jsonl")
    ap.add_argument("--summary_out", default=OUT_SUMMARY_DEFAULT, help="Output split_summary.json")
    ap.add_argument("--dev_ratio", type=float, default=0.2, help="Development split ratio, default=0.2")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible split")
    ap.add_argument(
        "--only_generated_complete",
        action="store_true",
        help="Split only users with both age_a and age_b variants present in generations.jsonl",
    )
    args = ap.parse_args()

    if not (0.0 < args.dev_ratio < 1.0):
        raise ValueError("--dev_ratio must be between 0 and 1")

    ensure_dir(os.path.dirname(args.out))
    ensure_dir(os.path.dirname(args.summary_out))

    if args.only_generated_complete:
        if not os.path.exists(args.generations):
            raise FileNotFoundError(f"generations file not found: {args.generations}")
        user_ids = collect_complete_user_ids_from_generations(args.generations)
        user_source = "generations_complete"
    else:
        if not os.path.exists(args.pairs):
            raise FileNotFoundError(f"pairs file not found: {args.pairs}")
        user_ids = collect_user_ids_from_pairs(args.pairs)
        user_source = "pairs_all"

    n_total = len(user_ids)

    if n_total == 0:
        raise RuntimeError("No valid users found for splitting.")

    rng = random.Random(args.seed)
    shuffled = list(user_ids)
    rng.shuffle(shuffled)

    n_dev = int(round(n_total * args.dev_ratio))
    n_dev = max(1, min(n_total - 1, n_dev))

    dev_ids = set(shuffled[:n_dev])
    eval_ids = set(shuffled[n_dev:])

    if len(dev_ids & eval_ids) != 0:
        raise RuntimeError("Split error: dev and eval overlap")

    if len(dev_ids) + len(eval_ids) != n_total:
        raise RuntimeError("Split error: dev + eval does not cover all users")

    with open(args.out, "w", encoding="utf-8") as f_out:
        for uid in sorted(dev_ids, key=lambda x: int(x) if x.isdigit() else x):
            rec = {"user_id": int(uid) if uid.isdigit() else uid, "split": "dev"}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for uid in sorted(eval_ids, key=lambda x: int(x) if x.isdigit() else x):
            rec = {"user_id": int(uid) if uid.isdigit() else uid, "split": "eval"}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "user_source": user_source,
        "pairs_file": args.pairs,
        "generations_file": args.generations if args.only_generated_complete else None,
        "split_file": args.out,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "n_total_users": n_total,
        "n_dev_users": len(dev_ids),
        "n_eval_users": len(eval_ids),
        "actual_dev_ratio": round(len(dev_ids) / n_total, 6),
        "actual_eval_ratio": round(len(eval_ids) / n_total, 6),
        "variant_a": VARIANT_A,
        "variant_b": VARIANT_B,
    }

    if args.only_generated_complete:
        seen = collect_variant_counts_from_generations(args.generations)
        n_age_a = sum(1 for _, variants in seen.items() if VARIANT_A in variants)
        n_age_b = sum(1 for _, variants in seen.items() if VARIANT_B in variants)
        n_complete = sum(1 for _, variants in seen.items() if {VARIANT_A, VARIANT_B}.issubset(variants))

        summary["generation_availability"] = {
            "n_users_with_age_a": n_age_a,
            "n_users_with_age_b": n_age_b,
            "n_users_with_both": n_complete,
        }

    with open(args.summary_out, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote split file: {args.out}")
    print(
        f"[OK] Source={user_source} users_total={n_total} "
        f"dev={len(dev_ids)} eval={len(eval_ids)} "
        f"(dev_ratio={summary['actual_dev_ratio']:.6f})"
    )
    print(f"[OK] Wrote summary: {args.summary_out}")


if __name__ == "__main__":
    main()