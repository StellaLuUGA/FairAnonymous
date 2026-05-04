
import argparse
import json
import math
import os
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


MATCH_EVAL_DEFAULT = "examples/toy_out/match10.eval.jsonl"
OUT_EVAL_DEFAULT = "examples/toy_out/output_distance.eval.jsonl"
IN_EVAL_DEFAULT = "examples/toy_out/internal_distance.eval.jsonl"

OUT_USERLEVEL_DEFAULT = "examples/toy_out/main_metrics_eval_userlevel.jsonl"
OUT_SUMMARY_DEFAULT = "examples/toy_out/main_metrics_summary.json"


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


def is_finite_number(x: Any) -> bool:
    try:
        val = float(x)
    except Exception:
        return False
    return math.isfinite(val)


def get_match_field_names(match_variant: str) -> Dict[str, str]:
    variant = str(match_variant).strip().lower()
    if variant == "strict":
        return {
            "age_a": "match10_age_a_strict",
            "age_b": "match10_age_b_strict",
            "mean": "match10_mean_strict",
            "delta": "delta_match10_strict",
        }
    if variant == "resolved":
        return {
            "age_a": "match10_age_a_resolved",
            "age_b": "match10_age_b_resolved",
            "mean": "match10_mean_resolved",
            "delta": "delta_match10_resolved",
        }
    raise ValueError("--match_variant must be one of: strict, resolved")


def load_match_eval(path: str, match_variant: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    field_names = get_match_field_names(match_variant)

    for row in iter_jsonl(path):
        if "user_id" not in row:
            continue
        try:
            uid = int(row["user_id"])
        except Exception:
            continue

        split = str(row.get("split", "eval")).strip().lower()
        if split != "eval":
            continue

        a_key = field_names["age_a"]
        b_key = field_names["age_b"]
        mean_key = field_names["mean"]
        delta_key = field_names["delta"]

        if a_key not in row or b_key not in row or mean_key not in row:
            continue

        if not (
            is_finite_number(row[a_key])
            and is_finite_number(row[b_key])
            and is_finite_number(row[mean_key])
        ):
            continue

        match_a = float(row[a_key])
        match_b = float(row[b_key])
        match_mean = float(row[mean_key])

        match_delta = None
        if delta_key in row and is_finite_number(row[delta_key]):
            match_delta = float(row[delta_key])

        out[uid] = {
            "user_id": uid,
            "match10_age_a": match_a,
            "match10_age_b": match_b,
            "match10_mean": match_mean,
            "delta_match10": match_delta,
            "match_variant": match_variant,
        }
    return out


def load_output_eval(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if "user_id" not in row:
            continue
        try:
            uid = int(row["user_id"])
        except Exception:
            continue

        split = str(row.get("split", "eval")).strip().lower()
        if split != "eval":
            continue

        if "d_out" not in row or not is_finite_number(row["d_out"]):
            continue

        d_out = float(row["d_out"])
        out[uid] = {
            "user_id": uid,
            "d_out": d_out,
        }
    return out


def load_internal_eval(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if "user_id" not in row:
            continue
        try:
            uid = int(row["user_id"])
        except Exception:
            continue

        split = str(row.get("split", "eval")).strip().lower()
        if split != "eval":
            continue

        if "d_in" not in row or not is_finite_number(row["d_in"]):
            continue

        d_in = float(row["d_in"])
        out[uid] = {
            "user_id": uid,
            "d_in": d_in,
        }
    return out


def rankdata_avg_ties(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1

    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size != y.size or x.size < 2:
        return float("nan")

    rx = rankdata_avg_ties(x)
    ry = rankdata_avg_ties(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return float("nan")

    return float(np.dot(rx, ry) / denom)


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def fmt_mean_std(mean: Optional[float], std: Optional[float], digits: int = 3) -> Optional[str]:
    if mean is None or std is None:
        return None
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def fmt_mean_only(mean: Optional[float], digits: int = 3) -> Optional[str]:
    if mean is None:
        return None
    return f"{mean:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match_eval", default=MATCH_EVAL_DEFAULT, help="Input match10.eval.jsonl")
    ap.add_argument("--out_eval", default=OUT_EVAL_DEFAULT, help="Input output_distance.eval.jsonl")
    ap.add_argument("--in_eval", default=IN_EVAL_DEFAULT, help="Input internal_distance.eval.jsonl")
    ap.add_argument("--out_userlevel", default=OUT_USERLEVEL_DEFAULT, help="Output merged eval user-level JSONL")
    ap.add_argument("--out_summary", default=OUT_SUMMARY_DEFAULT, help="Output summary JSON")

    ap.add_argument(
        "--match_variant",
        default="strict",
        choices=["strict", "resolved"],
        help="Which Match@10 field version to use from match_eval",
    )
    ap.add_argument(
        "--match_with_std",
        action="store_true",
        help="If set, table string for Match@10 is formatted as mean ± std",
    )
    ap.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimal places for table strings",
    )

    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_userlevel))
    ensure_dir(os.path.dirname(args.out_summary))

    match_map = load_match_eval(args.match_eval, args.match_variant)
    out_map = load_output_eval(args.out_eval)
    in_map = load_internal_eval(args.in_eval)

    match_ids = set(match_map.keys())
    out_ids = set(out_map.keys())
    in_ids = set(in_map.keys())

    common_ids = sorted(match_ids & out_ids & in_ids)

    only_match_ids = sorted(match_ids - out_ids - in_ids)
    only_out_ids = sorted(out_ids - match_ids - in_ids)
    only_in_ids = sorted(in_ids - match_ids - out_ids)

    missing_in_output_ids = sorted((match_ids & in_ids) - out_ids)
    missing_in_internal_ids = sorted((match_ids & out_ids) - in_ids)
    missing_in_match_ids = sorted((out_ids & in_ids) - match_ids)

    match10_mean_vals: List[float] = []
    match10_a_vals: List[float] = []
    match10_b_vals: List[float] = []
    delta_match10_vals: List[float] = []
    d_out_vals: List[float] = []
    d_in_vals: List[float] = []

    with open(args.out_userlevel, "w", encoding="utf-8") as f_out:
        for uid in common_ids:
            rec = {
                "user_id": uid,
                "match10_age_a": float(match_map[uid]["match10_age_a"]),
                "match10_age_b": float(match_map[uid]["match10_age_b"]),
                "match10_mean": float(match_map[uid]["match10_mean"]),
                "match_variant": args.match_variant,
                "d_out": float(out_map[uid]["d_out"]),
                "d_in": float(in_map[uid]["d_in"]),
            }

            if match_map[uid]["delta_match10"] is not None:
                rec["delta_match10"] = float(match_map[uid]["delta_match10"])
                delta_match10_vals.append(rec["delta_match10"])

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            match10_a_vals.append(rec["match10_age_a"])
            match10_b_vals.append(rec["match10_age_b"])
            match10_mean_vals.append(rec["match10_mean"])
            d_out_vals.append(rec["d_out"])
            d_in_vals.append(rec["d_in"])

    roa = spearmanr(
        np.asarray(d_in_vals, dtype=np.float64),
        np.asarray(d_out_vals, dtype=np.float64),
    )

    match10_mean_stats = summarize(match10_mean_vals)
    match10_a_stats = summarize(match10_a_vals)
    match10_b_stats = summarize(match10_b_vals)
    delta_match10_stats = summarize(delta_match10_vals) if delta_match10_vals else {
        "n": 0,
        "mean": None,
        "std": None,
        "min": None,
        "median": None,
        "max": None,
    }
    d_out_stats = summarize(d_out_vals)
    d_in_stats = summarize(d_in_vals)

    if args.match_with_std:
        match_table_str = fmt_mean_std(
            match10_mean_stats["mean"],
            match10_mean_stats["std"],
            digits=args.digits,
        )
    else:
        match_table_str = fmt_mean_only(
            match10_mean_stats["mean"],
            digits=args.digits,
        )

    summary = {
        "match_variant_used": args.match_variant,
        "n_common_eval_users": int(len(common_ids)),
        "merge_coverage": {
            "n_match_eval_users": int(len(match_ids)),
            "n_output_eval_users": int(len(out_ids)),
            "n_internal_eval_users": int(len(in_ids)),
            "n_common_eval_users": int(len(common_ids)),
            "n_only_match": int(len(only_match_ids)),
            "n_only_output": int(len(only_out_ids)),
            "n_only_internal": int(len(only_in_ids)),
            "n_missing_in_output_for_three_way_merge": int(len(missing_in_output_ids)),
            "n_missing_in_internal_for_three_way_merge": int(len(missing_in_internal_ids)),
            "n_missing_in_match_for_three_way_merge": int(len(missing_in_match_ids)),
        },
        "merge_debug_ids": {
            "only_match_first_20": only_match_ids[:20],
            "only_output_first_20": only_out_ids[:20],
            "only_internal_first_20": only_in_ids[:20],
            "missing_in_output_first_20": missing_in_output_ids[:20],
            "missing_in_internal_first_20": missing_in_internal_ids[:20],
            "missing_in_match_first_20": missing_in_match_ids[:20],
        },
        "metrics": {
            "match10_mean": match10_mean_stats,
            "match10_age_a": match10_a_stats,
            "match10_age_b": match10_b_stats,
            "delta_match10": delta_match10_stats,
            "OBS": d_out_stats,
            "IBS": d_in_stats,
            "ROA_spearman": float(roa) if math.isfinite(roa) else None,
        },
        "table_strings": {
            "PrefMatch@10": match_table_str,
            "OBS": fmt_mean_std(d_out_stats["mean"], d_out_stats["std"], digits=args.digits),
            "IBS": fmt_mean_std(d_in_stats["mean"], d_in_stats["std"], digits=args.digits),
            "ROA": fmt_mean_only(float(roa) if math.isfinite(roa) else None, digits=args.digits),
        },
    }

    with open(args.out_summary, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"[OK] match_variant_used={args.match_variant}")
    print(f"[OK] common eval users={len(common_ids)}")
    print(f"[OK] wrote merged user-level file -> {args.out_userlevel}")
    print(f"[OK] wrote summary -> {args.out_summary}")
    print(
        "[OK] Table-ready strings: "
        f"PrefMatch@10={summary['table_strings']['PrefMatch@10']}, "
        f"OBS={summary['table_strings']['OBS']}, "
        f"IBS={summary['table_strings']['IBS']}, "
        f"ROA={summary['table_strings']['ROA']}"
    )


if __name__ == "__main__":
    main()

