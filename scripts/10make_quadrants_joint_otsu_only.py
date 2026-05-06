#!/usr/bin/env python3
"""
make_quadrants_joint_otsu_only.py

Merge evaluation-set output/internal distances and assign each user to one of
four quadrants using 2D Joint Otsu thresholding.

Inputs
------
- output_distance.eval.jsonl
- internal_distance.eval.jsonl

Outputs
-------
- quadrants_joint_otsu.csv
- quadrants_joint_otsu.jsonl
- quadrants_joint_otsu_summary.json
- quadrants_joint_otsu_summary.txt

Quadrant definition
-------------------
Let X = d_out, where higher means larger observable output shift.
Let Y = d_in, where higher means larger internal representation shift.

Q1: X >= tx and Y <  ty   high output, low internal
Q2: X >= tx and Y >= ty   high output, high internal
Q3: X <  tx and Y <  ty   low output, low internal
Q4: X <  tx and Y >= ty   low output, high internal

Method
------
- Build a 2D histogram over (d_out, d_in).
- Search all threshold pairs (tx, ty) on the histogram grid.
- Choose the pair maximizing the 2D Otsu between-class variance.
- Map the selected histogram bin boundary back to real-value thresholds.

Example
-------
python scripts/10make_quadrants_joint_otsu_only.py \
  --out_dist data/movielens_smoke/gender/output_distance_sample.eval.jsonl \
  --in_dist data/movielens_smoke/gender/internal_distance_sample.eval.jsonl \
  --out_csv data/movielens_smoke/gender/quadrants_joint_otsu.csv \
  --out_jsonl data/movielens_smoke/gender/quadrants_joint_otsu.jsonl \
  --out_summary_json data/movielens_smoke/gender/quadrants_joint_otsu_summary.json \
  --out_summary_txt data/movielens_smoke/gender/quadrants_joint_otsu_summary.txt \
  --bins 128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl_strict(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {lineno} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object at line {lineno} in {path}, got {type(obj).__name__}"
                )
            yield obj


def _coerce_split(x: Any) -> str:
    return str(x).strip().lower()


def _safe_float(x: Any, field_name: str, row: Dict[str, Any]) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise ValueError(f"Field {field_name} is not a valid float in row: {row}") from e
    if not math.isfinite(v):
        raise ValueError(f"Field {field_name} is not finite in row: {row}")
    return v


def _safe_user_id(x: Any) -> str:
    return str(x)


def load_output_distances(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      user_id -> {"user_id": str, "split": "eval", "d_out": float}
    """
    out_map: Dict[str, Dict[str, Any]] = {}

    for row in iter_jsonl_strict(path):
        if "user_id" not in row:
            raise ValueError(f"Missing user_id in output-distance row: {row}")

        uid = _safe_user_id(row["user_id"])

        if "split" in row:
            split = _coerce_split(row["split"])
            if split != "eval":
                continue
        else:
            split = "eval"

        if "d_out" in row:
            d_out = _safe_float(row["d_out"], "d_out", row)
        elif "output_distance" in row:
            d_out = _safe_float(row["output_distance"], "output_distance", row)
        elif "distance" in row:
            d_out = _safe_float(row["distance"], "distance", row)
        else:
            raise ValueError(
                "Could not find output-distance field in row. "
                "Expected one of: d_out, output_distance, distance. "
                f"Row: {row}"
            )

        if uid in out_map:
            raise ValueError(f"Duplicate user_id={uid} in output-distance file: {path}")

        out_map[uid] = {
            "user_id": uid,
            "split": split,
            "d_out": d_out,
        }

    if not out_map:
        raise RuntimeError(f"No eval output-distance rows loaded from: {path}")

    return out_map


def load_internal_distances(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      user_id -> {"user_id": str, "split": "eval", "d_in": float, ...}
    """
    in_map: Dict[str, Dict[str, Any]] = {}

    for row in iter_jsonl_strict(path):
        if "user_id" not in row:
            raise ValueError(f"Missing user_id in internal-distance row: {row}")

        uid = _safe_user_id(row["user_id"])

        if "split" in row:
            split = _coerce_split(row["split"])
            if split != "eval":
                continue
        else:
            split = "eval"

        if "d_in" not in row:
            raise ValueError(f"Missing d_in in internal-distance row: {row}")

        d_in = _safe_float(row["d_in"], "d_in", row)

        if uid in in_map:
            raise ValueError(f"Duplicate user_id={uid} in internal-distance file: {path}")

        in_map[uid] = {
            "user_id": uid,
            "split": split,
            "d_in": d_in,
            "delta_q1": row.get("delta_q1"),
            "delta_q2": row.get("delta_q2"),
            "delta_q3": row.get("delta_q3"),
            "delta_q4": row.get("delta_q4"),
            "weights": row.get("weights"),
            "sep_excess_over_chance": row.get("sep_excess_over_chance"),
            "relative_layers": row.get("relative_layers"),
            "layer_indices": row.get("layer_indices"),
        }

    if not in_map:
        raise RuntimeError(f"No eval internal-distance rows loaded from: {path}")

    return in_map


def merge_eval_distances(
    out_map: Dict[str, Dict[str, Any]],
    in_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out_ids = set(out_map.keys())
    in_ids = set(in_map.keys())
    common_ids = sorted(out_ids & in_ids)

    if not common_ids:
        raise RuntimeError("No overlapping user_id values between output and internal eval files")

    merge_info = {
        "n_output_users": int(len(out_ids)),
        "n_internal_users": int(len(in_ids)),
        "n_common_users": int(len(common_ids)),
        "n_only_output": int(len(out_ids - in_ids)),
        "n_only_internal": int(len(in_ids - out_ids)),
    }

    rows: List[Dict[str, Any]] = []
    for uid in common_ids:
        r_out = out_map[uid]
        r_in = in_map[uid]

        rec = {
            "user_id": uid,
            "split": "eval",
            "d_out": float(r_out["d_out"]),
            "d_in": float(r_in["d_in"]),
            "delta_q1": r_in.get("delta_q1"),
            "delta_q2": r_in.get("delta_q2"),
            "delta_q3": r_in.get("delta_q3"),
            "delta_q4": r_in.get("delta_q4"),
            "weights": r_in.get("weights"),
            "sep_excess_over_chance": r_in.get("sep_excess_over_chance"),
            "relative_layers": r_in.get("relative_layers"),
            "layer_indices": r_in.get("layer_indices"),
        }
        rows.append(rec)

    return rows, merge_info


def _prefix_sum_2d(a: np.ndarray) -> np.ndarray:
    return np.cumsum(np.cumsum(a, axis=0), axis=1)


def _rect_sum(ps: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    if x0 > x1 or y0 > y1:
        return 0.0

    total = ps[x1, y1]
    if x0 > 0:
        total -= ps[x0 - 1, y1]
    if y0 > 0:
        total -= ps[x1, y0 - 1]
    if x0 > 0 and y0 > 0:
        total += ps[x0 - 1, y0 - 1]
    return float(total)


def fit_joint_otsu_thresholds(
    d_out: np.ndarray,
    d_in: np.ndarray,
    bins: int = 128,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    2D Joint Otsu over the (d_out, d_in) plane.
    """
    if d_out.ndim != 1 or d_in.ndim != 1 or len(d_out) != len(d_in):
        raise ValueError("d_out and d_in must be 1D arrays of equal length")
    if len(d_out) < 4:
        raise ValueError("Need at least 4 points for joint Otsu")

    x_min, x_max = float(np.min(d_out)), float(np.max(d_out))
    y_min, y_max = float(np.min(d_in)), float(np.max(d_in))

    if not all(map(math.isfinite, [x_min, x_max, y_min, y_max])):
        raise ValueError("Non-finite values encountered in d_out/d_in")

    if abs(x_max - x_min) < eps:
        raise RuntimeError("d_out is constant; joint Otsu thresholding is not meaningful")
    if abs(y_max - y_min) < eps:
        raise RuntimeError("d_in is constant; joint Otsu thresholding is not meaningful")

    H, x_edges, y_edges = np.histogram2d(
        d_out,
        d_in,
        bins=[bins, bins],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    H = np.asarray(H, dtype=np.float64)

    total_mass = float(H.sum())
    if total_mass <= 0:
        raise RuntimeError("2D histogram is empty")

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xc = x_centers[:, None] * np.ones((bins, bins), dtype=np.float64)
    Yc = np.ones((bins, bins), dtype=np.float64) * y_centers[None, :]

    P = H / total_mass
    PX = P * Xc
    PY = P * Yc

    P_ps = _prefix_sum_2d(P)
    PX_ps = _prefix_sum_2d(PX)
    PY_ps = _prefix_sum_2d(PY)

    mu_T_x = float(PX.sum())
    mu_T_y = float(PY.sum())

    best_score = -1.0
    best_tx_bin = None
    best_ty_bin = None

    for i in range(bins - 1):
        for j in range(bins - 1):
            w_q3 = _rect_sum(P_ps, 0, i, 0, j)
            w_q1 = _rect_sum(P_ps, i + 1, bins - 1, 0, j)
            w_q4 = _rect_sum(P_ps, 0, i, j + 1, bins - 1)
            w_q2 = _rect_sum(P_ps, i + 1, bins - 1, j + 1, bins - 1)

            weights = [w_q1, w_q2, w_q3, w_q4]
            if any(w <= eps for w in weights):
                continue

            def class_mean(x0: int, x1: int, y0: int, y1: int, w: float) -> Tuple[float, float]:
                mx = _rect_sum(PX_ps, x0, x1, y0, y1) / w
                my = _rect_sum(PY_ps, x0, x1, y0, y1) / w
                return float(mx), float(my)

            mu_q3 = class_mean(0, i, 0, j, w_q3)
            mu_q1 = class_mean(i + 1, bins - 1, 0, j, w_q1)
            mu_q4 = class_mean(0, i, j + 1, bins - 1, w_q4)
            mu_q2 = class_mean(i + 1, bins - 1, j + 1, bins - 1, w_q2)

            score = (
                w_q1 * ((mu_q1[0] - mu_T_x) ** 2 + (mu_q1[1] - mu_T_y) ** 2)
                + w_q2 * ((mu_q2[0] - mu_T_x) ** 2 + (mu_q2[1] - mu_T_y) ** 2)
                + w_q3 * ((mu_q3[0] - mu_T_x) ** 2 + (mu_q3[1] - mu_T_y) ** 2)
                + w_q4 * ((mu_q4[0] - mu_T_x) ** 2 + (mu_q4[1] - mu_T_y) ** 2)
            )

            if score > best_score:
                best_score = float(score)
                best_tx_bin = int(i)
                best_ty_bin = int(j)

    if best_tx_bin is None or best_ty_bin is None:
        raise RuntimeError(
            "Joint Otsu failed to find a valid threshold pair. "
            "Try reducing --bins or inspect whether d_out/d_in are too concentrated."
        )

    tx = float(x_edges[best_tx_bin + 1])
    ty = float(y_edges[best_ty_bin + 1])

    return {
        "tx": tx,
        "ty": ty,
        "score": float(best_score),
        "bins": int(bins),
        "tx_bin": int(best_tx_bin),
        "ty_bin": int(best_ty_bin),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }


def assign_quadrant(d_out: float, d_in: float, tx: float, ty: float) -> str:
    if d_out >= tx and d_in < ty:
        return "Q1"
    if d_out >= tx and d_in >= ty:
        return "Q2"
    if d_out < tx and d_in < ty:
        return "Q3"
    return "Q4"


def attach_quadrants(rows: List[Dict[str, Any]], tx: float, ty: float) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        d_out = float(row["d_out"])
        d_in = float(row["d_in"])
        q = assign_quadrant(d_out=d_out, d_in=d_in, tx=tx, ty=ty)

        rec = dict(row)
        rec["tx"] = float(tx)
        rec["ty"] = float(ty)
        rec["quadrant"] = q
        out_rows.append(rec)

    return out_rows


def summarize_quadrants(
    rows: List[Dict[str, Any]],
    tx: float,
    ty: float,
    otsu_meta: Dict[str, Any],
    merge_info: Dict[str, int],
) -> Dict[str, Any]:
    quadrants = ["Q1", "Q2", "Q3", "Q4"]
    n = len(rows)

    counts = {q: 0 for q in quadrants}
    d_out_by_q = {q: [] for q in quadrants}
    d_in_by_q = {q: [] for q in quadrants}

    for row in rows:
        q = row["quadrant"]
        counts[q] += 1
        d_out_by_q[q].append(float(row["d_out"]))
        d_in_by_q[q].append(float(row["d_in"]))

    def stats(xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {
                "n": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
            }
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }

    quadrant_summary = {}
    for q in quadrants:
        quadrant_summary[q] = {
            "count": int(counts[q]),
            "proportion": float(counts[q] / n) if n > 0 else None,
            "d_out": stats(d_out_by_q[q]),
            "d_in": stats(d_in_by_q[q]),
        }

    all_d_out = np.asarray([float(r["d_out"]) for r in rows], dtype=np.float64)
    all_d_in = np.asarray([float(r["d_in"]) for r in rows], dtype=np.float64)

    return {
        "n_users_scored": int(n),
        "merge_info": merge_info,
        "thresholds": {
            "tx_d_out": float(tx),
            "ty_d_in": float(ty),
            "method": "2d_joint_otsu",
            "bins": int(otsu_meta["bins"]),
            "objective_score": float(otsu_meta["score"]),
            "tx_bin": int(otsu_meta["tx_bin"]),
            "ty_bin": int(otsu_meta["ty_bin"]),
            "x_min": float(otsu_meta["x_min"]),
            "x_max": float(otsu_meta["x_max"]),
            "y_min": float(otsu_meta["y_min"]),
            "y_max": float(otsu_meta["y_max"]),
        },
        "global_stats": {
            "d_out": stats(all_d_out.tolist()),
            "d_in": stats(all_d_in.tolist()),
        },
        "quadrants": quadrant_summary,
        "quadrant_definition": {
            "Q1": "high output, low internal",
            "Q2": "high output, high internal",
            "Q3": "low output, low internal",
            "Q4": "low output, high internal",
        },
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)

    fieldnames = [
        "user_id",
        "split",
        "d_out",
        "d_in",
        "tx",
        "ty",
        "quadrant",
        "delta_q1",
        "delta_q2",
        "delta_q3",
        "delta_q4",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "user_id": row.get("user_id"),
                "split": row.get("split"),
                "d_out": row.get("d_out"),
                "d_in": row.get("d_in"),
                "tx": row.get("tx"),
                "ty": row.get("ty"),
                "quadrant": row.get("quadrant"),
                "delta_q1": row.get("delta_q1"),
                "delta_q2": row.get("delta_q2"),
                "delta_q3": row.get("delta_q3"),
                "delta_q4": row.get("delta_q4"),
            })


def write_summary_json(path: str, summary: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_summary_txt(path: str, summary: Dict[str, Any]) -> None:
    ensure_parent_dir(path)

    q = summary["quadrants"]
    g = summary["global_stats"]
    t = summary["thresholds"]
    m = summary["merge_info"]

    def _fmt(x: Any, digits: int = 6) -> str:
        if x is None:
            return "NA"
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)

    lines = []
    lines.append("Quadrants Summary (2D Joint Otsu)")
    lines.append("=" * 60)
    lines.append(f"Users scored: {summary['n_users_scored']}")
    lines.append(
        f"Merge coverage: output={m['n_output_users']} | internal={m['n_internal_users']} | "
        f"intersection={m['n_common_users']} | only_output={m['n_only_output']} | only_internal={m['n_only_internal']}"
    )
    lines.append(f"Threshold tx (d_out): {_fmt(t['tx_d_out'])}")
    lines.append(f"Threshold ty (d_in):  {_fmt(t['ty_d_in'])}")
    lines.append(f"Otsu bins:            {t['bins']}")
    lines.append(f"Otsu score:           {_fmt(t['objective_score'], digits=12)}")
    lines.append("")

    lines.append("Global statistics")
    lines.append("-" * 60)
    lines.append(
        f"d_out  mean±std: {_fmt(g['d_out']['mean'])} ± {_fmt(g['d_out']['std'])} | "
        f"median={_fmt(g['d_out']['median'])}"
    )
    lines.append(
        f"d_in   mean±std: {_fmt(g['d_in']['mean'])} ± {_fmt(g['d_in']['std'])} | "
        f"median={_fmt(g['d_in']['median'])}"
    )
    lines.append("")

    lines.append("Quadrant counts")
    lines.append("-" * 60)
    for quad in ["Q1", "Q2", "Q3", "Q4"]:
        lines.append(
            f"{quad}: n={q[quad]['count']:5d} | "
            f"prop={_fmt(q[quad]['proportion'])} | "
            f"d_out mean={_fmt(q[quad]['d_out']['mean'])} | "
            f"d_in mean={_fmt(q[quad]['d_in']['mean'])}"
        )

    lines.append("")
    lines.append("Definitions")
    lines.append("-" * 60)
    lines.append("Q1 = high output, low internal")
    lines.append("Q2 = high output, high internal")
    lines.append("Q3 = low output, low internal")
    lines.append("Q4 = low output, high internal")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dist", required=True, help="Input output_distance.eval.jsonl")
    ap.add_argument("--in_dist", required=True, help="Input internal_distance.eval.jsonl")
    ap.add_argument("--out_csv", required=True, help="Output quadrants csv")
    ap.add_argument("--out_jsonl", required=True, help="Output quadrants jsonl")
    ap.add_argument("--out_summary_json", required=True, help="Output summary json")
    ap.add_argument("--out_summary_txt", required=True, help="Output summary txt")
    ap.add_argument(
        "--bins",
        type=int,
        default=128,
        help="Number of bins per axis for 2D Joint Otsu",
    )
    args = ap.parse_args()

    if args.bins < 8:
        raise ValueError("--bins must be at least 8")

    for path, name in [
        (args.out_dist, "out_dist"),
        (args.in_dist, "in_dist"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input {name} file not found: {path}")

    out_map = load_output_distances(args.out_dist)
    in_map = load_internal_distances(args.in_dist)
    rows, merge_info = merge_eval_distances(out_map, in_map)

    d_out = np.asarray([float(r["d_out"]) for r in rows], dtype=np.float64)
    d_in = np.asarray([float(r["d_in"]) for r in rows], dtype=np.float64)

    otsu_meta = fit_joint_otsu_thresholds(d_out=d_out, d_in=d_in, bins=args.bins)
    tx = float(otsu_meta["tx"])
    ty = float(otsu_meta["ty"])

    rows = attach_quadrants(rows, tx=tx, ty=ty)
    summary = summarize_quadrants(rows, tx=tx, ty=ty, otsu_meta=otsu_meta, merge_info=merge_info)

    write_csv(args.out_csv, rows)
    write_jsonl(args.out_jsonl, rows)
    write_summary_json(args.out_summary_json, summary)
    write_summary_txt(args.out_summary_txt, summary)

    q = summary["quadrants"]
    print(
        f"[OK] Joint Otsu thresholds: tx(d_out)={tx:.6f}, ty(d_in)={ty:.6f} | bins={args.bins}"
    )
    print(
        f"[OK] Merge coverage: output={merge_info['n_output_users']}, "
        f"internal={merge_info['n_internal_users']}, common={merge_info['n_common_users']}"
    )
    print(
        f"[OK] Quadrant counts: "
        f"Q1={q['Q1']['count']}, Q2={q['Q2']['count']}, "
        f"Q3={q['Q3']['count']}, Q4={q['Q4']['count']}"
    )
    print(f"[OK] Wrote CSV: {args.out_csv}")
    print(f"[OK] Wrote JSONL: {args.out_jsonl}")
    print(f"[OK] Wrote summary JSON: {args.out_summary_json}")
    print(f"[OK] Wrote summary TXT: {args.out_summary_txt}")


if __name__ == "__main__":
    main()
