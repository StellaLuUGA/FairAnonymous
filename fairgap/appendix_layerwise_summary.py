#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


DEFAULT_SPECS = [
    {
        "attribute": "Age",
        "dataset": "MovieLens",
        "model": "LLaMA-8B",
        "path": "examples/toy_out/internal_distance_layers.eval.jsonl",
    },
]

OUT_JSON_DEFAULT = "examples/toy_out/appendix_layerwise_summary.json"
OUT_CSV_DEFAULT = "examples/toy_out/appendix_layerwise_summary.csv"
OUT_TEX_DEFAULT = "examples/toy_out/appendix_layerwise_rows.tex"


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
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


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def summarize_internal_layers(path: str) -> Dict[str, Any]:
    vals_q1: List[float] = []
    vals_q2: List[float] = []
    vals_q3: List[float] = []
    vals_q4: List[float] = []

    relative_layers = None
    layer_indices = None
    n_rows = 0

    for row in iter_jsonl(path):
        split = str(row.get("split", "eval")).strip().lower()
        if split != "eval":
            continue

        q1 = safe_float(row.get("delta_q1"))
        q2 = safe_float(row.get("delta_q2"))
        q3 = safe_float(row.get("delta_q3"))
        q4 = safe_float(row.get("delta_q4"))

        if None in (q1, q2, q3, q4):
            continue

        vals_q1.append(q1)
        vals_q2.append(q2)
        vals_q3.append(q3)
        vals_q4.append(q4)
        n_rows += 1

        if relative_layers is None and "relative_layers" in row:
            relative_layers = row["relative_layers"]
        if layer_indices is None and "layer_indices" in row:
            layer_indices = row["layer_indices"]

    if n_rows == 0:
        raise RuntimeError(f"No valid eval rows found in: {path}")

    return {
        "path": path,
        "n_eval_users": n_rows,
        "relative_layers": relative_layers if relative_layers is not None else [0.25, 0.5, 0.75, 1.0],
        "layer_indices": layer_indices if layer_indices is not None else [],
        "mean_delta_q1": float(np.mean(vals_q1)),
        "mean_delta_q2": float(np.mean(vals_q2)),
        "mean_delta_q3": float(np.mean(vals_q3)),
        "mean_delta_q4": float(np.mean(vals_q4)),
        "std_delta_q1": float(np.std(vals_q1, ddof=0)),
        "std_delta_q2": float(np.std(vals_q2, ddof=0)),
        "std_delta_q3": float(np.std(vals_q3, ddof=0)),
        "std_delta_q4": float(np.std(vals_q4, ddof=0)),
    }


def fmt_num(x: float, digits: int = 2) -> str:
    if x == 0:
        return f"{0:.{digits}f}"
    # 小量级用科学计数法，更适合你 appendix 这张表
    if abs(x) < 1e-4:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def make_latex_row(model: str, summary: Dict[str, Any], digits: int = 2) -> str:
    return (
        f"{model}"
        f" & {fmt_num(summary['mean_delta_q1'], digits)}"
        f" & {fmt_num(summary['mean_delta_q2'], digits)}"
        f" & {fmt_num(summary['mean_delta_q3'], digits)}"
        f" & {fmt_num(summary['mean_delta_q4'], digits)} \\\\"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--specs_json",
        default="",
        help=(
            "Optional JSON file containing a list of specs with keys: "
            "attribute, dataset, model, path. "
            "If omitted, built-in DEFAULT_SPECS is used."
        ),
    )
    ap.add_argument("--out_json", default=OUT_JSON_DEFAULT)
    ap.add_argument("--out_csv", default=OUT_CSV_DEFAULT)
    ap.add_argument("--out_tex", default=OUT_TEX_DEFAULT)
    ap.add_argument("--digits", type=int, default=2, help="Digits for LaTeX numeric formatting")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_json))
    ensure_dir(os.path.dirname(args.out_csv))
    ensure_dir(os.path.dirname(args.out_tex))

    if args.specs_json:
        with open(args.specs_json, "r", encoding="utf-8") as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            raise ValueError("--specs_json must contain a JSON list")
    else:
        specs = DEFAULT_SPECS

    results: List[Dict[str, Any]] = []

    for spec in specs:
        attribute = str(spec["attribute"])
        dataset = str(spec["dataset"])
        model = str(spec["model"])
        path = str(spec["path"])

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file for {attribute}/{dataset}/{model}: {path}")

        summary = summarize_internal_layers(path)
        rec = {
            "attribute": attribute,
            "dataset": dataset,
            "model": model,
            **summary,
            "latex_row": make_latex_row(model, summary, digits=args.digits),
        }
        results.append(rec)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "attribute",
        "dataset",
        "model",
        "n_eval_users",
        "mean_delta_q1",
        "mean_delta_q2",
        "mean_delta_q3",
        "mean_delta_q4",
        "std_delta_q1",
        "std_delta_q2",
        "std_delta_q3",
        "std_delta_q4",
        "path",
        "latex_row",
    ]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results:
            writer.writerow({k: rec.get(k) for k in fieldnames})

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for rec in results:
        grouped.setdefault(rec["attribute"], {}).setdefault(rec["dataset"], []).append(rec)

    with open(args.out_tex, "w", encoding="utf-8") as f:
        for attribute in ["Gender", "Age", "Race"]:
            if attribute not in grouped:
                continue
            f.write(f"% ===== {attribute} =====\n")
            for dataset in ["MovieLens", "Goodreads", "SteamReviews"]:
                if dataset not in grouped[attribute]:
                    continue
                f.write(f"% --- {dataset} ---\n")
                rows = sorted(grouped[attribute][dataset], key=lambda x: x["model"])
                for rec in rows:
                    f.write(rec["latex_row"] + "\n")
                if rows:
                    avg_q1 = float(np.mean([r["mean_delta_q1"] for r in rows]))
                    avg_q2 = float(np.mean([r["mean_delta_q2"] for r in rows]))
                    avg_q3 = float(np.mean([r["mean_delta_q3"] for r in rows]))
                    avg_q4 = float(np.mean([r["mean_delta_q4"] for r in rows]))
                    avg_row = (
                        f"\\textit{{Avg.}}"
                        f" & {fmt_num(avg_q1, args.digits)}"
                        f" & {fmt_num(avg_q2, args.digits)}"
                        f" & {fmt_num(avg_q3, args.digits)}"
                        f" & {fmt_num(avg_q4, args.digits)} \\\\"
                    )
                    f.write(avg_row + "\n")
                f.write("\n")

    print(f"[OK] Wrote JSON summary -> {args.out_json}")
    print(f"[OK] Wrote CSV summary  -> {args.out_csv}")
    print(f"[OK] Wrote LaTeX rows   -> {args.out_tex}")


if __name__ == "__main__":
    main()