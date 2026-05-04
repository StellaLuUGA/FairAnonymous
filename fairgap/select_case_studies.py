#!/usr/bin/env python3
import argparse
import json
import math
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

"""
[OK] Wrote outputs to: examples/toy_out/case_studies
Selected cases:
Case 1: Output-visible mismatch | user=2003 | quadrant=Q1 | OBS=1.0000 | IBS=0.003315
Case 2: Hidden-internal mismatch | user=979 | quadrant=Q4 | OBS=0.1139 | IBS=0.024425

"""


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_first(row, keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def get_user_id(row):
    return str(
        pick_first(
            row,
            [
                "user_id",
                "uid",
                "user",
                "pair_id",
                "id",
            ],
            "",
        )
    )


def get_obs(row):
    return to_float(
        pick_first(
            row,
            [
                "OBS",
                "obs",
                "d_out",
                "output_distance",
                "rbo_distance",
                "output_shift",
                "output_bias_score",
            ],
        )
    )


def get_ibs(row):
    return to_float(
        pick_first(
            row,
            [
                "IBS",
                "ibs",
                "d_in",
                "internal_distance",
                "internal_shift",
                "weighted_internal_shift",
                "internal_bias_score",
            ],
        )
    )


def get_quadrant(row):
    q = pick_first(
        row,
        [
            "quadrant",
            "region",
            "mismatch_region",
            "joint_quadrant",
        ],
        None,
    )
    return str(q) if q is not None else ""


def collect_list_fields(obj):
    """
    Recursively collect list-of-string fields from a json object.
    This is intentionally flexible because ranked_lists.jsonl schemas vary.
    """
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and len(v) >= 3 and all(isinstance(x, str) for x in v):
                found.append((k, v))
            elif isinstance(v, dict):
                found.extend(collect_list_fields(v))
    return found


def build_rank_index(ranked_rows):
    """
    Return user_id -> list of candidate recommendation lists.
    Supports either:
    1) one row per user containing two list fields; or
    2) multiple rows per user each containing one list field.
    """
    idx = {}

    for row in ranked_rows:
        uid = get_user_id(row)
        if not uid:
            continue

        lists = collect_list_fields(row)

        for name, recs in lists:
            # Filter out fields that are clearly not recommendation lists.
            lname = name.lower()
            if any(bad in lname for bad in ["profile", "history", "liked", "prompt"]):
                continue
            idx.setdefault(uid, []).append((name, recs))

    return idx


def build_profile_index(pair_rows):
    idx = {}
    for row in pair_rows:
        uid = get_user_id(row)
        if not uid:
            continue

        profile = pick_first(
            row,
            [
                "profile",
                "user_profile",
                "profile_text",
                "history_text",
                "prompt",
                "prompt_a",
                "base_prompt",
            ],
            "",
        )

        if isinstance(profile, dict):
            profile = json.dumps(profile, ensure_ascii=False)
        elif not isinstance(profile, str):
            profile = str(profile)

        idx[uid] = profile
    return idx


def short_profile(text, max_chars=220):
    if not text:
        return ""
    text = " ".join(text.replace("\n", " ").split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def normalize(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {}
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {v: 0.0 for v in vals}
    return {v: (v - mn) / (mx - mn) for v in vals}


def select_cases(qrows, rank_idx, profile_idx, n_per_type=1):
    records = []

    for row in qrows:
        uid = get_user_id(row)
        obs = get_obs(row)
        ibs = get_ibs(row)
        quadrant = get_quadrant(row)

        if not uid or obs is None or ibs is None:
            continue

        rec_lists = rank_idx.get(uid, [])
        if len(rec_lists) < 2:
            # Still keep it, but it cannot be used for full case card.
            has_lists = False
        else:
            has_lists = True

        records.append(
            {
                "user_id": uid,
                "obs": obs,
                "ibs": ibs,
                "quadrant": quadrant,
                "profile": short_profile(profile_idx.get(uid, "")),
                "has_lists": has_lists,
                "raw": row,
            }
        )

    if not records:
        raise ValueError("No usable case records found. Check field names in quadrants file.")

    obs_vals = [r["obs"] for r in records]
    ibs_vals = [r["ibs"] for r in records]

    obs_min, obs_max = min(obs_vals), max(obs_vals)
    ibs_min, ibs_max = min(ibs_vals), max(ibs_vals)

    def z_obs(x):
        return 0 if obs_max == obs_min else (x - obs_min) / (obs_max - obs_min)

    def z_ibs(x):
        return 0 if ibs_max == ibs_min else (x - ibs_min) / (ibs_max - ibs_min)

    for r in records:
        r["output_visible_score"] = z_obs(r["obs"]) - z_ibs(r["ibs"])
        r["hidden_internal_score"] = z_ibs(r["ibs"]) - z_obs(r["obs"])

    # Prefer rows with actual recommendation lists.
    usable = [r for r in records if r["has_lists"]]
    if not usable:
        usable = records

    q1_like = [
        r for r in usable
        if "1" in r["quadrant"] or "output" in r["quadrant"].lower()
    ]
    q4_like = [
        r for r in usable
        if "4" in r["quadrant"] or "hidden" in r["quadrant"].lower()
    ]

    # Fallback if quadrant labels are absent or nonstandard.
    if not q1_like:
        q1_like = sorted(usable, key=lambda r: r["output_visible_score"], reverse=True)
    else:
        q1_like = sorted(q1_like, key=lambda r: r["output_visible_score"], reverse=True)

    if not q4_like:
        q4_like = sorted(usable, key=lambda r: r["hidden_internal_score"], reverse=True)
    else:
        q4_like = sorted(q4_like, key=lambda r: r["hidden_internal_score"], reverse=True)

    selected = []
    for r in q1_like[:n_per_type]:
        r = dict(r)
        r["case_type"] = "Output-visible mismatch"
        selected.append(r)

    for r in q4_like[:n_per_type]:
        r = dict(r)
        r["case_type"] = "Hidden-internal mismatch"
        selected.append(r)

    return records, selected


def write_csv(path, selected, rank_idx):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_type",
                "user_id",
                "quadrant",
                "OBS",
                "IBS",
                "profile",
                "list_A",
                "list_B",
            ],
        )
        writer.writeheader()

        for r in selected:
            lists = rank_idx.get(r["user_id"], [])
            list_a = lists[0][1] if len(lists) >= 1 else []
            list_b = lists[1][1] if len(lists) >= 2 else []

            writer.writerow(
                {
                    "case_type": r["case_type"],
                    "user_id": r["user_id"],
                    "quadrant": r["quadrant"],
                    "OBS": f"{r['obs']:.4f}",
                    "IBS": f"{r['ibs']:.6f}",
                    "profile": r["profile"],
                    "list_A": " | ".join(list_a[:10]),
                    "list_B": " | ".join(list_b[:10]),
                }
            )


def latex_escape(s):
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def write_latex(path, selected, dataset, attribute, model, rank_idx):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{Representative mismatch cases for {latex_escape(model)} on {latex_escape(dataset)} / {latex_escape(attribute)}.}}"
    )
    lines.append(
        rf"\label{{tab:case-{dataset.lower()}-{attribute.lower()}-{model.lower().replace('/', '-')}}}"
    )
    lines.append(r"\begin{tabular}{llccp{7.2cm}}")
    lines.append(r"\toprule")
    lines.append(r"Case type & User & OBS & IBS & Qualitative pattern \\")
    lines.append(r"\midrule")

    for r in selected:
        if r["case_type"] == "Output-visible mismatch":
            pattern = (
                "The protected-attribute flip substantially changes the visible recommendation list, "
                "while the internal representation shift remains comparatively limited."
            )
        else:
            pattern = (
                "The final recommendation lists remain broadly similar, but the internal representation "
                "shift is comparatively large."
            )

        lines.append(
            f"{latex_escape(r['case_type'])} & "
            f"{latex_escape(r['user_id'])} & "
            f"{r['obs']:.3f} & "
            f"{r['ibs']:.4f} & "
            f"{latex_escape(pattern)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append("% Detailed recommendation lists for appendix")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        rf"\caption{{Detailed recommendation lists for selected {latex_escape(attribute)} cases.}}"
    )
    lines.append(r"\begin{tabular}{p{2.6cm}p{5.8cm}p{5.8cm}}")
    lines.append(r"\toprule")
    lines.append(r"Case & List A & List B \\")
    lines.append(r"\midrule")

    for r in selected:
        lists = rank_idx.get(r["user_id"], [])
        list_a = lists[0][1][:10] if len(lists) >= 1 else []
        list_b = lists[1][1][:10] if len(lists) >= 2 else []

        lines.append(
            f"{latex_escape(r['case_type'])} & "
            f"{latex_escape('; '.join(list_a))} & "
            f"{latex_escape('; '.join(list_b))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def make_scatter(path, records, selected, dataset, attribute, model):
    xs = [r["obs"] for r in records]
    ys = [r["ibs"] for r in records]

    x_thr = sorted(xs)[int(0.5 * (len(xs) - 1))]
    y_thr = sorted(ys)[int(0.5 * (len(ys) - 1))]

    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(xs, ys, alpha=0.35, s=18)
    plt.axvline(x_thr, linestyle="--", linewidth=1)
    plt.axhline(y_thr, linestyle="--", linewidth=1)

    for i, r in enumerate(selected, start=1):
        plt.scatter([r["obs"]], [r["ibs"]], s=90, marker="x")
        plt.text(r["obs"], r["ibs"], f" Case {i}", fontsize=9)

    plt.xlabel("OBS: output shift")
    plt.ylabel("IBS: internal shift")
    plt.title(f"{model} on {dataset} / {attribute}")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quadrants", required=True)
    ap.add_argument("--ranked_lists", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_type", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qrows = read_jsonl(args.quadrants)
    ranked_rows = read_jsonl(args.ranked_lists)
    pair_rows = read_jsonl(args.pairs)

    rank_idx = build_rank_index(ranked_rows)
    profile_idx = build_profile_index(pair_rows)

    records, selected = select_cases(
        qrows,
        rank_idx=rank_idx,
        profile_idx=profile_idx,
        n_per_type=args.n_per_type,
    )

    stem = f"{args.dataset}_{args.attribute}_{args.model}".replace("/", "_").replace(" ", "_")

    write_csv(out_dir / f"{stem}_case_cards.csv", selected, rank_idx)
    write_latex(out_dir / f"{stem}_case_cards.tex", selected, args.dataset, args.attribute, args.model, rank_idx)
    make_scatter(out_dir / f"{stem}_ibs_obs_cases.png", records, selected, args.dataset, args.attribute, args.model)

    print(f"[OK] Wrote outputs to: {out_dir}")
    print("Selected cases:")
    for i, r in enumerate(selected, start=1):
        print(
            f"Case {i}: {r['case_type']} | user={r['user_id']} | "
            f"quadrant={r['quadrant']} | OBS={r['obs']:.4f} | IBS={r['ibs']:.6f}"
        )


if __name__ == "__main__":
    main()