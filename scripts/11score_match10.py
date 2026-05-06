#!/usr/bin/env python3
"""
score_match10.py

Compute Match@10 as a profile-consistency metric for counterfactual
recommendation pairs.

Definition
----------
For each user u, let G_u be the set of genres/tags appearing in the user's
profile items. For each recommended title r in the top-10 list, we recover its
item genres/tags and count it as a match iff:

    GenresOrTags(r) ∩ G_u != empty.

For each counterfactual variant, we report two versions:

1) strict:
    Match@10_strict_v(u) = (# matched recommended items among top-10) / 10

    Unresolved titles are treated as non-matching.

2) resolved:
    Match@10_resolved_v(u) = (# matched recommended items among resolved top-10)
                             / max(# resolved recommended items among top-10, 1)

    Unresolved titles are excluded from both numerator and denominator.

We report pair means and disparities:

    match10_mean_strict(u)
    match10_mean_resolved(u)
    delta_match10_strict(u)
    delta_match10_resolved(u)

Inputs
------
- profiles.jsonl
- ranked_lists.jsonl
- split.jsonl
- optional catalog_jsonl or catalog_csv with title and genres/tags

Outputs
-------
- match10.dev.jsonl
- match10.eval.jsonl
- match10_summary.json

Example
-------
python scripts/11score_match10.py \
  --profiles data/movielens_smoke/gender/profiles_sample.jsonl \
  --ranked_lists data/movielens_smoke/gender/ranked_lists_sample.jsonl \
  --split data/movielens_smoke/gender/split_sample.jsonl \
  --catalog_csv data/movielens_smoke/gender/catalog_sample.csv \
  --catalog_title_col title \
  --catalog_genres_col genres \
  --variant_a female \
  --variant_b male \
  --require_parse_ok \
  --out_dev data/movielens_smoke/gender/match10_sample.dev.jsonl \
  --out_eval data/movielens_smoke/gender/match10_sample.eval.jsonl \
  --out_summary data/movielens_smoke/gender/match10_summary.json
"""

import argparse
import csv
import json
import os
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


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


def clean_genre_list(genres: Any) -> List[str]:
    if genres is None:
        return []
    if isinstance(genres, list):
        vals = genres
    elif isinstance(genres, str):
        if "|" in genres:
            vals = genres.split("|")
        else:
            vals = genres.split(",")
    else:
        return []

    out: List[str] = []
    for g in vals:
        gs = str(g).strip().casefold()
        if gs and gs not in {"(no genres listed)", "no genres listed"}:
            out.append(gs)
    return out


def normalize_title(title: str) -> str:
    s = str(title or "").strip()
    s = s.strip("\"'“”‘’")
    s = re.sub(r"\s+", " ", s)

    # Remove a trailing year suffix like "Toy Story (1995)".
    s = re.sub(r"\s*\((19|20)\d{2}\)\s*$", "", s)

    s = s.replace("\u2019", "'").replace("`", "'")
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s.casefold()


def normalize_variant_name(x: str) -> str:
    return str(x).strip().lower()


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


def _get_profile_items(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports common profile item fields used across domains.
    """
    for key in [
        "top_rated_movies",
        "top_liked_items",
        "top_liked_games",
        "top_rated_books",
        "profile_items",
        "items",
    ]:
        vals = row.get(key)
        if isinstance(vals, list):
            return [x for x in vals if isinstance(x, dict)]
    return []


def _get_item_genres_or_tags(item: Dict[str, Any]) -> List[str]:
    """
    Supports genres/tags/categories fields across movie/book/game profiles.
    """
    for key in ["genres", "genre", "tags", "tag", "categories", "category"]:
        if key in item:
            vals = clean_genre_list(item.get(key))
            if vals:
                return vals
    return []


def load_profiles(
    path: str,
) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]], Dict[str, Set[str]]]:
    """
    Returns:
      profile_genres_by_user[user_id] = set of genres/tags from profile items
      profile_genres_sorted_by_user[user_id] = sorted genre/tag list
      fallback_title_to_genres[normalized_title] = genres/tags aggregated from profiles
    """
    profile_genres_by_user: Dict[str, Set[str]] = {}
    profile_genres_sorted_by_user: Dict[str, List[str]] = {}
    fallback_title_to_genres: Dict[str, Set[str]] = defaultdict(set)

    for row in iter_jsonl(path):
        if "user_id" not in row:
            continue

        uid = str(row["user_id"])
        profile_items = _get_profile_items(row)
        genre_set: Set[str] = set()

        for item in profile_items:
            title = str(item.get("title", "") or item.get("name", "") or "").strip()
            genres = _get_item_genres_or_tags(item)

            for g in genres:
                genre_set.add(g)

            if title and genres:
                norm_title = normalize_title(title)
                for g in genres:
                    fallback_title_to_genres[norm_title].add(g)

        profile_genres_by_user[uid] = genre_set
        profile_genres_sorted_by_user[uid] = sorted(genre_set)

    return (
        profile_genres_by_user,
        profile_genres_sorted_by_user,
        fallback_title_to_genres,
    )


def load_catalog_jsonl(path: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    if not path:
        return out
    for row in iter_jsonl(path):
        title = str(row.get("title", "") or row.get("name", "") or "").strip()
        genres = []
        for key in ["genres", "genre", "tags", "tag", "categories", "category"]:
            if key in row:
                genres = clean_genre_list(row.get(key))
                if genres:
                    break
        if not title or not genres:
            continue
        norm_title = normalize_title(title)
        for g in genres:
            out[norm_title].add(g)
    return out


def load_catalog_csv(path: str, title_col: str, genres_col: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    if not path:
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = str(row.get(title_col, "") or "").strip()
            genres = clean_genre_list(row.get(genres_col, []))
            if not title or not genres:
                continue
            norm_title = normalize_title(title)
            for g in genres:
                out[norm_title].add(g)
    return out


def merge_title_maps(
    primary: Dict[str, Set[str]], fallback: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    merged: Dict[str, Set[str]] = defaultdict(set)
    for d in (primary, fallback):
        for t, gs in d.items():
            for g in gs:
                merged[t].add(g)
    return merged


def load_ranked_lists(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    rows_by_user: Dict[str, Dict[str, Dict[str, Any]]] = {}
    seen: Dict[Tuple[str, str], int] = defaultdict(int)

    for row in iter_jsonl(path):
        if "user_id" not in row or "variant" not in row:
            continue

        uid = str(row["user_id"])
        variant = normalize_variant_name(row["variant"])

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


def score_variant_match10(
    ranked_titles: List[str],
    profile_genres: Set[str],
    title_to_genres: Dict[str, Set[str]],
    k: int = 10,
) -> Dict[str, Any]:
    """
    Returns strict and resolved Match@10 scores.
    """
    titles = ranked_titles[:k]
    n_titles_used = len(titles)
    n_matched = 0
    n_unresolved = 0
    n_resolved = 0

    for title in titles:
        norm_t = normalize_title(title)
        rec_genres = title_to_genres.get(norm_t)

        if not rec_genres:
            n_unresolved += 1
            continue

        n_resolved += 1
        if rec_genres.intersection(profile_genres):
            n_matched += 1

    strict_score = float(n_matched) / float(k)
    resolved_score = float(n_matched) / float(n_resolved) if n_resolved > 0 else 0.0

    return {
        "strict_score": strict_score,
        "resolved_score": resolved_score,
        "n_titles_used": n_titles_used,
        "n_resolved": n_resolved,
        "n_matched": n_matched,
        "n_unresolved": n_unresolved,
    }


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

    vals = sorted(float(x) for x in values)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = var ** 0.5
    median = (
        vals[n // 2]
        if n % 2 == 1
        else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    )

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": vals[0],
        "median": median,
        "max": vals[-1],
    }


def safe_rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True, help="Input profiles JSONL")
    ap.add_argument("--ranked_lists", required=True, help="Input ranked_lists JSONL")
    ap.add_argument("--split", required=True, help="Input split JSONL")

    ap.add_argument("--catalog_jsonl", default="", help="Optional catalog JSONL")
    ap.add_argument("--catalog_csv", default="", help="Optional catalog CSV")
    ap.add_argument("--catalog_title_col", default="title")
    ap.add_argument("--catalog_genres_col", default="genres")

    ap.add_argument("--variant_a", default="female", help="First counterfactual variant name")
    ap.add_argument("--variant_b", default="male", help="Second counterfactual variant name")

    ap.add_argument("--out_dev", required=True, help="Output dev JSONL")
    ap.add_argument("--out_eval", required=True, help="Output eval JSONL")
    ap.add_argument("--out_summary", required=True, help="Output summary JSON")

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--require_parse_ok",
        action="store_true",
        help="Skip users unless both variant rows have parse_ok=True.",
    )
    args = ap.parse_args()

    if args.k != 10:
        raise ValueError("This script outputs Match@10 fields, so --k must be 10.")

    variant_a = normalize_variant_name(args.variant_a)
    variant_b = normalize_variant_name(args.variant_b)
    if variant_a == variant_b:
        raise ValueError("--variant_a and --variant_b must be different")

    ensure_dir(os.path.dirname(args.out_dev))
    ensure_dir(os.path.dirname(args.out_eval))
    ensure_dir(os.path.dirname(args.out_summary))

    split_map = load_split_map(args.split)
    (
        profile_genres_by_user,
        profile_genres_sorted_by_user,
        fallback_title_to_genres,
    ) = load_profiles(args.profiles)

    catalog_title_to_genres: Dict[str, Set[str]] = defaultdict(set)
    if args.catalog_jsonl:
        catalog_title_to_genres = merge_title_maps(
            catalog_title_to_genres, load_catalog_jsonl(args.catalog_jsonl)
        )
    if args.catalog_csv:
        catalog_title_to_genres = merge_title_maps(
            catalog_title_to_genres,
            load_catalog_csv(
                args.catalog_csv, args.catalog_title_col, args.catalog_genres_col
            ),
        )

    # Catalog first, profiles as fallback.
    title_to_genres = merge_title_maps(catalog_title_to_genres, fallback_title_to_genres)

    rows_by_user = load_ranked_lists(args.ranked_lists)

    n_users_ranked = 0
    n_pairs_scored = 0
    n_missing_split = 0
    n_missing_profile = 0
    n_missing_pair = 0
    n_parse_failed = 0
    n_dev = 0
    n_eval = 0

    dev_mean_strict_vals: List[float] = []
    eval_mean_strict_vals: List[float] = []
    dev_mean_resolved_vals: List[float] = []
    eval_mean_resolved_vals: List[float] = []

    dev_delta_strict_vals: List[float] = []
    eval_delta_strict_vals: List[float] = []
    dev_delta_resolved_vals: List[float] = []
    eval_delta_resolved_vals: List[float] = []

    total_unresolved_a = 0
    total_unresolved_b = 0
    total_resolved_a = 0
    total_resolved_b = 0
    total_titles_used_a = 0
    total_titles_used_b = 0

    with (
        open(args.out_dev, "w", encoding="utf-8") as f_dev,
        open(args.out_eval, "w", encoding="utf-8") as f_eval,
    ):
        for uid in sorted(rows_by_user.keys()):
            n_users_ranked += 1

            if uid not in split_map:
                n_missing_split += 1
                continue

            if uid not in profile_genres_by_user:
                n_missing_profile += 1
                continue

            variants = rows_by_user[uid]
            if variant_a not in variants or variant_b not in variants:
                n_missing_pair += 1
                continue

            row_a = variants[variant_a]
            row_b = variants[variant_b]

            parse_ok_a = bool(row_a.get("parse_ok", False))
            parse_ok_b = bool(row_b.get("parse_ok", False))
            if args.require_parse_ok and (not parse_ok_a or not parse_ok_b):
                n_parse_failed += 1
                continue

            titles_a = row_a.get("ranked_titles", []) or []
            titles_b = row_b.get("ranked_titles", []) or []
            if not isinstance(titles_a, list) or not isinstance(titles_b, list):
                n_missing_pair += 1
                continue

            titles_a = [str(x).strip() for x in titles_a if str(x).strip()]
            titles_b = [str(x).strip() for x in titles_b if str(x).strip()]

            profile_genres = profile_genres_by_user[uid]

            score_a = score_variant_match10(
                ranked_titles=titles_a,
                profile_genres=profile_genres,
                title_to_genres=title_to_genres,
                k=args.k,
            )
            score_b = score_variant_match10(
                ranked_titles=titles_b,
                profile_genres=profile_genres,
                title_to_genres=title_to_genres,
                k=args.k,
            )

            match10_mean_strict = 0.5 * (
                score_a["strict_score"] + score_b["strict_score"]
            )
            match10_mean_resolved = 0.5 * (
                score_a["resolved_score"] + score_b["resolved_score"]
            )
            delta_match10_strict = abs(
                score_a["strict_score"] - score_b["strict_score"]
            )
            delta_match10_resolved = abs(
                score_a["resolved_score"] - score_b["resolved_score"]
            )

            rec = {
                "user_id": uid,
                "split": split_map[uid],
                "variant_a": variant_a,
                "variant_b": variant_b,
                "profile_genres": profile_genres_sorted_by_user.get(uid, []),

                "match10_a_strict": score_a["strict_score"],
                "match10_b_strict": score_b["strict_score"],
                "match10_mean_strict": match10_mean_strict,
                "delta_match10_strict": delta_match10_strict,

                "match10_a_resolved": score_a["resolved_score"],
                "match10_b_resolved": score_b["resolved_score"],
                "match10_mean_resolved": match10_mean_resolved,
                "delta_match10_resolved": delta_match10_resolved,

                "n_titles_a": score_a["n_titles_used"],
                "n_titles_b": score_b["n_titles_used"],
                "n_resolved_a": score_a["n_resolved"],
                "n_resolved_b": score_b["n_resolved"],
                "n_matched_a": score_a["n_matched"],
                "n_matched_b": score_b["n_matched"],
                "n_unresolved_a": score_a["n_unresolved"],
                "n_unresolved_b": score_b["n_unresolved"],
                "unresolved_rate_a": safe_rate(
                    score_a["n_unresolved"], score_a["n_titles_used"]
                ),
                "unresolved_rate_b": safe_rate(
                    score_b["n_unresolved"], score_b["n_titles_used"]
                ),

                "parse_ok_a": parse_ok_a,
                "parse_ok_b": parse_ok_b,
                "k": int(args.k),
            }

            total_unresolved_a += score_a["n_unresolved"]
            total_unresolved_b += score_b["n_unresolved"]
            total_resolved_a += score_a["n_resolved"]
            total_resolved_b += score_b["n_resolved"]
            total_titles_used_a += score_a["n_titles_used"]
            total_titles_used_b += score_b["n_titles_used"]

            if split_map[uid] == "dev":
                f_dev.write(json.dumps(rec, ensure_ascii=False) + "\n")
                dev_mean_strict_vals.append(match10_mean_strict)
                dev_mean_resolved_vals.append(match10_mean_resolved)
                dev_delta_strict_vals.append(delta_match10_strict)
                dev_delta_resolved_vals.append(delta_match10_resolved)
                n_dev += 1
            else:
                f_eval.write(json.dumps(rec, ensure_ascii=False) + "\n")
                eval_mean_strict_vals.append(match10_mean_strict)
                eval_mean_resolved_vals.append(match10_mean_resolved)
                eval_delta_strict_vals.append(delta_match10_strict)
                eval_delta_resolved_vals.append(delta_match10_resolved)
                n_eval += 1

            n_pairs_scored += 1

    summary = {
        "k": int(args.k),
        "variant_a": variant_a,
        "variant_b": variant_b,
        "users_with_ranked_rows": int(n_users_ranked),
        "pairs_scored": int(n_pairs_scored),
        "missing_split": int(n_missing_split),
        "missing_profile": int(n_missing_profile),
        "missing_pair": int(n_missing_pair),
        "parse_failed_skipped": int(n_parse_failed),
        "rows_written_dev": int(n_dev),
        "rows_written_eval": int(n_eval),

        "catalog_sources": {
            "catalog_jsonl": args.catalog_jsonl,
            "catalog_csv": args.catalog_csv,
            "fallback_profiles_used": True,
        },

        "title_resolution": {
            "total_titles_used_a": int(total_titles_used_a),
            "total_titles_used_b": int(total_titles_used_b),
            "total_resolved_a": int(total_resolved_a),
            "total_resolved_b": int(total_resolved_b),
            "total_unresolved_a": int(total_unresolved_a),
            "total_unresolved_b": int(total_unresolved_b),
            "unresolved_rate_a": safe_rate(total_unresolved_a, total_titles_used_a),
            "unresolved_rate_b": safe_rate(total_unresolved_b, total_titles_used_b),
            "resolved_rate_a": safe_rate(total_resolved_a, total_titles_used_a),
            "resolved_rate_b": safe_rate(total_resolved_b, total_titles_used_b),
        },

        "dev": {
            "match10_mean_strict": summarize(dev_mean_strict_vals),
            "match10_mean_resolved": summarize(dev_mean_resolved_vals),
            "delta_match10_strict": summarize(dev_delta_strict_vals),
            "delta_match10_resolved": summarize(dev_delta_resolved_vals),
        },
        "eval": {
            "match10_mean_strict": summarize(eval_mean_strict_vals),
            "match10_mean_resolved": summarize(eval_mean_resolved_vals),
            "delta_match10_strict": summarize(eval_delta_strict_vals),
            "delta_match10_resolved": summarize(eval_delta_resolved_vals),
        },
    }

    with open(args.out_summary, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"[OK] users_with_ranked_rows={n_users_ranked}")
    print(f"[OK] pairs_scored={n_pairs_scored}")
    print(f"[OK] missing_split={n_missing_split}")
    print(f"[OK] missing_profile={n_missing_profile}")
    print(f"[OK] missing_pair={n_missing_pair}")
    print(f"[OK] parse_failed_skipped={n_parse_failed}")
    print(f"[OK] wrote dev rows={n_dev} -> {args.out_dev}")
    print(f"[OK] wrote eval rows={n_eval} -> {args.out_eval}")
    print(f"[OK] wrote summary -> {args.out_summary}")


if __name__ == "__main__":
    main()
