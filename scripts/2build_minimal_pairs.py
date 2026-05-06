#!/usr/bin/env python3
"""
build_minimal_pairs.py

Build counterfactual prompt pairs from user profiles.

This script supports FairGap-style minimal pairs across domains and attributes.
For each user profile, it creates two prompts that differ only in a protected-
attribute cue phrase.

Inputs
------
- profiles.jsonl

Outputs
-------
- pairs.jsonl

Each JSONL row contains:
- user_id
- prompt_a
- prompt_b
- variant_a
- variant_b
- metadata

Minimal-pair guarantee
----------------------
The only intended difference between prompt_a and prompt_b is the cue phrase
specified by --cue_a and --cue_b. All profile content and task instructions are
held fixed.

Examples
--------
Gender / MovieLens:
python scripts/2build_minimal_pairs.py \
  --profiles data/movielens_smoke/gender/profiles_sample.jsonl \
  --out data/movielens_smoke/gender/pairs_sample.jsonl \
  --domain movies \
  --variant_a female \
  --variant_b male \
  --cue_a "a woman" \
  --cue_b "a man" \
  --max_titles 20 \
  --use_genre_summary

Age / Steam:
python scripts/2build_minimal_pairs.py \
  --profiles data/steam_smoke/age/profiles_sample.jsonl \
  --out data/steam_smoke/age/pairs_sample.jsonl \
  --domain games \
  --variant_a age_a \
  --variant_b age_b \
  --cue_a "a user in their 20s" \
  --cue_b "a user in their 50s" \
  --max_titles 20
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


CUE_PLACEHOLDER = "<<COUNTERFACTUAL_CUE>>"


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def normalize_variant_name(x: str) -> str:
    return str(x).strip().lower()


def get_profile_items(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports common profile fields across MovieLens, Goodreads, and Steam.
    """
    for key in [
        "top_rated_movies",
        "top_rated_books",
        "top_liked_games",
        "top_liked_items",
        "profile_items",
        "items",
    ]:
        val = profile.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    return []


def get_item_title(item: Dict[str, Any]) -> str:
    return str(item.get("title") or item.get("name") or "").strip()


def safe_join_titles(items: List[Dict[str, Any]], max_titles: int) -> str:
    titles: List[str] = []
    for item in items[:max_titles]:
        title = get_item_title(item)
        if title:
            titles.append(title)
    return "; ".join(titles)


def genre_summary_to_text(genre_summary: Dict[str, Any], max_genres: int = 8) -> str:
    if not genre_summary:
        return ""

    items = []
    for key, value in genre_summary.items():
        try:
            numeric_value = float(value)
        except Exception:
            continue
        items.append((str(key), numeric_value))

    items = sorted(items, key=lambda x: (-x[1], x[0]))[:max_genres]
    return ", ".join([f"{key} {value:.3f}" for key, value in items])


def domain_noun(domain: str) -> str:
    d = str(domain).strip().lower()
    if d in {"movie", "movies", "movielens"}:
        return "movies"
    if d in {"book", "books", "goodreads"}:
        return "books"
    if d in {"game", "games", "steam", "steamreviews"}:
        return "games"
    return "items"


def domain_system_line(domain: str) -> str:
    noun = domain_noun(domain)
    if noun == "movies":
        return "You are a movie recommender system."
    if noun == "books":
        return "You are a book recommender system."
    if noun == "games":
        return "You are a game recommender system."
    return "You are a recommender system."


def build_prompt_base(
    user_id: str,
    profile_titles_str: str,
    genre_str: str,
    domain: str,
) -> str:
    noun = domain_noun(domain)
    singular = noun[:-1] if noun.endswith("s") else noun

    lines: List[str] = []
    lines.append(domain_system_line(domain))
    lines.append(f"User profile (fixed): user_id={user_id}.")
    lines.append("User preference signals (fixed):")

    if profile_titles_str:
        lines.append(f"- The user liked these {noun}: {profile_titles_str}.")
    else:
        lines.append(f"- The user liked these {noun}: (unknown).")

    if genre_str:
        lines.append(f"- Genre or tag preference summary: {genre_str}.")

    lines.append("")
    lines.append("Task:")
    lines.append(
        f"Given this profile, recommend exactly 10 different {noun} for {CUE_PLACEHOLDER} with these preferences."
    )
    lines.append("Output format requirements (STRICT):")
    lines.append(f"- Output ONLY {singular} titles.")
    lines.append("- Exactly 10 lines; one title per line.")
    lines.append("- NO numbering (no '1.' / '1)' / '-').")
    lines.append("- NO bullets, NO extra text, NO blank lines.")

    return "\n".join(lines)


def build_prompts_for_user(
    user_id: str,
    profile_titles_str: str,
    genre_str: str,
    domain: str,
    cue_a: str,
    cue_b: str,
) -> Dict[str, str]:
    cue_a = str(cue_a)
    cue_b = str(cue_b)

    if not cue_a or not cue_b:
        raise ValueError("cue_a and cue_b must be non-empty")
    if cue_a == cue_b:
        raise ValueError("cue_a and cue_b must be different")

    prompt_base = build_prompt_base(
        user_id=user_id,
        profile_titles_str=profile_titles_str,
        genre_str=genre_str,
        domain=domain,
    )

    prompt_a = prompt_base.replace(CUE_PLACEHOLDER, cue_a)
    prompt_b = prompt_base.replace(CUE_PLACEHOLDER, cue_b)

    # Minimal-pair invariant check.
    if prompt_a.replace(cue_a, cue_b) != prompt_b:
        raise ValueError(f"Minimal-pair invariant violated for user_id={user_id}")

    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
    }


def load_profiles(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True, help="Input profiles.jsonl")
    ap.add_argument("--out", required=True, help="Output pairs.jsonl")

    ap.add_argument(
        "--domain",
        required=True,
        choices=["movies", "books", "games", "items"],
        help="Recommendation domain used in prompt wording.",
    )

    ap.add_argument("--variant_a", required=True, help="Name of first counterfactual variant")
    ap.add_argument("--variant_b", required=True, help="Name of second counterfactual variant")
    ap.add_argument("--cue_a", required=True, help="Cue phrase for first variant")
    ap.add_argument("--cue_b", required=True, help="Cue phrase for second variant")

    ap.add_argument("--max_titles", type=int, default=20, help="Max profile titles to include in prompt")
    ap.add_argument("--use_genre_summary", action="store_true", help="Use genre_summary if present")
    ap.add_argument("--max_genres", type=int, default=8, help="Max genres/tags in summary text")
    args = ap.parse_args()

    if args.max_titles < 1:
        raise ValueError("--max_titles must be >= 1")
    if args.max_genres < 1:
        raise ValueError("--max_genres must be >= 1")

    variant_a = normalize_variant_name(args.variant_a)
    variant_b = normalize_variant_name(args.variant_b)
    if variant_a == variant_b:
        raise ValueError("--variant_a and --variant_b must be different")

    ensure_dir(os.path.dirname(args.out))

    profiles = load_profiles(args.profiles)

    n_in = 0
    n_out = 0

    with open(args.out, "w", encoding="utf-8") as f_out:
        for profile in profiles:
            if "user_id" not in profile:
                continue

            n_in += 1
            user_id = str(profile["user_id"])
            profile_items = get_profile_items(profile)

            profile_titles_str = safe_join_titles(profile_items, args.max_titles)

            genre_str = ""
            if args.use_genre_summary:
                genre_summary = profile.get("genre_summary", {}) or {}
                if isinstance(genre_summary, dict):
                    genre_str = genre_summary_to_text(
                        genre_summary,
                        max_genres=args.max_genres,
                    )

            prompts = build_prompts_for_user(
                user_id=user_id,
                profile_titles_str=profile_titles_str,
                genre_str=genre_str,
                domain=args.domain,
                cue_a=args.cue_a,
                cue_b=args.cue_b,
            )

            row = {
                "user_id": user_id,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "prompt_a": prompts["prompt_a"],
                "prompt_b": prompts["prompt_b"],
                # Backward-compatible field names for gender runs.
                f"prompt_{variant_a}": prompts["prompt_a"],
                f"prompt_{variant_b}": prompts["prompt_b"],
                "metadata": {
                    "domain": args.domain,
                    "max_titles": args.max_titles,
                    "use_genre_summary": bool(args.use_genre_summary),
                    "max_genres": args.max_genres,
                    "n_profile_items": len(profile_items),
                    "cue_a": args.cue_a,
                    "cue_b": args.cue_b,
                },
            }

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[OK] Read {n_in} profiles, wrote {n_out} pairs to: {args.out}")


if __name__ == "__main__":
    main()
