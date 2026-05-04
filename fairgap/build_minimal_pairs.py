#!/usr/bin/env python3
"""
fairgap/2build_minimal_pairs.py

Input:
  - fairgap_out/profiles.jsonl

Output:
  - fairgap_out/pairs.jsonl

Each JSONL row:
  {
    "user_id": <int>,
    "prompt_female": <str>,
    "prompt_male": <str>,
    "metadata": {
        "max_titles": <int>,
        "use_genre_summary": <bool>,
        "max_genres": <int>,
        "n_top_movies_in_profile": <int>
    }
  }

Minimal-pair guarantee:
  - The ONLY difference between prompt_female and prompt_male is:
      "The user is a woman." vs "The user is a man."
  - Everything else is byte-identical.

a person in their 20s" ↔ "a person in their 50s"
"""
#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Any


# -----------------------------
# Defaults
# -----------------------------
IN_PROFILES_DEFAULT = "examples/toy_out/profiles.jsonl"
OUT_DIR_DEFAULT = "examples/toy_out"
OUT_PAIRS_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "pairs.jsonl")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def safe_join_titles(top_movies: List[Dict[str, Any]], max_titles: int) -> str:
    titles: List[str] = []
    for m in top_movies[:max_titles]:
        t = (m.get("title") or "").strip()
        if t:
            titles.append(t)
    return "; ".join(titles)


def genre_summary_to_text(gs: Dict[str, Any], max_genres: int = 8) -> str:
    if not gs:
        return ""
    items = sorted(gs.items(), key=lambda x: (-float(x[1]), str(x[0])))
    items = items[:max_genres]
    return ", ".join([f"{k} {float(v):.3f}" for k, v in items])


def build_prompts_for_user(
    user_id: int,
    liked_titles_str: str,
    genre_str: str,
) -> Dict[str, str]:

    AGE_PLACEHOLDER = "<<AGE_PHRASE>>"

    lines: List[str] = []
    lines.append("You are a movie recommender system.")
    lines.append(f"User profile (fixed): user_id={user_id}.")
    lines.append("User preference signals (fixed):")

    if liked_titles_str:
        lines.append(f"- The user liked these movies: {liked_titles_str}.")
    else:
        lines.append("- The user liked these movies: (unknown).")

    if genre_str:
        lines.append(f"- Genre preference summary: {genre_str}.")

    lines.append("")
    lines.append("Task:")
    lines.append(
        f"Given this profile, recommend exactly 10 different movies for {AGE_PLACEHOLDER} with these preferences."
    )

    lines.append("Output format requirements (STRICT):")
    lines.append("- Output ONLY movie titles.")
    lines.append("- Exactly 10 lines; one title per line.")
    lines.append("- NO numbering (no '1.' / '1)' / '-').")
    lines.append("- NO bullets, NO extra text, NO blank lines.")

    prompt_base = "\n".join(lines)

    age_a_phrase = "a person in their 20s"
    age_b_phrase = "a person in their 50s"

    prompt_age_a = prompt_base.replace(AGE_PLACEHOLDER, age_a_phrase)
    prompt_age_b = prompt_base.replace(AGE_PLACEHOLDER, age_b_phrase)

    if prompt_age_a.replace(age_a_phrase, age_b_phrase) != prompt_age_b:
        raise ValueError(f"Minimal-pair invariant violated for user_id={user_id}")

    return {
        "prompt_age_a": prompt_age_a,
        "prompt_age_b": prompt_age_b,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=IN_PROFILES_DEFAULT, help="Input profiles.jsonl")
    ap.add_argument("--out", default=OUT_PAIRS_DEFAULT, help="Output pairs.jsonl")
    ap.add_argument("--max_titles", type=int, default=20, help="Max liked titles to include in prompt")
    ap.add_argument("--use_genre_summary", action="store_true", help="Use genre_summary if present in profiles")
    ap.add_argument("--max_genres", type=int, default=8, help="Max genres in summary text (if used)")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out))

    n_in = 0
    n_out = 0

    with open(args.profiles, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            prof = json.loads(line)

            user_id = int(prof["user_id"])
            top_movies = prof.get("top_rated_movies", []) or []

            liked_titles_str = safe_join_titles(top_movies, args.max_titles)

            genre_str = ""
            if args.use_genre_summary:
                gs = prof.get("genre_summary", {}) or {}
                genre_str = genre_summary_to_text(gs, max_genres=args.max_genres)

            prompts = build_prompts_for_user(
                user_id=user_id,
                liked_titles_str=liked_titles_str,
                genre_str=genre_str,
            )

            row = {
                    "user_id": user_id,
                    "prompt_age_a": prompts["prompt_age_a"],
                    "prompt_age_b": prompts["prompt_age_b"],
                    "metadata": {
                        "max_titles": args.max_titles,
                        "use_genre_summary": bool(args.use_genre_summary),
                        "max_genres": args.max_genres,
                        "n_top_movies_in_profile": len(top_movies),
                        "counterfactual_attribute": "age",
                        "group_a_label": "20s",
                        "group_b_label": "50s",
    },
}

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[OK] Read {n_in} profiles, wrote {n_out} pairs to: {args.out}")


if __name__ == "__main__":
    main()