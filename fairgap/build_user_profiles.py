#!/usr/bin/env python3
"""
fairgap_out/1build_user_profiles.py

Build per-user movie-only profiles from MovieLens-1M.

Inputs:
  - examples/ml-1m/users.dat
  - examples/ml-1m/ratings.dat
  - examples/ml-1m/movies.dat

Output:
  - fairgap_out/profiles.jsonl

Each JSONL row contains:
  - user_id
  - top_rated_movies: list of {movie_id, title, rating, timestamp, genres}
  - genre_summary (optional)

Construction rule:
  - keep only users with at least 10 valid historical interactions
  - valid means the rated movie exists in movies.dat
  - retain up to top_n items per user using deterministic ranking

python /path/to/fairgap/facter/5sec_Gemma7/1build_user_profiles.py \
  --out examples/toy_out/profiles.jsonl \
  --top_n 20 \
  --min_interactions 10 \
  --include_genre_summary
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


MOVIES_PATH_DEFAULT = "examples/ml-1m/movies.dat"
RATINGS_PATH_DEFAULT = "examples/ml-1m/ratings.dat"
USERS_PATH_DEFAULT = "examples/ml-1m/users.dat"

OUT_DIR_DEFAULT = "examples/toy_out"
OUT_JSONL_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "profiles.jsonl")


@dataclass
class Movie:
    movie_id: int
    title: str
    genres: List[str]


def parse_movies(movies_path: str) -> Dict[int, Movie]:
    movies: Dict[int, Movie] = {}
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 3:
                continue
            mid = int(parts[0])
            title = parts[1]
            genres = parts[2].split("|") if parts[2] else []
            movies[mid] = Movie(movie_id=mid, title=title, genres=genres)
    return movies


def parse_users(users_path: str) -> Dict[int, Dict[str, int]]:
    """
    users.dat format:
      UserID::Gender::Age::Occupation::Zip-code

    We only keep user IDs so output stays movie-only.
    """
    users: Dict[int, Dict[str, int]] = {}
    with open(users_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 5:
                continue
            uid = int(parts[0])
            users[uid] = {"user_id": uid}
    return users


def parse_ratings(ratings_path: str) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    ratings.dat format:
      UserID::MovieID::Rating::Timestamp

    Return:
      ratings_by_user[user_id] = list of (movie_id, rating, timestamp)
    """
    ratings_by_user: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 4:
                continue
            uid = int(parts[0])
            mid = int(parts[1])
            rating = int(parts[2])
            ts = int(parts[3])
            ratings_by_user[uid].append((mid, rating, ts))
    return ratings_by_user


def select_top_n(ratings: List[Tuple[int, int, int]], n: int) -> List[Tuple[int, int, int]]:
    """
    Select Top-N movies with a deterministic rule:
      - group by rating desc (5 -> 1)
      - within each rating group, sort by timestamp desc
      - take first N from concatenated groups
    """
    buckets: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for mid, r, ts in ratings:
        buckets[r].append((mid, r, ts))

    selected: List[Tuple[int, int, int]] = []
    for r in [5, 4, 3, 2, 1]:
        if r not in buckets:
            continue
        bucket = sorted(buckets[r], key=lambda x: x[2], reverse=True)
        selected.extend(bucket)
        if len(selected) >= n:
            break
    return selected[:n]


def make_genre_summary(top_movies: List[Dict], normalize: bool = True) -> Dict[str, float]:
    c = Counter()
    for m in top_movies:
        for g in m.get("genres", []):
            if g:
                c[g] += 1

    if not c:
        return {}

    if not normalize:
        return dict(c)

    total = sum(c.values())
    return {g: round(cnt / total, 6) for g, cnt in c.most_common()}


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", default=MOVIES_PATH_DEFAULT)
    ap.add_argument("--ratings", default=RATINGS_PATH_DEFAULT)
    ap.add_argument("--users", default=USERS_PATH_DEFAULT)
    ap.add_argument("--out", default=OUT_JSONL_DEFAULT)
    ap.add_argument("--top_n", type=int, default=20, help="Top-N movies per user after filtering")
    ap.add_argument("--min_interactions", type=int, default=10, help="Minimum number of valid interactions required")
    ap.add_argument("--include_genre_summary", action="store_true", help="Add normalized genre distribution")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out))

    movies = parse_movies(args.movies)
    users = parse_users(args.users)
    ratings_by_user = parse_ratings(args.ratings)

    written = 0
    skipped_too_sparse = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for uid in sorted(users.keys()):
            uratings = ratings_by_user.get(uid, [])

            # Keep only valid interactions whose movie exists in movies.dat
            valid_ratings = [(mid, r, ts) for mid, r, ts in uratings if mid in movies]

            # Filter out users with fewer than the required number of valid interactions
            if len(valid_ratings) < args.min_interactions:
                skipped_too_sparse += 1
                continue

            top_raw = select_top_n(valid_ratings, args.top_n)
            top = []
            for mid, r, ts in top_raw:
                mv = movies[mid]
                top.append(
                    {
                        "movie_id": mv.movie_id,
                        "title": mv.title,
                        "rating": r,
                        "timestamp": ts,
                        "genres": mv.genres,
                    }
                )

            profile = {
                "user_id": uid,
                "top_rated_movies": top,
            }

            if args.include_genre_summary:
                profile["genre_summary"] = make_genre_summary(top, normalize=True)

            out_f.write(json.dumps(profile, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] Wrote {written} user profiles to: {args.out}")
    print(f"[OK] Skipped {skipped_too_sparse} users with < {args.min_interactions} valid interactions")


if __name__ == "__main__":
    main()