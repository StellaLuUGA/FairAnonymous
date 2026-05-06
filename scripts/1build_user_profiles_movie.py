#!/usr/bin/env python3
"""
build_movielens_user_profiles.py

Build per-user movie-only profiles from MovieLens-1M.

Inputs
------
- users.dat
- ratings.dat
- movies.dat

Output
------
- profiles.jsonl

Each JSONL row contains:
- user_id
- top_rated_movies: list of {movie_id, title, rating, timestamp, genres}
- genre_summary, if requested

Construction rule
-----------------
- Keep only users with at least a specified number of valid historical interactions.
- A valid interaction means the rated movie exists in movies.dat.
- Retain up to top_n items per user using deterministic ranking.

Example
-------
python scripts/1build_user_profiles.py \
  --movies raw/ml-1m/movies.dat \
  --ratings raw/ml-1m/ratings.dat \
  --users raw/ml-1m/users.dat \
  --out data/movielens_smoke/gender/profiles_sample.jsonl \
  --top_n 20 \
  --min_interactions 10 \
  --include_genre_summary
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


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

    Returns:
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
    - group by rating desc
    - within each rating group, sort by timestamp desc
    - take first N from concatenated groups
    """
    buckets: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for mid, rating, ts in ratings:
        buckets[rating].append((mid, rating, ts))

    selected: List[Tuple[int, int, int]] = []
    for rating in [5, 4, 3, 2, 1]:
        if rating not in buckets:
            continue
        bucket = sorted(buckets[rating], key=lambda x: x[2], reverse=True)
        selected.extend(bucket)
        if len(selected) >= n:
            break
    return selected[:n]


def make_genre_summary(top_movies: List[Dict], normalize: bool = True) -> Dict[str, float]:
    counter = Counter()
    for movie in top_movies:
        for genre in movie.get("genres", []):
            if genre:
                counter[genre] += 1

    if not counter:
        return {}

    if not normalize:
        return dict(counter)

    total = sum(counter.values())
    return {genre: round(count / total, 6) for genre, count in counter.most_common()}


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", required=True, help="Path to MovieLens movies.dat")
    ap.add_argument("--ratings", required=True, help="Path to MovieLens ratings.dat")
    ap.add_argument("--users", required=True, help="Path to MovieLens users.dat")
    ap.add_argument("--out", required=True, help="Output profiles JSONL")
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
            user_ratings = ratings_by_user.get(uid, [])

            valid_ratings = [
                (mid, rating, ts)
                for mid, rating, ts in user_ratings
                if mid in movies
            ]

            if len(valid_ratings) < args.min_interactions:
                skipped_too_sparse += 1
                continue

            top_raw = select_top_n(valid_ratings, args.top_n)
            top_movies = []
            for mid, rating, ts in top_raw:
                movie = movies[mid]
                top_movies.append(
                    {
                        "movie_id": movie.movie_id,
                        "title": movie.title,
                        "rating": rating,
                        "timestamp": ts,
                        "genres": movie.genres,
                    }
                )

            profile = {
                "user_id": uid,
                "top_rated_movies": top_movies,
            }

            if args.include_genre_summary:
                profile["genre_summary"] = make_genre_summary(top_movies, normalize=True)

            out_f.write(json.dumps(profile, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] Wrote {written} user profiles to: {args.out}")
    print(f"[OK] Skipped {skipped_too_sparse} users with < {args.min_interactions} valid interactions")


if __name__ == "__main__":
    main()
