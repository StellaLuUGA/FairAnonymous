#!/usr/bin/env python3
from __future__ import annotations

"""
Download MovieLens-1M and build movie-only user profiles.

This script is a lightweight preprocessing step for movie recommendation
experiments. It automatically downloads the standard MovieLens-1M dataset,
extracts the required files, and exports attribute-neutral movie-history
profiles.

Output
------
- profiles.jsonl

Each JSONL row contains:
- user_id
- top_rated_movies: list of {movie_id, title, rating, timestamp, genres}
- genre_summary, if requested

Important design choice
-----------------------
Although users.dat contains gender, age, occupation, and zip code, this script
only uses users.dat to enumerate valid user IDs. Demographic fields are not
exported into profiles.jsonl. Downstream counterfactual prompts can inject
protected-attribute cues separately, so the movie-history profile remains
attribute-neutral.

Construction rule
-----------------
- Keep only users with at least --min_interactions valid historical ratings.
- A valid interaction means the rated movie exists in movies.dat.
- Retain up to --top_n items per user using deterministic ranking:
  rating descending, timestamp descending.

Example
-------
python scripts/1build_user_profiles.py \
  --data_dir data/raw \
  --out data/movielens/gender/profiles.jsonl \
  --top_n 20 \
  --min_interactions 10 \
  --include_genre_summary
"""

import argparse
import json
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: List[str]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_movielens_1m(data_dir: Path, force: bool = False) -> Path:
    """
    Download and extract MovieLens-1M.

    Parameters
    ----------
    data_dir:
        Directory where the zip file and extracted ml-1m folder will be stored.
    force:
        If True, re-download and re-extract the dataset.

    Returns
    -------
    Path to the extracted ml-1m directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "ml-1m.zip"
    extract_dir = data_dir / "ml-1m"

    required_files = [
        extract_dir / "movies.dat",
        extract_dir / "ratings.dat",
        extract_dir / "users.dat",
    ]

    if extract_dir.exists() and all(p.exists() for p in required_files) and not force:
        print(f"[OK] MovieLens-1M already exists: {extract_dir}")
        return extract_dir

    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)

    if force and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        print(f"[INFO] Downloading MovieLens-1M from: {MOVIELENS_1M_URL}")
        urllib.request.urlretrieve(MOVIELENS_1M_URL, zip_path)
        print(f"[OK] Downloaded to: {zip_path}")

    print(f"[INFO] Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "MovieLens-1M extraction completed, but required files are missing:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    print(f"[OK] Extracted MovieLens-1M to: {extract_dir}")
    return extract_dir


def parse_movies(movies_path: Path) -> Dict[int, Movie]:
    movies: Dict[int, Movie] = {}

    with movies_path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("::")
            if len(parts) != 3:
                continue

            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2].split("|") if parts[2] else []

            movies[movie_id] = Movie(
                movie_id=movie_id,
                title=title,
                genres=genres,
            )

    return movies


def parse_users(users_path: Path) -> List[int]:
    """
    users.dat format:
      UserID::Gender::Age::Occupation::Zip-code

    This function intentionally keeps only user IDs. Demographic fields are
    not exported into the movie-history profiles.
    """
    user_ids: List[int] = []

    with users_path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("::")
            if len(parts) != 5:
                continue

            user_ids.append(int(parts[0]))

    return sorted(set(user_ids))


def parse_ratings(ratings_path: Path) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    ratings.dat format:
      UserID::MovieID::Rating::Timestamp

    Returns:
      ratings_by_user[user_id] = list of (movie_id, rating, timestamp)
    """
    ratings_by_user: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

    with ratings_path.open("r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("::")
            if len(parts) != 4:
                continue

            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = int(parts[2])
            timestamp = int(parts[3])

            ratings_by_user[user_id].append((movie_id, rating, timestamp))

    return ratings_by_user


def select_top_n(
    ratings: List[Tuple[int, int, int]],
    n: int,
) -> List[Tuple[int, int, int]]:
    """
    Select Top-N movies with a deterministic rule:
    1. rating descending;
    2. timestamp descending within the same rating.
    """
    return sorted(
        ratings,
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )[:n]


def make_genre_summary(
    top_movies: List[Dict],
    normalize: bool = True,
) -> Dict[str, float]:
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

    return {
        genre: round(count / total, 6)
        for genre, count in counter.most_common()
    }


def build_profiles(
    ml1m_dir: Path,
    out_path: Path,
    top_n: int,
    min_interactions: int,
    include_genre_summary: bool,
) -> None:
    movies_path = ml1m_dir / "movies.dat"
    ratings_path = ml1m_dir / "ratings.dat"
    users_path = ml1m_dir / "users.dat"

    required_files = [movies_path, ratings_path, users_path]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required MovieLens-1M file(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    ensure_parent_dir(out_path)

    movies = parse_movies(movies_path)
    user_ids = parse_users(users_path)
    ratings_by_user = parse_ratings(ratings_path)

    written = 0
    skipped_too_sparse = 0
    skipped_no_valid_ratings = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for user_id in user_ids:
            user_ratings = ratings_by_user.get(user_id, [])

            valid_ratings = [
                (movie_id, rating, timestamp)
                for movie_id, rating, timestamp in user_ratings
                if movie_id in movies
            ]

            if not valid_ratings:
                skipped_no_valid_ratings += 1
                continue

            if len(valid_ratings) < min_interactions:
                skipped_too_sparse += 1
                continue

            top_raw = select_top_n(valid_ratings, top_n)

            top_movies = []
            for movie_id, rating, timestamp in top_raw:
                movie = movies[movie_id]
                top_movies.append(
                    {
                        "movie_id": movie.movie_id,
                        "title": movie.title,
                        "rating": rating,
                        "timestamp": timestamp,
                        "genres": movie.genres,
                    }
                )

            profile = {
                "user_id": user_id,
                "top_rated_movies": top_movies,
            }

            if include_genre_summary:
                profile["genre_summary"] = make_genre_summary(
                    top_movies,
                    normalize=True,
                )

            out_f.write(json.dumps(profile, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] Input directory: {ml1m_dir}")
    print(f"[OK] Wrote {written} user profiles to: {out_path}")
    print(
        f"[OK] Skipped {skipped_too_sparse} users with "
        f"< {min_interactions} valid interactions"
    )
    print(f"[OK] Skipped {skipped_no_valid_ratings} users with no valid ratings")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download MovieLens-1M and build movie-only user profiles."
        )
    )

    parser.add_argument(
        "--data_dir",
        default="data/raw",
        help="Directory for downloaded and extracted MovieLens-1M files.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download and re-extract MovieLens-1M even if files already exist.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output profiles JSONL.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Top-N movies per user after filtering.",
    )
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=10,
        help="Minimum number of valid interactions required.",
    )
    parser.add_argument(
        "--include_genre_summary",
        action="store_true",
        help="Add normalized genre distribution to each user profile.",
    )

    args = parser.parse_args()

    ml1m_dir = download_movielens_1m(
        data_dir=Path(args.data_dir),
        force=args.force_download,
    )

    build_profiles(
        ml1m_dir=ml1m_dir,
        out_path=Path(args.out),
        top_n=args.top_n,
        min_interactions=args.min_interactions,
        include_genre_summary=args.include_genre_summary,
    )


if __name__ == "__main__":
    main()