#!/usr/bin/env python3
"""
build_goodreads_user_profiles.py

Build per-user Goodreads book profiles for FairGap.

This script processes locally downloaded Goodreads interaction and metadata
files. The Goodreads data can be obtained from the UCSD Book Graph / Goodreads
dataset page:

https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

Inputs
------
- Goodreads interactions file:
  - goodreads_interactions.csv, or
  - goodreads_interactions_{genre}.json.gz

- Goodreads book metadata file:
  - goodreads_books.json.gz, or
  - goodreads_books_{genre}.json.gz

Output
------
- profiles_books.jsonl

Each JSONL row contains:
- user_id
- top_rated_books: list of {book_id, title, rating, genres}
- genre_summary, if requested

Construction rule
-----------------
- Keep users with at least min_ratings explicit ratings.
- Valid ratings are integers in {1, 2, 3, 4, 5}.
- Retain up to top_n books per user using deterministic rating-priority order.
- Genres are approximated using top popular_shelves tags from Goodreads metadata.

Example
-------
python scripts/1build_user_profiles_book.py \
  --interactions raw/goodreads/goodreads_interactions.csv \
  --books_meta raw/goodreads/goodreads_books.json.gz \
  --out data/goodreads_smoke/gender/profiles_sample.jsonl \
  --top_n 20 \
  --min_ratings 10 \
  --max_users 1000 \
  --include_genre_summary
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Tuple


GOODREADS_SOURCE_URL = "https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html"
MAX_SHELF_GENRES_DEFAULT = 5
MIN_RATINGS_DEFAULT = 10
MAX_USERS_DEFAULT = 10000
SAMPLE_SEED_DEFAULT = 42


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def open_text_maybe_gzip(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def check_file_exists(path: str, label: str) -> None:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {label}: {path}\n"
            f"Please download the Goodreads files from:\n"
            f"{GOODREADS_SOURCE_URL}"
        )


def load_book_metadata(meta_path: str, max_shelf_genres: int) -> Dict[str, Dict[str, Any]]:
    """
    Load Goodreads book metadata from gzip-compressed NDJSON or plain NDJSON.

    Returns:
      books[book_id] = {"title": str, "genres": list[str]}

    Genres are derived from the top popular_shelves names by count.
    """
    print(f"[INFO] Loading book metadata from: {meta_path}")
    books: Dict[str, Dict[str, Any]] = {}

    with open_text_maybe_gzip(meta_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            book_id = str(obj.get("book_id", "")).strip()
            if not book_id:
                continue

            title = str(obj.get("title_without_series") or obj.get("title") or "").strip()

            shelves = obj.get("popular_shelves", []) or []
            shelf_counts: List[Tuple[str, int]] = []

            if isinstance(shelves, list):
                for shelf in shelves:
                    if not isinstance(shelf, dict):
                        continue
                    name = str(shelf.get("name", "")).strip()
                    try:
                        count = int(shelf.get("count", 0))
                    except (ValueError, TypeError):
                        count = 0
                    if name and count > 0:
                        shelf_counts.append((name, count))

            shelf_counts.sort(key=lambda x: -x[1])
            genres = [name for name, _ in shelf_counts[:max_shelf_genres]]

            books[book_id] = {
                "title": title,
                "genres": genres,
            }

            if (i + 1) % 100000 == 0:
                print(f"  ... {i + 1:,} metadata rows processed")

    print(f"[INFO] Loaded metadata for {len(books):,} books.")
    return books


def _parse_csv_line(line: str, ratings_by_user: Dict[str, List[Tuple[str, int]]]) -> bool:
    """
    Parse one CSV data line.

    Expected full Goodreads CSV format:
      user_id,book_id,is_read,rating,is_reviewed
    """
    try:
        parts = line.split(",")
        uid = parts[0].strip()
        bid = parts[1].strip()
        rating = int(parts[3])
        if uid and bid and 1 <= rating <= 5:
            ratings_by_user[uid].append((bid, rating))
            return True
    except (ValueError, IndexError):
        pass
    return False


def _parse_json_line(line: str, ratings_by_user: Dict[str, List[Tuple[str, int]]]) -> bool:
    """
    Parse one Goodreads NDJSON interaction row.
    """
    try:
        obj = json.loads(line)
        uid = str(obj.get("user_id", "")).strip()
        bid = str(obj.get("book_id", "")).strip()
        rating = int(obj.get("rating", 0))
        if uid and bid and 1 <= rating <= 5:
            ratings_by_user[uid].append((bid, rating))
            return True
    except (ValueError, KeyError, json.JSONDecodeError, TypeError):
        pass
    return False


def load_interactions(interactions_path: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Load valid historical Goodreads interactions.

    Supported inputs:
    - .csv
    - .json.gz / .jsonl.gz / .jsonl / NDJSON

    Valid interaction:
    - explicit rating in {1,2,3,4,5}
    """
    print(f"[INFO] Loading interactions from: {interactions_path}")
    ratings_by_user: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    n_rows = 0
    n_kept = 0

    with open_text_maybe_gzip(interactions_path) as f:
        first_line = f.readline().strip()
        is_json = first_line.startswith("{")

        if not is_json:
            has_header = first_line.lower().startswith("user_id")
            if not has_header:
                n_rows += 1
                if _parse_csv_line(first_line, ratings_by_user):
                    n_kept += 1

            reader = csv.reader(f)
            for row in reader:
                n_rows += 1
                if len(row) < 4:
                    continue
                try:
                    uid = row[0].strip()
                    bid = row[1].strip()
                    rating = int(row[3])
                except (ValueError, IndexError):
                    continue

                if uid and bid and 1 <= rating <= 5:
                    ratings_by_user[uid].append((bid, rating))
                    n_kept += 1

                if n_rows % 5000000 == 0:
                    print(f"  ... {n_rows:,} rows read, {n_kept:,} kept")

        else:
            n_rows += 1
            if _parse_json_line(first_line, ratings_by_user):
                n_kept += 1

            for line in f:
                line = line.strip()
                if not line:
                    continue

                n_rows += 1
                if _parse_json_line(line, ratings_by_user):
                    n_kept += 1

                if n_rows % 1000000 == 0:
                    print(f"  ... {n_rows:,} rows read, {n_kept:,} kept")

    print(f"[INFO] Read {n_rows:,} interaction rows.")
    print(f"[INFO] Valid interactions kept: {n_kept:,}")
    print(f"[INFO] Users with >=1 valid interaction: {len(ratings_by_user):,}")
    return dict(ratings_by_user)


def select_top_n_books(ratings: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
    """
    Select up to top-N books using deterministic rating-priority order:
    - rating descending from 5 to 1
    - stable within-bucket order
    """
    buckets: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for book_id, rating in ratings:
        buckets[rating].append((book_id, rating))

    selected: List[Tuple[str, int]] = []
    for rating in [5, 4, 3, 2, 1]:
        selected.extend(buckets.get(rating, []))
        if len(selected) >= n:
            break

    return selected[:n]


def make_genre_summary(top_books: List[Dict[str, Any]], normalize: bool = True) -> Dict[str, float]:
    counter: Counter = Counter()
    for book in top_books:
        for genre in book.get("genres", []):
            if genre:
                counter[genre] += 1

    if not counter:
        return {}

    if not normalize:
        return dict(counter)

    total = sum(counter.values())
    return {genre: round(count / total, 6) for genre, count in counter.most_common()}


def build_profiles(
    interactions_path: str,
    books_meta_path: str,
    out_path: str,
    top_n: int,
    min_ratings: int,
    max_users: int,
    include_genre_summary: bool,
    max_shelf_genres: int,
    sample_seed: int,
) -> None:
    check_file_exists(interactions_path, "Goodreads interactions file")
    check_file_exists(books_meta_path, "Goodreads book metadata file")
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))

    books_meta = load_book_metadata(books_meta_path, max_shelf_genres=max_shelf_genres)
    ratings_by_user = load_interactions(interactions_path)

    qualifying_users = sorted(
        uid for uid, ratings in ratings_by_user.items()
        if len(ratings) >= min_ratings
    )
    print(f"[INFO] Users with >= {min_ratings} valid interactions: {len(qualifying_users):,}")

    if max_users > 0 and len(qualifying_users) > max_users:
        rng = random.Random(sample_seed)
        qualifying_users = rng.sample(qualifying_users, max_users)
        qualifying_users.sort()
        print(f"[INFO] Sampled {len(qualifying_users):,} users with seed={sample_seed}.")

    written = 0
    n_no_meta = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for uid in qualifying_users:
            user_ratings = ratings_by_user[uid]
            top_raw = select_top_n_books(user_ratings, top_n)

            top_books: List[Dict[str, Any]] = []
            for book_id, rating in top_raw:
                meta = books_meta.get(book_id)
                if meta is None:
                    n_no_meta += 1
                    top_books.append(
                        {
                            "book_id": book_id,
                            "title": f"[Book {book_id}]",
                            "rating": rating,
                            "genres": [],
                        }
                    )
                else:
                    top_books.append(
                        {
                            "book_id": book_id,
                            "title": meta["title"] or f"[Book {book_id}]",
                            "rating": rating,
                            "genres": meta["genres"],
                        }
                    )

            profile: Dict[str, Any] = {
                "user_id": uid,
                "top_rated_books": top_books,
            }

            if include_genre_summary:
                profile["genre_summary"] = make_genre_summary(top_books, normalize=True)

            out_f.write(json.dumps(profile, ensure_ascii=False) + "\n")
            written += 1

    print(f"[INFO] Metadata misses: {n_no_meta:,}")
    print(f"[OK] Wrote {written:,} user profiles to: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-user Goodreads book profiles for FairGap."
    )
    ap.add_argument(
        "--interactions",
        required=True,
        help="Path to goodreads_interactions.csv or goodreads_interactions_{genre}.json.gz.",
    )
    ap.add_argument(
        "--books_meta",
        required=True,
        help="Path to goodreads_books.json.gz or goodreads_books_{genre}.json.gz.",
    )
    ap.add_argument("--out", required=True, help="Output profiles_books.jsonl path.")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--min_ratings", type=int, default=MIN_RATINGS_DEFAULT)
    ap.add_argument(
        "--max_users",
        type=int,
        default=MAX_USERS_DEFAULT,
        help="Max number of users to output; 0 means all qualifying users.",
    )
    ap.add_argument("--include_genre_summary", action="store_true")
    ap.add_argument("--max_shelf_genres", type=int, default=MAX_SHELF_GENRES_DEFAULT)
    ap.add_argument("--sample_seed", type=int, default=SAMPLE_SEED_DEFAULT)
    args = ap.parse_args()

    build_profiles(
        interactions_path=args.interactions,
        books_meta_path=args.books_meta,
        out_path=args.out,
        top_n=args.top_n,
        min_ratings=args.min_ratings,
        max_users=args.max_users,
        include_genre_summary=args.include_genre_summary,
        max_shelf_genres=args.max_shelf_genres,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
