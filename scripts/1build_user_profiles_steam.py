#!/usr/bin/env python3
"""
build_steam_user_profiles.py

Download the Kaggle Steam recommendation dataset, construct a sampled positive
interaction subset, and build Steam user profiles.

Dataset
-------
antonkozyriev/game-recommendations-on-steam

Inputs
------
- Kaggle dataset files downloaded into --work_dir:
  - recommendations.csv
  - games.csv
  - games_metadata.json or games_metadata.jsonl

Outputs
-------
- sampled benchmark files under --benchmark_dir
- profiles.jsonl

Example
-------
python scripts/1build_user_profiles_steam.py \
  --work_dir raw/steam \
  --benchmark_dir raw/steam/benchmark_ge10 \
  --out data/steam_smoke/gender/profiles_sample.jsonl \
  --n_users 3000 \
  --rows_per_user 20 \
  --min_user_likes 20 \
  --top_n 20 \
  --min_interactions 10 \
  --random_seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd


DEFAULT_KAGGLE_DATASET = "antonkozyriev/game-recommendations-on-steam"


def ensure_dir(path: str | Path) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def parse_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def parse_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def parse_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


# ---------------------------------------------------------------------
# Kaggle download and sampled benchmark construction
# ---------------------------------------------------------------------

def check_kaggle_auth() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    access_token_txt = Path.home() / ".kaggle" / "access_token.txt"

    has_file = kaggle_json.exists()
    has_access_token_file = access_token.exists() or access_token_txt.exists()
    has_env_old = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    has_env_new = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_env_new_with_username = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_API_TOKEN"))

    if not has_file and not has_access_token_file and not has_env_old and not has_env_new and not has_env_new_with_username:
        raise RuntimeError(
            "Kaggle credentials not found. Set up one of: "
            "~/.kaggle/kaggle.json, ~/.kaggle/access_token, "
            "KAGGLE_USERNAME+KAGGLE_KEY, or KAGGLE_API_TOKEN."
        )

    if os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]


def download_kaggle_dataset(dataset: str, work_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("The 'kaggle' package is not installed. Install it with: pip install kaggle") from e

    check_kaggle_auth()
    ensure_dir(work_dir)

    zip_files = sorted(work_dir.glob("*.zip"))
    if zip_files:
        print(f"[INFO] Existing zip found; skipping download: {zip_files[0]}")
        return

    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Downloading Kaggle dataset: {dataset}")
    print(f"[INFO] Target directory: {work_dir}")

    api.dataset_download_files(
        dataset,
        path=str(work_dir),
        unzip=False,
        quiet=False,
    )


def find_zip_file(work_dir: Path) -> Path:
    zip_files = sorted(work_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {work_dir}. Download may have failed.")
    return zip_files[0]


def extract_zip_if_needed(zip_path: Path, work_dir: Path) -> None:
    rec_path = work_dir / "recommendations.csv"
    if rec_path.exists():
        print("[INFO] recommendations.csv already extracted; skipping unzip.")
        return

    print(f"[INFO] Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(work_dir)
    print(f"[OK] Extracted into: {work_dir}")


def compute_liked_user_counts(recommendations_csv: Path) -> Dict[str, int]:
    user_like_counts: Dict[str, int] = defaultdict(int)

    print(f"[INFO] Counting positive interactions in {recommendations_csv}")
    for chunk in pd.read_csv(
        recommendations_csv,
        usecols=["user_id", "is_recommended"],
        chunksize=100000,
    ):
        chunk = chunk.dropna(subset=["user_id", "is_recommended"])
        for row in chunk.itertuples(index=False):
            if bool(row.is_recommended):
                user_like_counts[str(row.user_id)] += 1

    return dict(user_like_counts)


def sample_users(
    user_like_counts: Dict[str, int],
    n_users: int,
    min_user_likes: int,
    random_seed: int,
) -> List[str]:
    eligible_users = [u for u, count in user_like_counts.items() if count >= min_user_likes]
    if len(eligible_users) < n_users:
        raise RuntimeError(
            f"Not enough eligible users with >= {min_user_likes} liked interactions. "
            f"Need {n_users}, found {len(eligible_users)}."
        )

    rng = random.Random(random_seed)
    sampled_users = rng.sample(eligible_users, n_users)
    return sorted(sampled_users)


def sort_rows_for_user(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["playtime_forever", "votes_up", "votes_funny", "unix_timestamp_created", "appid"]
    existing_cols = [c for c in sort_cols if c in df.columns]
    if not existing_cols:
        return df

    ascending = []
    for col in existing_cols:
        ascending.append(col == "appid")

    return df.sort_values(
        by=existing_cols,
        ascending=ascending,
        kind="mergesort",
    )


def save_filtered_subset(
    recommendations_csv: Path,
    sampled_users: List[str],
    benchmark_dir: Path,
    rows_per_user: int,
) -> Tuple[Path, Path, int, int, Dict[str, int]]:
    ensure_dir(benchmark_dir)

    filtered_csv = benchmark_dir / "steam_liked_ge10.csv"
    user_hist_csv = benchmark_dir / "steam_user_hist_ge10.csv"

    if filtered_csv.exists():
        filtered_csv.unlink()

    sampled_user_set = set(sampled_users)
    kept_counts: Dict[str, int] = defaultdict(int)
    item_set = set()
    n_rows = 0
    first_write = True

    print(f"[INFO] Filtering recommendations.csv for {len(sampled_users)} sampled users")

    usecols = [
        "user_id", "app_id", "helpful", "funny",
        "date", "is_recommended", "hours", "review_id",
    ]

    for chunk in pd.read_csv(recommendations_csv, usecols=usecols, chunksize=100000):
        chunk = chunk.dropna(subset=["user_id", "app_id", "is_recommended"]).copy()
        chunk["user_id"] = chunk["user_id"].astype(str)
        chunk["app_id"] = chunk["app_id"].astype(str)

        sub = chunk[
            (chunk["user_id"].isin(sampled_user_set)) &
            (chunk["is_recommended"].astype(bool))
        ].copy()

        if len(sub) == 0:
            continue

        sub = sub.rename(columns={
            "user_id": "steamid",
            "app_id": "appid",
            "is_recommended": "voted_up",
            "helpful": "votes_up",
            "funny": "votes_funny",
            "hours": "playtime_forever",
            "date": "unix_timestamp_created",
        })

        sub["playtime_at_review"] = sub["playtime_forever"]
        sub["num_games_owned"] = pd.NA
        sub["num_reviews"] = pd.NA
        sub["review"] = ""
        sub["unix_timestamp_updated"] = sub["unix_timestamp_created"]

        sub = sub[
            [
                "steamid", "appid", "voted_up",
                "votes_up", "votes_funny", "playtime_forever",
                "playtime_at_review", "num_games_owned", "num_reviews",
                "review", "unix_timestamp_created", "unix_timestamp_updated",
                "review_id",
            ]
        ]

        kept_parts = []
        for uid, user_df in sub.groupby("steamid", sort=False):
            remaining = rows_per_user - kept_counts[uid]
            if remaining <= 0:
                continue

            user_df = sort_rows_for_user(user_df)
            user_df = user_df.head(remaining)

            if len(user_df) == 0:
                continue

            kept_counts[uid] += len(user_df)
            kept_parts.append(user_df)

        if not kept_parts:
            continue

        out_df = pd.concat(kept_parts, axis=0, ignore_index=True)

        item_set.update(out_df["appid"].astype(str).tolist())
        n_rows += len(out_df)

        out_df.to_csv(
            filtered_csv,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
        )
        first_write = False

        if all(kept_counts[u] >= rows_per_user for u in sampled_users):
            break

    hist_rows = [{"steamid": u, "liked_count": int(kept_counts.get(u, 0))} for u in sampled_users]
    pd.DataFrame(hist_rows).to_csv(user_hist_csv, index=False)

    return filtered_csv, user_hist_csv, n_rows, len(item_set), dict(kept_counts)


def write_subset_stats(
    benchmark_dir: Path,
    n_users_sampled: int,
    n_items_kept: int,
    n_rows_kept: int,
    kept_counts: Dict[str, int],
    min_user_likes: int,
    rows_per_user: int,
) -> None:
    vals = sorted(kept_counts.values())
    avg_hist = sum(vals) / len(vals) if vals else 0.0
    med_hist = vals[len(vals) // 2] if vals else 0.0

    stats = {
        "domain": "SteamReviews",
        "source_dataset": "Game Recommendations on Steam",
        "filter_rule": (
            f"Sample users with >= {min_user_likes} positive recommendations, "
            f"then keep up to {rows_per_user} positive rows per sampled user."
        ),
        "n_users_sampled": int(n_users_sampled),
        "n_items_from_kept_users": int(n_items_kept),
        "n_positive_rows_kept": int(n_rows_kept),
        "avg_liked_history_kept_users": float(avg_hist),
        "median_liked_history_kept_users": float(med_hist),
    }

    stats_json = benchmark_dir / "steam_benchmark_stats.json"
    stats_txt = benchmark_dir / "steam_benchmark_stats.txt"

    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    lines = [
        "Steam Benchmark Subset Statistics",
        "=" * 60,
        f"Domain: {stats['domain']}",
        f"Source dataset: {stats['source_dataset']}",
        f"Filter rule: {stats['filter_rule']}",
        f"Users sampled: {stats['n_users_sampled']}",
        f"Items from kept users: {stats['n_items_from_kept_users']}",
        f"Positive rows kept: {stats['n_positive_rows_kept']}",
        f"Avg liked history: {stats['avg_liked_history_kept_users']:.2f}",
        f"Median liked history: {stats['median_liked_history_kept_users']:.2f}",
        "",
    ]

    with open(stats_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote subset stats: {stats_json}")
    print(f"[OK] Wrote subset stats: {stats_txt}")


def prepare_steam_subset(args: argparse.Namespace) -> Tuple[str, str, str]:
    work_dir = Path(args.work_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else work_dir / "benchmark_ge10"

    download_kaggle_dataset(args.kaggle_dataset, work_dir)

    zip_path = find_zip_file(work_dir)
    extract_zip_if_needed(zip_path, work_dir)

    recommendations_csv = work_dir / "recommendations.csv"
    games_csv = work_dir / "games.csv"
    metadata_path = work_dir / "games_metadata.json"

    if not recommendations_csv.exists():
        raise FileNotFoundError(f"Missing recommendations.csv in work_dir: {work_dir}")
    if not games_csv.exists():
        raise FileNotFoundError(f"Missing games.csv in work_dir: {work_dir}")
    if not metadata_path.exists():
        alt = work_dir / "games_metadata.jsonl"
        if alt.exists():
            metadata_path = alt
        else:
            raise FileNotFoundError(
                f"Missing games_metadata.json or games_metadata.jsonl in work_dir: {work_dir}"
            )

    user_like_counts = compute_liked_user_counts(recommendations_csv)
    sampled_users = sample_users(
        user_like_counts=user_like_counts,
        n_users=args.n_users,
        min_user_likes=args.min_user_likes,
        random_seed=args.random_seed,
    )

    filtered_csv, _user_hist_csv, n_rows, n_items, kept_counts = save_filtered_subset(
        recommendations_csv=recommendations_csv,
        sampled_users=sampled_users,
        benchmark_dir=benchmark_dir,
        rows_per_user=args.rows_per_user,
    )

    write_subset_stats(
        benchmark_dir=benchmark_dir,
        n_users_sampled=len(sampled_users),
        n_items_kept=n_items,
        n_rows_kept=n_rows,
        kept_counts=kept_counts,
        min_user_likes=args.min_user_likes,
        rows_per_user=args.rows_per_user,
    )

    return str(filtered_csv), str(games_csv), str(metadata_path)


# ---------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------

def iter_json_records(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            obj = json.load(f)
            if isinstance(obj, list):
                for row in obj:
                    if isinstance(row, dict):
                        yield row
            return

        if first == "{":
            text = f.read()
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    yield obj
                elif isinstance(obj, list):
                    for row in obj:
                        if isinstance(row, dict):
                            yield row
                return
            except Exception:
                pass

    with open(path, "r", encoding="utf-8") as f2:
        for line in f2:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def load_interactions(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def load_games(games_path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(games_path)

    keep_cols = [
        "app_id",
        "appid",
        "title",
        "date_release",
        "win",
        "mac",
        "linux",
        "rating",
        "positive_ratio",
        "user_reviews",
        "price_final",
        "price_original",
        "discount",
        "steam_deck",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    out: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        app_id = parse_str(row.get("app_id")) or parse_str(row.get("appid"))
        if not app_id:
            continue
        out[app_id] = row
    return out


def load_games_metadata(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for row in iter_json_records(metadata_path):
        app_id = parse_str(row.get("app_id")) or parse_str(row.get("appid"))
        if not app_id:
            continue

        tags = row.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        clean_tags = [parse_str(tag) for tag in tags if parse_str(tag)]
        out[app_id] = {"tags": clean_tags}

    return out


def group_rows_by_user(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        steamid = parse_str(row.get("steamid")) or parse_str(row.get("user_id"))
        if not steamid:
            continue
        by_user[steamid].append(row)
    return by_user


def get_appid(row: Dict[str, Any]) -> str:
    return parse_str(row.get("appid")) or parse_str(row.get("app_id"))


def enrich_rows_with_game_meta(
    rows: List[Dict[str, Any]],
    games_meta: Dict[str, Dict[str, Any]],
    games_metadata: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        appid = get_appid(row)
        meta = games_meta.get(appid, {})
        meta_extra = games_metadata.get(appid, {})

        merged = dict(row)
        merged["appid"] = appid
        merged["title"] = parse_str(meta.get("title")) or parse_str(row.get("title"))
        merged["date_release"] = parse_str(meta.get("date_release"))
        merged["win"] = bool(meta.get("win")) if meta.get("win") is not None else False
        merged["mac"] = bool(meta.get("mac")) if meta.get("mac") is not None else False
        merged["linux"] = bool(meta.get("linux")) if meta.get("linux") is not None else False
        merged["rating"] = parse_str(meta.get("rating"))
        merged["positive_ratio"] = parse_int(meta.get("positive_ratio"), 0)
        merged["user_reviews_meta"] = parse_int(meta.get("user_reviews"), 0)
        merged["price_final"] = parse_float(meta.get("price_final"), 0.0)
        merged["price_original"] = parse_float(meta.get("price_original"), 0.0)
        merged["discount"] = parse_float(meta.get("discount"), 0.0)
        merged["steam_deck"] = bool(meta.get("steam_deck")) if meta.get("steam_deck") is not None else False
        merged["tags"] = meta_extra.get("tags", [])
        enriched.append(merged)

    return enriched


def sort_user_games(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(row: Dict[str, Any]):
        return (
            -parse_int(row.get("playtime_forever"), 0),
            -parse_int(row.get("votes_up"), 0),
            -parse_int(row.get("votes_funny"), 0),
            -parse_int(row.get("unix_timestamp_created"), 0),
            get_appid(row),
        )

    return sorted(rows, key=key_fn)


def build_profile_rows(rows: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    sorted_rows = sort_user_games(rows)
    top_rows = sorted_rows[:top_n]

    games: List[Dict[str, Any]] = []
    for row in top_rows:
        games.append(
            {
                "appid": get_appid(row),
                "title": parse_str(row.get("title")),
                "voted_up": parse_bool(row.get("voted_up")) or parse_bool(row.get("is_recommended")),
                "votes_up": parse_int(row.get("votes_up"), 0),
                "votes_funny": parse_int(row.get("votes_funny"), 0),
                "playtime_forever": parse_int(row.get("playtime_forever"), 0),
                "playtime_at_review": parse_int(row.get("playtime_at_review"), 0),
                "unix_timestamp_created": parse_int(row.get("unix_timestamp_created"), 0),
                "unix_timestamp_updated": parse_int(row.get("unix_timestamp_updated"), 0),
                "date_release": parse_str(row.get("date_release")),
                "rating": parse_str(row.get("rating")),
                "positive_ratio": parse_int(row.get("positive_ratio"), 0),
                "user_reviews_meta": parse_int(row.get("user_reviews_meta"), 0),
                "price_final": parse_float(row.get("price_final"), 0.0),
                "price_original": parse_float(row.get("price_original"), 0.0),
                "discount": parse_float(row.get("discount"), 0.0),
                "win": bool(row.get("win")),
                "mac": bool(row.get("mac")),
                "linux": bool(row.get("linux")),
                "steam_deck": bool(row.get("steam_deck")),
                "tags": row.get("tags", []),
            }
        )

    return games


def build_profiles(
    csv_path: str,
    games_path: str,
    metadata_path: str,
    out_path: str,
    top_n: int,
    min_interactions: int,
) -> None:
    if min_interactions < 1:
        raise ValueError("min_interactions must be >= 1")

    ensure_dir(Path(out_path).parent)

    rows = load_interactions(csv_path)
    games_meta = load_games(games_path)
    games_metadata = load_games_metadata(metadata_path)
    by_user = group_rows_by_user(rows)

    n_input_rows = len(rows)
    n_input_users = len(by_user)
    written = 0
    skipped_too_sparse = 0
    skipped_zero_liked = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for steamid in sorted(by_user.keys()):
            user_rows = by_user[steamid]

            liked_rows = [
                row for row in user_rows
                if parse_bool(row.get("voted_up")) or parse_bool(row.get("is_recommended"))
            ]

            if len(liked_rows) == 0:
                skipped_zero_liked += 1
                continue

            if len(liked_rows) < min_interactions:
                skipped_too_sparse += 1
                continue

            liked_rows = enrich_rows_with_game_meta(liked_rows, games_meta, games_metadata)
            top_games = build_profile_rows(liked_rows, top_n)

            profile = {
                "user_id": steamid,
                "top_liked_games": top_games,
                "profile_metadata": {
                    "n_rows_for_user_in_sample": len(liked_rows),
                    "min_interactions_applied_on_sample": int(min_interactions),
                    "top_n_kept": min(len(top_games), top_n),
                },
            }

            out_f.write(json.dumps(profile, ensure_ascii=False) + "\n")
            written += 1

    print(f"[OK] Input sampled rows: {n_input_rows}")
    print(f"[OK] Input users in sampled CSV: {n_input_users}")
    print(f"[OK] Wrote {written} user profiles to: {out_path}")
    print(f"[OK] Skipped {skipped_zero_liked} users with 0 liked rows in sampled CSV")
    print(f"[OK] Skipped {skipped_too_sparse} users with < {min_interactions} liked interactions in sampled CSV")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True, help="Directory containing or receiving raw Steam files")
    ap.add_argument("--benchmark_dir", default="", help="Directory for sampled benchmark CSV files")
    ap.add_argument("--kaggle_dataset", default=DEFAULT_KAGGLE_DATASET)

    ap.add_argument("--out", required=True, help="Output profiles.jsonl")

    ap.add_argument("--n_users", type=int, default=1000)
    ap.add_argument("--rows_per_user", type=int, default=20)
    ap.add_argument("--min_user_likes", type=int, default=20)
    ap.add_argument("--random_seed", type=int, default=42)

    ap.add_argument("--top_n", type=int, default=20, help="Max number of liked games kept per user")
    ap.add_argument("--min_interactions", type=int, default=10, help="Minimum liked interactions in sampled CSV")

    args = ap.parse_args()

    csv_path, games_path, metadata_path = prepare_steam_subset(args)

    build_profiles(
        csv_path=csv_path,
        games_path=games_path,
        metadata_path=metadata_path,
        out_path=args.out,
        top_n=args.top_n,
        min_interactions=args.min_interactions,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
