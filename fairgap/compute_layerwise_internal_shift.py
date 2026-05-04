
import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np


NPZ_DEFAULT = "examples/toy_out/internal_vectors.npz"
SPLIT_DEFAULT = "examples/toy_out/split.jsonl"

OUT_DEV_DEFAULT = "examples/toy_out/internal_distance_layers.dev.jsonl"
OUT_EVAL_DEFAULT = "examples/toy_out/internal_distance_layers.eval.jsonl"


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


def load_split_map(path: str) -> Dict[int, str]:
    split_map: Dict[int, str] = {}
    for row in iter_jsonl(path):
        if "user_id" not in row or "split" not in row:
            continue
        try:
            uid = int(row["user_id"])
        except Exception:
            continue
        split = str(row["split"]).strip().lower()
        if split not in ("dev", "eval"):
            continue
        split_map[uid] = split
    return split_map


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < eps and nb < eps:
        return 0.0
    if na < eps or nb < eps:
        return 1.0

    cos_sim = float(np.dot(a, b) / (na * nb))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def load_internal_vectors(npz_path: str) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[float], List[int]]:
    """
    Returns:
      vectors_by_user[user_id]["age_a"] = np.ndarray [4, H]
      vectors_by_user[user_id]["age_b"] = np.ndarray [4, H]
      relative_layers: list[float]
      layer_indices: list[int]
    """
    ex = np.load(npz_path, allow_pickle=True)

    user_ids = ex["user_id"]
    variants = ex["variant"]      # [N], 0=age_a, 1=age_b
    vectors = ex["vectors"]       # [N, 4, H]
    relative_layers = ex["relative_layers"].tolist()
    layer_indices = ex["layer_indices"].tolist()

    if vectors.ndim != 3:
        raise ValueError(f"Expected vectors shape [N, 4, H], got {vectors.shape}")

    if vectors.shape[1] != len(relative_layers):
        raise ValueError(
            f"Mismatch: vectors.shape[1]={vectors.shape[1]} "
            f"but len(relative_layers)={len(relative_layers)}"
        )

    if vectors.shape[1] != len(layer_indices):
        raise ValueError(
            f"Mismatch: vectors.shape[1]={vectors.shape[1]} "
            f"but len(layer_indices)={len(layer_indices)}"
        )

    vectors_by_user: Dict[int, Dict[str, np.ndarray]] = {}
    for i in range(len(user_ids)):
        uid = int(user_ids[i])
        variant_code = int(variants[i])
        if variant_code == 0:
            variant = "age_a"
        elif variant_code == 1:
            variant = "age_b"
        else:
            continue

        vectors_by_user.setdefault(uid, {})
        vectors_by_user[uid][variant] = np.asarray(vectors[i], dtype=np.float32)

    return vectors_by_user, relative_layers, layer_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ_DEFAULT, help="Input internal_vectors.npz")
    ap.add_argument("--split", default=SPLIT_DEFAULT, help="Input split.jsonl")
    ap.add_argument("--out_dev", default=OUT_DEV_DEFAULT, help="Output dev jsonl")
    ap.add_argument("--out_eval", default=OUT_EVAL_DEFAULT, help="Output eval jsonl")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_dev))
    ensure_dir(os.path.dirname(args.out_eval))

    split_map = load_split_map(args.split)
    vectors_by_user, relative_layers, layer_indices = load_internal_vectors(args.npz)

    n_users_seen = 0
    n_matched = 0
    n_dev = 0
    n_eval = 0
    n_missing_split = 0
    n_missing_pair = 0

    with open(args.out_dev, "w", encoding="utf-8") as f_dev, open(args.out_eval, "w", encoding="utf-8") as f_eval:
        for uid in sorted(vectors_by_user.keys()):
            n_users_seen += 1

            if uid not in split_map:
                n_missing_split += 1
                continue

            variants = vectors_by_user[uid]
            if "age_a" not in variants or "age_b" not in variants:
                n_missing_pair += 1
                continue

            vec_a = variants["age_a"]
            vec_b = variants["age_b"]

            if vec_a.shape != vec_b.shape:
                n_missing_pair += 1
                continue

            if vec_a.ndim != 2:
                n_missing_pair += 1
                continue

            deltas: List[float] = []
            for j in range(vec_a.shape[0]):
                delta = cosine_distance(vec_a[j], vec_b[j])
                deltas.append(float(delta))

            rec = {
                "user_id": uid,
                "split": split_map[uid],
                "relative_layers": relative_layers,
                "layer_indices": layer_indices,
                "delta_q1": deltas[0],
                "delta_q2": deltas[1],
                "delta_q3": deltas[2],
                "delta_q4": deltas[3],
                "delta_by_quartile": {
                    "q1": deltas[0],
                    "q2": deltas[1],
                    "q3": deltas[2],
                    "q4": deltas[3],
                },
            }

            if split_map[uid] == "dev":
                f_dev.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_dev += 1
            else:
                f_eval.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_eval += 1

            n_matched += 1

    print(f"[OK] users_seen={n_users_seen}")
    print(f"[OK] matched_pairs={n_matched}")
    print(f"[OK] missing_split={n_missing_split}")
    print(f"[OK] missing_pair={n_missing_pair}")
    print(f"[OK] wrote dev rows={n_dev} -> {args.out_dev}")
    print(f"[OK] wrote eval rows={n_eval} -> {args.out_eval}")


if __name__ == "__main__":
    main()

