#!/usr/bin/env python3
"""
compute_layerwise_internal_shift.py

Compute layerwise internal shift for matched counterfactual prompt pairs.

Inputs
------
- internal_vectors.npz
- split.jsonl

Outputs
-------
- internal_distance_layers.dev.jsonl
- internal_distance_layers.eval.jsonl

Behavior
--------
- Reads quartile-based hidden vectors from NPZ.
- Reads user-level dev/eval split assignments.
- Matches the two counterfactual variants by user_id.
- Computes layerwise cosine distance:
      delta = 1 - cosine_similarity
- Writes one row per matched user pair into the corresponding split output.

Expected NPZ fields
-------------------
- user_id: [N], user identifiers
- variant: [N], binary variant codes 0/1
- vectors: [N, 4, H], sampled layer vectors
- relative_layers: [4]
- layer_indices: [4]
- variant_order: optional [2], maps variant code to variant name

Example
-------
python scripts/8compute_layerwise_internal_shift.py \
  --npz data/movielens_smoke/gender/internal_vectors_sample.npz \
  --split data/movielens_smoke/gender/split_sample.jsonl \
  --out_dev data/movielens_smoke/gender/internal_distance_layers_sample.dev.jsonl \
  --out_eval data/movielens_smoke/gender/internal_distance_layers_sample.eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np


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


def load_split_map(path: str) -> Dict[str, str]:
    split_map: Dict[str, str] = {}

    for row in iter_jsonl(path):
        if "user_id" not in row or "split" not in row:
            continue

        uid = str(row["user_id"]).strip()
        split = str(row["split"]).strip().lower()

        if not uid or split not in ("dev", "eval"):
            continue

        split_map[uid] = split

    return split_map


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < eps and norm_b < eps:
        return 0.0
    if norm_a < eps or norm_b < eps:
        return 1.0

    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def decode_variant_order(ex: Any) -> List[str]:
    """
    If variant_order exists in the NPZ, use it.
    Otherwise fall back to generic names ["a", "b"].

    Older gender-only NPZ files encoded 0=female and 1=male. For artifact
    generality, we avoid assuming protected-attribute labels unless the NPZ
    explicitly stores variant_order.
    """
    if "variant_order" in ex.files:
        vals = ex["variant_order"].tolist()
        variant_order = [str(x).strip().lower() for x in vals]
        variant_order = [x for x in variant_order if x]
        if len(variant_order) >= 2:
            return variant_order[:2]

    return ["a", "b"]


def load_internal_vectors(
    npz_path: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[float], List[int], List[str]]:
    """
    Returns:
      vectors_by_user[user_id][variant_name] = np.ndarray [4, H]
      relative_layers: list[float]
      layer_indices: list[int]
      variant_order: list[str]
    """
    ex = np.load(npz_path, allow_pickle=True)

    required = ["user_id", "variant", "vectors", "relative_layers", "layer_indices"]
    missing = [key for key in required if key not in ex.files]
    if missing:
        raise ValueError(f"NPZ is missing required fields: {missing}")

    user_ids = ex["user_id"]
    variants = ex["variant"]
    vectors = ex["vectors"]
    relative_layers = ex["relative_layers"].tolist()
    layer_indices = ex["layer_indices"].tolist()
    variant_order = decode_variant_order(ex)

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

    if len(user_ids) != len(variants) or len(user_ids) != vectors.shape[0]:
        raise ValueError(
            "Mismatch among user_id, variant, and vectors lengths: "
            f"len(user_id)={len(user_ids)}, len(variant)={len(variants)}, "
            f"vectors.shape[0]={vectors.shape[0]}"
        )

    vectors_by_user: Dict[str, Dict[str, np.ndarray]] = {}

    for i in range(len(user_ids)):
        uid = str(user_ids[i]).strip()
        if not uid:
            continue

        variant_code = int(variants[i])
        if variant_code < 0 or variant_code >= len(variant_order):
            continue

        variant_name = variant_order[variant_code]
        vectors_by_user.setdefault(uid, {})
        vectors_by_user[uid][variant_name] = np.asarray(vectors[i], dtype=np.float32)

    return vectors_by_user, relative_layers, layer_indices, variant_order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Input internal_vectors.npz")
    ap.add_argument("--split", required=True, help="Input split.jsonl")
    ap.add_argument("--out_dev", required=True, help="Output dev JSONL")
    ap.add_argument("--out_eval", required=True, help="Output eval JSONL")
    args = ap.parse_args()

    if not os.path.isfile(args.npz):
        raise FileNotFoundError(f"Input NPZ file not found: {args.npz}")
    if not os.path.isfile(args.split):
        raise FileNotFoundError(f"Input split file not found: {args.split}")

    ensure_dir(os.path.dirname(args.out_dev))
    ensure_dir(os.path.dirname(args.out_eval))

    split_map = load_split_map(args.split)
    vectors_by_user, relative_layers, layer_indices, variant_order = load_internal_vectors(args.npz)

    if len(variant_order) != 2:
        raise ValueError(f"Expected exactly two variants, got: {variant_order}")

    variant_a, variant_b = variant_order[0], variant_order[1]

    n_users_seen = 0
    n_matched = 0
    n_dev = 0
    n_eval = 0
    n_missing_split = 0
    n_missing_pair = 0
    n_bad_shape = 0

    with open(args.out_dev, "w", encoding="utf-8") as f_dev, open(args.out_eval, "w", encoding="utf-8") as f_eval:
        for uid in sorted(vectors_by_user.keys()):
            n_users_seen += 1

            if uid not in split_map:
                n_missing_split += 1
                continue

            variants = vectors_by_user[uid]
            if variant_a not in variants or variant_b not in variants:
                n_missing_pair += 1
                continue

            vec_a = variants[variant_a]
            vec_b = variants[variant_b]

            if vec_a.shape != vec_b.shape or vec_a.ndim != 2:
                n_bad_shape += 1
                continue

            deltas: List[float] = []
            for layer_pos in range(vec_a.shape[0]):
                delta = cosine_distance(vec_a[layer_pos], vec_b[layer_pos])
                deltas.append(float(delta))

            if len(deltas) != 4:
                n_bad_shape += 1
                continue

            split = split_map[uid]
            rec = {
                "user_id": uid,
                "split": split,
                "variant_a": variant_a,
                "variant_b": variant_b,
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

            if split == "dev":
                f_dev.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_dev += 1
            else:
                f_eval.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_eval += 1

            n_matched += 1

    print(f"[OK] variant_order={variant_order}")
    print(f"[OK] users_seen={n_users_seen}")
    print(f"[OK] matched_pairs={n_matched}")
    print(f"[OK] missing_split={n_missing_split}")
    print(f"[OK] missing_pair={n_missing_pair}")
    print(f"[OK] bad_shape={n_bad_shape}")
    print(f"[OK] wrote dev rows={n_dev} -> {args.out_dev}")
    print(f"[OK] wrote eval rows={n_eval} -> {args.out_eval}")


if __name__ == "__main__":
    main()
