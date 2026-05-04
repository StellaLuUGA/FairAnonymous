import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict


NPZ_DEFAULT = "examples/toy_out/internal_vectors.npz"
SPLIT_DEFAULT = "examples/toy_out/split.jsonl"
INTERNAL_LAYERS_EVAL_DEFAULT = "examples/toy_out/internal_distance_layers.eval.jsonl"

OUT_WEIGHTS_DEFAULT = "examples/toy_out/probe_weights.json"
OUT_INTERNAL_EVAL_DEFAULT = "examples/toy_out/internal_distance.eval.jsonl"


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl_strict(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {lineno} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object at line {lineno} in {path}, got {type(obj).__name__}"
                )
            yield obj


def load_split_map(path: str) -> Dict[int, str]:
    split_map: Dict[int, str] = {}
    for row in iter_jsonl_strict(path):
        if "user_id" not in row or "split" not in row:
            raise ValueError(f"Malformed split row: missing user_id or split: {row}")

        try:
            uid = int(row["user_id"])
        except Exception as e:
            raise ValueError(f"Invalid user_id in split row: {row}") from e

        split = str(row["split"]).strip().lower()
        if split not in ("dev", "eval"):
            raise ValueError(f"Invalid split value for user_id={uid}: {split}")

        if uid in split_map and split_map[uid] != split:
            raise ValueError(
                f"Conflicting split assignments for user_id={uid}: "
                f"{split_map[uid]} vs {split}"
            )

        split_map[uid] = split

    if not split_map:
        raise RuntimeError(f"No valid split rows loaded from: {path}")

    return split_map


def load_internal_vectors(
    npz_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[int]]:
    """
    Returns:
      user_ids: [N]
      variants: [N] where age_a=0, age_b=1
      vectors: [N, 4, H]
      relative_layers: list[float]
      layer_indices: list[int]
    """
    ex = np.load(npz_path, allow_pickle=False)

    user_ids = np.asarray(ex["user_id"], dtype=np.int64)
    variants = np.asarray(ex["variant"], dtype=np.int64)
    vectors = np.asarray(ex["vectors"], dtype=np.float32)
    relative_layers = ex["relative_layers"].tolist()
    layer_indices = ex["layer_indices"].tolist()

    if vectors.ndim != 3:
        raise ValueError(f"Expected vectors shape [N, 4, H], got {vectors.shape}")
    if vectors.shape[1] != 4:
        raise ValueError(
            f"Expected exactly 4 sampled layers, got vectors.shape[1]={vectors.shape[1]}"
        )
    if len(relative_layers) != 4 or len(layer_indices) != 4:
        raise ValueError("Expected exactly 4 relative_layers and 4 layer_indices")
    if len(user_ids) != len(variants) or len(user_ids) != vectors.shape[0]:
        raise ValueError(
            "Mismatch among user_ids, variants, and vectors first dimension: "
            f"len(user_ids)={len(user_ids)}, len(variants)={len(variants)}, "
            f"vectors.shape[0]={vectors.shape[0]}"
        )

    unique_variants = set(int(x) for x in np.unique(variants))
    if not unique_variants.issubset({0, 1}):
        raise ValueError(f"Variant codes must be in {{0,1}}, got: {sorted(unique_variants)}")

    return user_ids, variants, vectors, relative_layers, layer_indices


def build_dev_probe_data(
    user_ids: np.ndarray,
    variants: np.ndarray,
    vectors: np.ndarray,
    split_map: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter NPZ rows to dev split.

    Returns:
      dev_variants: [N_dev_rows]
      dev_vectors: [N_dev_rows, 4, H]
      dev_user_ids: [N_dev_rows]
    """
    keep_idx: List[int] = []
    for i, uid in enumerate(user_ids):
        if split_map.get(int(uid)) == "dev":
            keep_idx.append(i)

    if not keep_idx:
        raise RuntimeError("No dev rows found in NPZ after applying split map")

    dev_variants = np.asarray(variants[keep_idx], dtype=np.int64)
    dev_vectors = np.asarray(vectors[keep_idx], dtype=np.float32)
    dev_user_ids = np.asarray(user_ids[keep_idx], dtype=np.int64)

    return dev_variants, dev_vectors, dev_user_ids


def fit_probe_auc_per_layer_grouped_cv(
    dev_variants: np.ndarray,
    dev_vectors: np.ndarray,
    dev_user_ids: np.ndarray,
    seed: int = 1234,
    max_iter: int = 2000,
    C: float = 1.0,
    n_splits: int = 5,
) -> List[float]:
    """
    For each layer j in {0,1,2,3}, fit a logistic regression probe on dev rows
    and estimate separability with grouped out-of-fold AUC.

    Grouping:
      user_id is used as the group key so all rows from the same user stay
      in the same fold.

    Labels:
        age_a -> 0
        age_b -> 1
    """
    aucs: List[float] = []

    y = np.asarray(dev_variants, dtype=np.int64)
    groups = np.asarray(dev_user_ids, dtype=np.int64)

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise RuntimeError("Dev split probe labels have fewer than 2 classes")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError("Need at least 2 unique dev users for grouped CV")

    group_to_labels: Dict[int, set] = {}
    for uid, label in zip(groups.tolist(), y.tolist()):
        group_to_labels.setdefault(int(uid), set()).add(int(label))

    n_groups_with_0 = sum(1 for labs in group_to_labels.values() if 0 in labs)
    n_groups_with_1 = sum(1 for labs in group_to_labels.values() if 1 in labs)

    max_feasible_splits = min(len(unique_groups), n_groups_with_0, n_groups_with_1)
    if max_feasible_splits < 2:
        raise RuntimeError(
            "Grouped CV is not feasible: fewer than 2 groups available for at least one class"
        )

    n_splits = min(n_splits, max_feasible_splits)
    if n_splits < 2:
        raise RuntimeError("n_splits became < 2 after grouped feasibility checks")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for j in range(dev_vectors.shape[1]):
        X = np.asarray(dev_vectors[:, j, :], dtype=np.float64)

        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="liblinear",
            max_iter=max_iter,
            random_state=seed,
        )

        oof_scores = cross_val_predict(
            clf,
            X,
            y,
            groups=groups,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]

        auc = float(roc_auc_score(y, oof_scores))
        aucs.append(auc)

    return aucs


def normalize_weights_from_auc(
    aucs: List[float],
    eps: float = 1e-12,
) -> Tuple[List[float], List[float]]:
    """
    Convert raw dev AUCs into chance-corrected separability weights.

    sep_l = max(AUC_l - 0.5, 0)

    Returns:
      sep_excess: list[float]
      weights: list[float]
    """
    arr = np.asarray(aucs, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)

    sep_excess = np.maximum(arr - 0.5, 0.0)
    total = float(sep_excess.sum())

    if total < eps:
        weights = np.full_like(sep_excess, 1.0 / len(sep_excess), dtype=np.float64)
    else:
        weights = sep_excess / total

    return [float(x) for x in sep_excess.tolist()], [float(x) for x in weights.tolist()]


def load_layerwise_internal_eval(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    required = {"user_id", "split", "delta_q1", "delta_q2", "delta_q3", "delta_q4"}

    for row in iter_jsonl_strict(path):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"Malformed eval layerwise row; missing {sorted(missing)}: {row}")

        split = str(row["split"]).strip().lower()
        if split != "eval":
            raise ValueError(f"Expected eval row, got split={split}: {row}")

        rows.append(row)

    return rows


def compute_d_in_for_eval_rows(
    rows: List[Dict[str, Any]],
    weights: List[float],
    sep_excess: List[float],
) -> List[Dict[str, Any]]:
    if len(weights) != 4:
        raise ValueError("Expected 4 weights")
    if len(sep_excess) != 4:
        raise ValueError("Expected 4 chance-corrected separability values")

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        uid = int(row["user_id"])

        d1 = float(row["delta_q1"])
        d2 = float(row["delta_q2"])
        d3 = float(row["delta_q3"])
        d4 = float(row["delta_q4"])

        d_in = (
            weights[0] * d1
            + weights[1] * d2
            + weights[2] * d3
            + weights[3] * d4
        )

        rec = {
            "user_id": uid,
            "split": "eval",
            "relative_layers": row.get("relative_layers", [0.25, 0.5, 0.75, 1.0]),
            "layer_indices": row.get("layer_indices", []),
            "delta_q1": d1,
            "delta_q2": d2,
            "delta_q3": d3,
            "delta_q4": d4,
            "sep_excess_over_chance": {
                "q1": sep_excess[0],
                "q2": sep_excess[1],
                "q3": sep_excess[2],
                "q4": sep_excess[3],
            },
            "weights": {
                "q1": weights[0],
                "q2": weights[1],
                "q3": weights[2],
                "q4": weights[3],
            },
            "d_in": float(d_in),
        }
        out_rows.append(rec)

    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ_DEFAULT, help="Input internal_vectors.npz")
    ap.add_argument("--split", default=SPLIT_DEFAULT, help="Input split.jsonl")
    ap.add_argument(
        "--internal_layers_eval",
        default=INTERNAL_LAYERS_EVAL_DEFAULT,
        help="Input internal_distance_layers.eval.jsonl",
    )
    ap.add_argument("--out_weights", default=OUT_WEIGHTS_DEFAULT, help="Output probe_weights.json")
    ap.add_argument("--out_eval", default=OUT_INTERNAL_EVAL_DEFAULT, help="Output internal_distance.eval.jsonl")

    ap.add_argument("--seed", type=int, default=1234, help="Random seed for probe fitting")
    ap.add_argument("--max_iter", type=int, default=2000, help="Max iterations for logistic regression")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    ap.add_argument("--cv_folds", type=int, default=5, help="Number of grouped CV folds on dev split")
    args = ap.parse_args()

    for p, name in [
        (args.npz, "npz"),
        (args.split, "split"),
        (args.internal_layers_eval, "internal_layers_eval"),
    ]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Input {name} file not found: {p}")

    ensure_parent_dir(args.out_weights)
    ensure_parent_dir(args.out_eval)

    split_map = load_split_map(args.split)
    user_ids, variants, vectors, relative_layers, layer_indices = load_internal_vectors(args.npz)

    dev_variants, dev_vectors, dev_user_ids = build_dev_probe_data(
        user_ids=user_ids,
        variants=variants,
        vectors=vectors,
        split_map=split_map,
    )

    aucs = fit_probe_auc_per_layer_grouped_cv(
        dev_variants=dev_variants,
        dev_vectors=dev_vectors,
        dev_user_ids=dev_user_ids,
        seed=args.seed,
        max_iter=args.max_iter,
        C=args.C,
        n_splits=args.cv_folds,
    )

    sep_excess, weights = normalize_weights_from_auc(aucs)

    weights_obj = {
        "seed": args.seed,
        "max_iter": args.max_iter,
        "C": args.C,
        "cv_folds_requested": args.cv_folds,
        "relative_layers": relative_layers,
        "layer_indices": layer_indices,
        "auc_dev_q1": float(aucs[0]),
        "auc_dev_q2": float(aucs[1]),
        "auc_dev_q3": float(aucs[2]),
        "auc_dev_q4": float(aucs[3]),
        "sep_excess_over_chance": {
            "q1": float(sep_excess[0]),
            "q2": float(sep_excess[1]),
            "q3": float(sep_excess[2]),
            "q4": float(sep_excess[3]),
        },
        "weights": {
            "q1": float(weights[0]),
            "q2": float(weights[1]),
            "q3": float(weights[2]),
            "q4": float(weights[3]),
        },
        "n_dev_rows": int(dev_vectors.shape[0]),
        "n_dev_users_unique": int(len(np.unique(dev_user_ids))),
    }

    with open(args.out_weights, "w", encoding="utf-8") as f:
        json.dump(weights_obj, f, ensure_ascii=False, indent=2)

    eval_rows = load_layerwise_internal_eval(args.internal_layers_eval)
    out_rows = compute_d_in_for_eval_rows(eval_rows, weights, sep_excess)

    with open(args.out_eval, "w", encoding="utf-8") as f_out:
        for rec in out_rows:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        "[OK] Dev grouped-CV AUCs: "
        f"q1={aucs[0]:.6f}, q2={aucs[1]:.6f}, q3={aucs[2]:.6f}, q4={aucs[3]:.6f}"
    )
    print(
        "[OK] Excess-over-chance separability: "
        f"q1={sep_excess[0]:.6f}, q2={sep_excess[1]:.6f}, "
        f"q3={sep_excess[2]:.6f}, q4={sep_excess[3]:.6f}"
    )
    print(
        "[OK] Normalized weights: "
        f"q1={weights[0]:.6f}, q2={weights[1]:.6f}, "
        f"q3={weights[2]:.6f}, q4={weights[3]:.6f}"
    )
    print(f"[OK] Wrote probe weights: {args.out_weights}")
    print(f"[OK] Wrote eval internal distances: {len(out_rows)} rows -> {args.out_eval}")


if __name__ == "__main__":
    main()