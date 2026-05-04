#!/usr/bin/env python3

"""
python3 fairgap/4extract_internal_vector.py \
  --fairgap_out/pairs.jsonl \
  --out_jsonl fairgap_out/vectors.jsonl \
  --out_npz fairgap_out/internal_vectors.npz \
  --model google/gemma-7b-it \
  --batch_size 8 \
  --max_length 2048 \
  --dtype bfloat16 \
  --device_map auto \
  --use_chat_template \
  --save_dtype float32

"""
#!/usr/bin/env python3
import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"

_BASE_DIR = "examples/toy_out"
_PAIRS_DEFAULT = os.path.join(_BASE_DIR, "pairs.jsonl")
_OUT_JSONL_DEFAULT = os.path.join(_BASE_DIR, "vectors.jsonl")
_OUT_NPZ_DEFAULT = os.path.join(_BASE_DIR, "internal_vectors.npz")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_num_transformer_layers(model: AutoModelForCausalLM) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return len(layers)
    raise RuntimeError("Could not infer number of transformer layers.")


def quartile_layer_indices(num_transformer_layers: int) -> List[int]:
    if num_transformer_layers <= 0:
        raise ValueError("num_transformer_layers must be positive")
    relative_points = [0.25, 0.5, 0.75, 1.0]
    indices = []
    for r in relative_points:
        idx = int(np.ceil(r * num_transformer_layers) - 1)
        idx = max(0, min(num_transformer_layers - 1, idx))
        indices.append(idx)
    return indices


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def load_done_keys_jsonl(out_jsonl: str) -> Set[Tuple[int, str]]:
    done: Set[Tuple[int, str]] = set()
    if not out_jsonl or not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                uid = int(obj.get("user_id"))
                variant = str(obj.get("variant"))
                if variant in ("age_a", "age_b"):
                    done.add((uid, variant))
            except Exception:
                continue
    return done


def get_device_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch_version": torch.__version__}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info.update({
            "cuda_available": True,
            "cuda_device": idx,
            "cuda_name": torch.cuda.get_device_name(idx),
            "cuda_capability": torch.cuda.get_device_capability(idx),
        })
    else:
        info["cuda_available"] = False
    return info


def wrap_with_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


@torch.no_grad()
def extract_batch_vectors_all_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_indices: List[int],
    max_length: int,
) -> Tuple[np.ndarray, List[int], List[int], int]:

    try:
        from transformers.masking_utils import create_causal_mask
        has_create_causal_mask = True
    except ImportError:
        has_create_causal_mask = False

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"]
    attention_mask_2d = enc.get("attention_mask", None)

    B, T = input_ids.shape

    if attention_mask_2d is None:
        last_idx = [T - 1] * B
        prompt_lens = [T] * B
    else:
        am = attention_mask_2d.cpu().numpy().astype(np.int32)
        prompt_lens = am.sum(axis=1).tolist()
        last_idx = [max(0, l - 1) for l in prompt_lens]

    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise RuntimeError("Unsupported architecture. Expected model.model.layers.")

    all_layers = inner.layers
    num_transformer_layers = len(all_layers)

    for li in layer_indices:
        if li < 0 or li >= num_transformer_layers:
            raise ValueError(f"layer_idx={li} invalid for {num_transformer_layers} layers.")

    layer_index_set: Dict[int, int] = {li: pos for pos, li in enumerate(layer_indices)}

    embed_layer = inner.embed_tokens
    embed_dev = next(embed_layer.parameters()).device
    hidden_states = embed_layer(input_ids.to(embed_dev))

    cache_position = torch.arange(T, device=embed_dev, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0).expand(B, -1)

    am_on_dev = attention_mask_2d.to(embed_dev) if attention_mask_2d is not None else None
    if has_create_causal_mask:
        causal_mask = create_causal_mask(
            config=inner.config,
            input_embeds=hidden_states,
            attention_mask=am_on_dev,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
    else:
        dtype = hidden_states.dtype
        causal_mask = torch.tril(torch.ones(T, T, device=embed_dev, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = torch.zeros_like(causal_mask, dtype=dtype).masked_fill(
            ~causal_mask,
            torch.finfo(dtype).min if dtype != torch.bfloat16 else -3.3895e+38,
        )
    del am_on_dev

    rotary = inner.rotary_emb
    rotary_params = list(rotary.parameters())
    rotary_dev = rotary_params[0].device if rotary_params else embed_dev
    position_embeddings = rotary(
        hidden_states.to(rotary_dev),
        position_ids.to(rotary_dev),
    )

    captured: Dict[int, np.ndarray] = {}

    for li, layer in enumerate(all_layers):
        layer_dev = next(layer.parameters()).device

        h = hidden_states.to(layer_dev)
        pos_ids = position_ids.to(layer_dev)
        cache_pos = cache_position.to(layer_dev)
        pos_emb = (
            position_embeddings[0].to(layer_dev),
            position_embeddings[1].to(layer_dev),
        )
        causal_m = causal_mask.to(layer_dev) if causal_mask is not None else None

        out = layer(
            h,
            attention_mask=causal_m,
            position_ids=pos_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_pos,
            position_embeddings=pos_emb,
        )

        hidden_states = out[0] if isinstance(out, (tuple, list)) else out

        if li in layer_index_set:
            li_pos = layer_index_set[li]
            arr = np.zeros((B, hidden_states.shape[-1]), dtype=np.float32)

            for i in range(B):
                last_token_idx = last_idx[i]
                arr[i] = (
                    hidden_states[i, last_token_idx, :]
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

            captured[li_pos] = arr

        del h, causal_m, pos_ids, cache_pos, pos_emb, out

    missing = [layer_indices[p] for p in range(len(layer_indices)) if p not in captured]
    if missing:
        raise RuntimeError(f"No vectors captured for layers: {missing}")

    H = next(iter(captured.values())).shape[1]
    all_vecs = np.zeros((B, len(layer_indices), H), dtype=np.float32)
    for li_pos in range(len(layer_indices)):
        all_vecs[:, li_pos] = captured[li_pos]

    del hidden_states, causal_mask, position_embeddings, captured, input_ids, attention_mask_2d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_vecs, prompt_lens, last_idx, num_transformer_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=_PAIRS_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--hf_token", default="true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_jsonl", default=_OUT_JSONL_DEFAULT)
    ap.add_argument("--out_npz", default=_OUT_NPZ_DEFAULT)
    ap.add_argument("--save_dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit_users", type=int, default=0)
    args = ap.parse_args()

    if not args.out_jsonl and not args.out_npz:
        raise SystemExit("You must set at least one of --out_jsonl or --out_npz")

    if args.out_jsonl:
        ensure_dir(os.path.dirname(args.out_jsonl))
    if args.out_npz:
        ensure_dir(os.path.dirname(args.out_npz))

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if isinstance(args.hf_token, str) and args.hf_token.lower() == "false":
        token_arg = None
    elif isinstance(args.hf_token, str) and args.hf_token.lower() == "true":
        token_arg = True
    else:
        token_arg = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=token_arg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.device_map == "auto":
        n_gpus = torch.cuda.device_count()
        gpu_cap = "46GiB"
        max_mem: Dict[Any, str] = {i: gpu_cap for i in range(n_gpus)}
        max_mem["cpu"] = "150GiB"
    else:
        max_mem = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        token=token_arg,
        max_memory=max_mem,
    )
    model.eval()

    device_info = get_device_summary()

    done: Set[Tuple[int, str]] = set()
    if args.resume and args.out_jsonl:
        done = load_done_keys_jsonl(args.out_jsonl)

    num_transformer_layers = get_num_transformer_layers(model)
    layer_indices = quartile_layer_indices(num_transformer_layers)
    relative_layers = [0.25, 0.5, 0.75, 1.0]

    print(
        f"[INFO] anchor=last_prompt_token "
        f"num_layers={num_transformer_layers} "
        f"layer_indices={layer_indices} "
        f"save_dtype={args.save_dtype}"
    )

    npz_user_ids: List[int] = []
    npz_variant: List[int] = []
    npz_vectors: List[np.ndarray] = []

    if args.resume and args.out_npz and os.path.exists(args.out_npz):
        try:
            ex = np.load(args.out_npz, allow_pickle=False)
            n_loaded = len(ex["user_id"])
            for i in range(n_loaded):
                npz_user_ids.append(int(ex["user_id"][i]))
                npz_variant.append(int(ex["variant"][i]))
                npz_vectors.append(ex["vectors"][i])
            print(f"[RESUME] Loaded {n_loaded} existing rows: {args.out_npz}")
        except Exception as e:
            print(f"[RESUME] Warning: could not load NPZ ({e!r}) — starting fresh.")

    def maybe_wrap(prompt: str) -> str:
        return wrap_with_chat_template(tokenizer, prompt) if args.use_chat_template else prompt

    t0 = time.time()
    n_pairs_seen = 0
    n_variants_written = 0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    f_out = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None
    batch_items: List[Tuple[int, str, str, Dict[str, Any]]] = []

    def flush_batch(items: List[Tuple[int, str, str, Dict[str, Any]]]) -> None:
        nonlocal n_variants_written
        if not items:
            return

        texts = [it[2] for it in items]
        all_vecs = None
        prompt_lens = None
        last_idx = None
        actual_max_length = None
        uids_skipped = [it[0] for it in items]

        for attempt_max_length in [args.max_length, 512, 256]:
            try:
                all_vecs, prompt_lens, last_idx, _ = extract_batch_vectors_all_layers(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    layer_indices=layer_indices,
                    max_length=attempt_max_length,
                )
                actual_max_length = attempt_max_length
                break
            except torch.cuda.OutOfMemoryError as oom:
                print(f"[OOM] max_length={attempt_max_length} batch={uids_skipped} — {oom!r}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        else:
            print(f"[SKIP] Skipping users {uids_skipped} after OOM.")
            return

        for i, (uid, variant, _text, meta) in enumerate(items):
            vec_all = all_vecs[i]

            if f_out is not None:
                rec = {
                    "user_id": uid,
                    "variant": variant,
                    "layer_mode": "quartiles",
                    "relative_layers": relative_layers,
                    "layer_indices": layer_indices,
                    "anchor": "last_prompt_token",
                    "vector_dim": int(vec_all.shape[-1]),
                    "vectors_by_quartile": {
                        "q1": vec_all[0].tolist(),
                        "q2": vec_all[1].tolist(),
                        "q3": vec_all[2].tolist(),
                        "q4": vec_all[3].tolist(),
                    },
                    "tokenization": {
                        "prompt_len": int(prompt_lens[i]),
                        "last_index": int(last_idx[i]),
                        "use_chat_template": bool(args.use_chat_template),
                        "max_length": int(actual_max_length),
                    },
                    "run": {
                        "model": args.model,
                        "dtype": str(args.dtype),
                        "save_dtype": str(args.save_dtype),
                        "device_map": args.device_map,
                        "seed": int(args.seed),
                        "timestamp_utc": now_iso(),
                        "device_info": device_info,
                    },
                    "pair_metadata": meta,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()

            if args.out_npz:
                npz_user_ids.append(uid)
                npz_variant.append(0 if variant == "age_a" else 1)
                npz_vectors.append(vec_all)

            n_variants_written += 1

    for row in iter_jsonl(args.pairs):
        uid = int(row["user_id"])
        meta = row.get("metadata", {}) or {}

        n_pairs_seen += 1
        if args.limit_users > 0 and n_pairs_seen > args.limit_users:
            break

        if (uid, "age_a") not in done:
            pa = maybe_wrap(row["prompt_age_a"])
            batch_items.append((uid, "age_a", pa, meta))

        if (uid, "age_b") not in done:
            pb = maybe_wrap(row["prompt_age_b"])
            batch_items.append((uid, "age_b", pb, meta))

        if len(batch_items) >= args.batch_size:
            flush_batch(batch_items[: args.batch_size])
            batch_items = batch_items[args.batch_size:]

        if n_pairs_seen % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"[PROGRESS] pairs_seen={n_pairs_seen} "
                f"variants_written={n_variants_written} "
                f"elapsed_sec={elapsed:.1f}"
            )

        if n_pairs_seen % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    while batch_items:
        chunk = batch_items[: args.batch_size]
        batch_items = batch_items[args.batch_size:]
        flush_batch(chunk)

    if f_out is not None:
        f_out.close()

    if args.out_npz:
        vec_mat = (
            np.stack(npz_vectors, axis=0)
            if npz_vectors
            else np.zeros((0, 4, 0), dtype=np.float32)
        )
        vec_mat = vec_mat.astype(
            np.float16 if args.save_dtype == "float16" else np.float32
        )

        np.savez_compressed(
            args.out_npz,
            user_id=np.array(npz_user_ids, dtype=np.int32),
            variant=np.array(npz_variant, dtype=np.int8),
            layer_mode=np.array(["quartiles"]),
            relative_layers=np.array(relative_layers, dtype=np.float32),
            layer_indices=np.array(layer_indices, dtype=np.int32),
            anchor=np.array(["last_prompt_token"]),
            vectors=vec_mat,
        )
        print(f"[OK] Saved NPZ: {args.out_npz} shape={vec_mat.shape} dtype={vec_mat.dtype}")

    elapsed = time.time() - t0
    print(
        f"[OK] Done. pairs_seen={n_pairs_seen} "
        f"variants_written={n_variants_written} "
        f"elapsed_sec={elapsed:.1f}"
    )
    if args.out_jsonl:
        print(f"JSONL: {args.out_jsonl}")
    if args.out_npz:
        print(f"NPZ:   {args.out_npz}")


if __name__ == "__main__":
    main()