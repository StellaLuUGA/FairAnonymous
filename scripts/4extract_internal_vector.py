#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_internal_vector.py

Extract internal representation vectors for FairGap counterfactual prompt pairs.

Inputs
------
- pairs.jsonl from build_minimal_pairs.py

Expected pair fields
--------------------
Preferred generic fields:
- user_id
- prompt_a
- prompt_b
- variant_a
- variant_b
- metadata

Backward-compatible fields are also supported:
- prompt_female
- prompt_male

Outputs
-------
- vectors.jsonl
- internal_vectors.npz

Method
------
For each counterfactual prompt, this script extracts hidden-state vectors from
four relative layer depths: 25%, 50%, 75%, and 100% of transformer layers. The
anchor is the last prompt token.

Notes
-----
This is a GPU-dependent full-reproduction step. Lightweight smoke-test artifacts
may provide precomputed internal-distance files instead of raw hidden vectors.

Example
-------
python scripts/4extract_internal_vector.py \
  --pairs data/movielens_smoke/gender/pairs_sample.jsonl \
  --out_jsonl data/movielens_smoke/gender/vectors_sample.jsonl \
  --out_npz data/movielens_smoke/gender/internal_vectors_sample.npz \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --batch_size 1 \
  --max_length 512 \
  --dtype float16 \
  --device_map cuda:0 \
  --save_dtype float32
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_variant_name(x: Any) -> str:
    return str(x).strip().lower()


def get_num_transformer_layers(model: AutoModelForCausalLM) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            return len(layers)
    raise RuntimeError(
        "Could not infer number of transformer layers. "
        "Expected model.config.num_hidden_layers or model.model.layers."
    )


def quartile_layer_indices(num_transformer_layers: int) -> List[int]:
    if num_transformer_layers <= 0:
        raise ValueError("num_transformer_layers must be positive")
    relative_points = [0.25, 0.5, 0.75, 1.0]
    indices: List[int] = []
    for rel in relative_points:
        idx = int(np.ceil(rel * num_transformer_layers) - 1)
        idx = max(0, min(num_transformer_layers - 1, idx))
        indices.append(idx)
    return indices


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


def load_done_keys_jsonl(out_jsonl: str) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not out_jsonl or not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                uid = str(obj.get("user_id"))
                variant = normalize_variant_name(obj.get("variant"))
                if uid and variant:
                    done.add((uid, variant))
            except Exception:
                continue
    return done


def get_device_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch_version": torch.__version__}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info.update(
            {
                "cuda_available": True,
                "cuda_device": idx,
                "cuda_name": torch.cuda.get_device_name(idx),
                "cuda_capability": torch.cuda.get_device_capability(idx),
                "cuda_total_memory_gib": round(
                    torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3), 2
                ),
            }
        )
    else:
        info["cuda_available"] = False
    return info


def wrap_with_chat_template(
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: str = "",
) -> str:
    messages = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cudaerrormemoryallocation" in msg
    )


def _model_load_kwargs(
    dtype: torch.dtype,
    device_map: str,
    token_arg: Any,
    max_mem: Optional[Dict[Any, str]],
    trust_remote_code: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "token": token_arg,
        "trust_remote_code": bool(trust_remote_code),
    }

    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in sig.parameters:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype

    if max_mem is not None:
        kwargs["max_memory"] = max_mem

    return kwargs


@torch.no_grad()
def extract_batch_vectors_all_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_indices: List[int],
    max_length: int,
) -> Tuple[np.ndarray, List[int], List[int], int]:
    """
    Extract hidden-state vectors using the last prompt token as anchor.

    This implementation is designed for common decoder-only Hugging Face models
    exposing model.model.layers, such as Llama/Mistral/Qwen-style architectures.
    """
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

    batch_size, seq_len = input_ids.shape

    if attention_mask_2d is None:
        last_idx = [seq_len - 1] * batch_size
        prompt_lens = [seq_len] * batch_size
    else:
        attn = attention_mask_2d.cpu().numpy().astype(np.int32)
        prompt_lens = attn.sum(axis=1).tolist()
        last_idx = [max(0, length - 1) for length in prompt_lens]

    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise RuntimeError(
            "Unsupported architecture. Expected a decoder-only model with model.model.layers."
        )

    all_layers = inner.layers
    num_transformer_layers = len(all_layers)

    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_transformer_layers:
            raise ValueError(
                f"layer_idx={layer_idx} invalid for {num_transformer_layers} layers."
            )

    layer_index_set: Dict[int, int] = {
        layer_idx: pos for pos, layer_idx in enumerate(layer_indices)
    }

    embed_layer = inner.embed_tokens
    embed_dev = next(embed_layer.parameters()).device
    hidden_states = embed_layer(input_ids.to(embed_dev))

    cache_position = torch.arange(seq_len, device=embed_dev, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

    attn_on_dev = attention_mask_2d.to(embed_dev) if attention_mask_2d is not None else None
    if has_create_causal_mask:
        causal_mask = create_causal_mask(
            config=inner.config,
            input_embeds=hidden_states,
            attention_mask=attn_on_dev,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
    else:
        dtype = hidden_states.dtype
        causal_mask_bool = torch.tril(torch.ones(seq_len, seq_len, device=embed_dev, dtype=torch.bool))
        causal_mask_bool = causal_mask_bool.unsqueeze(0).unsqueeze(0)
        fill_value = torch.finfo(dtype).min if dtype != torch.bfloat16 else -3.3895e38
        causal_mask = torch.zeros_like(causal_mask_bool, dtype=dtype).masked_fill(
            ~causal_mask_bool,
            fill_value,
        )
    del attn_on_dev

    if not hasattr(inner, "rotary_emb"):
        raise RuntimeError(
            "Unsupported architecture. Expected model.model.rotary_emb for this extraction script."
        )

    rotary = inner.rotary_emb
    rotary_params = list(rotary.parameters())
    rotary_dev = rotary_params[0].device if rotary_params else embed_dev
    position_embeddings = rotary(
        hidden_states.to(rotary_dev),
        position_ids.to(rotary_dev),
    )

    captured: Dict[int, np.ndarray] = {}

    for layer_idx, layer in enumerate(all_layers):
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

        if layer_idx in layer_index_set:
            layer_pos = layer_index_set[layer_idx]
            arr = np.zeros((batch_size, hidden_states.shape[-1]), dtype=np.float32)
            for i in range(batch_size):
                arr[i] = (
                    hidden_states[i, last_idx[i], :]
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
            captured[layer_pos] = arr

        del h, causal_m, pos_ids, cache_pos, pos_emb, out

    missing = [
        layer_indices[pos]
        for pos in range(len(layer_indices))
        if pos not in captured
    ]
    if missing:
        raise RuntimeError(f"No vectors captured for layers: {missing}")

    hidden_dim = next(iter(captured.values())).shape[1]
    all_vecs = np.zeros((batch_size, len(layer_indices), hidden_dim), dtype=np.float32)
    for pos in range(len(layer_indices)):
        all_vecs[:, pos] = captured[pos]

    del hidden_states, causal_mask, position_embeddings, captured, input_ids, attention_mask_2d
    _empty_cuda_cache()

    return all_vecs, prompt_lens, last_idx, num_transformer_layers


def get_pair_prompts(row: Dict[str, Any]) -> Optional[List[Tuple[str, str]]]:
    """
    Return [(variant_a, prompt_a), (variant_b, prompt_b)].

    Supports both generic upload schema and older gender-specific schema.
    """
    if row.get("prompt_a") and row.get("prompt_b"):
        variant_a = normalize_variant_name(row.get("variant_a", "a"))
        variant_b = normalize_variant_name(row.get("variant_b", "b"))
        return [
            (variant_a, str(row["prompt_a"])),
            (variant_b, str(row["prompt_b"])),
        ]

    if row.get("prompt_female") and row.get("prompt_male"):
        return [
            ("female", str(row["prompt_female"])),
            ("male", str(row["prompt_male"])),
        ]

    return None


def variant_to_code(variant: str, variant_order: List[str]) -> int:
    if variant not in variant_order:
        variant_order.append(variant)
    if len(variant_order) > 2:
        raise ValueError(f"Expected exactly two variants, got: {variant_order}")
    return int(variant_order.index(variant))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Input pairs.jsonl")
    ap.add_argument("--model", required=True, help="Hugging Face model id or local model path")
    ap.add_argument("--hf_token", default="true")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="cuda:0")

    ap.add_argument(
        "--use_chat_template",
        action="store_true",
        help=(
            "Apply tokenizer chat template. Use for structured/optimized prompt families "
            "when the model has a reliable chat template."
        ),
    )
    ap.add_argument(
        "--system_prompt",
        default="",
        help=(
            "Optional system message used only when --use_chat_template is set. "
            "Empty string means no separate system message."
        ),
    )

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_jsonl", required=True, help="Output vectors JSONL")
    ap.add_argument("--out_npz", required=True, help="Output internal_vectors NPZ")
    ap.add_argument(
        "--save_dtype",
        default="float32",
        choices=["float16", "float32"],
        help="float32 recommended to preserve IBS signal fidelity.",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit_users", type=int, default=0)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    ensure_dir(os.path.dirname(args.out_jsonl))
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

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if isinstance(args.hf_token, str) and args.hf_token.lower() == "false":
        token_arg: Any = None
    elif isinstance(args.hf_token, str) and args.hf_token.lower() == "true":
        token_arg = env_token if env_token else True
    else:
        token_arg = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        token=token_arg,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.device_map == "auto" and torch.cuda.is_available():
        max_mem: Dict[Any, str] = {}
        for i in range(torch.cuda.device_count()):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            usable_gib = max(1, int(total_gib * 0.85))
            max_mem[i] = f"{usable_gib}GiB"
        max_mem["cpu"] = "120GiB"
    else:
        max_mem = None

    model_kwargs = _model_load_kwargs(
        dtype=dtype,
        device_map=args.device_map,
        token_arg=token_arg,
        max_mem=max_mem,
        trust_remote_code=bool(args.trust_remote_code),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.eval()
    device_info = get_device_summary()

    done: Set[Tuple[str, str]] = set()
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

    npz_user_ids: List[str] = []
    npz_variant: List[int] = []
    npz_vectors: List[np.ndarray] = []
    variant_order: List[str] = []

    if args.resume and args.out_npz and os.path.exists(args.out_npz):
        try:
            existing = np.load(args.out_npz, allow_pickle=False)
            n_loaded = len(existing["user_id"])
            for i in range(n_loaded):
                npz_user_ids.append(str(existing["user_id"][i]))
                npz_variant.append(int(existing["variant"][i]))
                npz_vectors.append(existing["vectors"][i])
            print(f"[RESUME] Loaded {n_loaded} existing rows: {args.out_npz}")
        except Exception as e:
            print(f"[RESUME] Warning: could not load NPZ ({e!r}) — starting fresh.")

    def maybe_wrap(prompt: str) -> str:
        if args.use_chat_template:
            return wrap_with_chat_template(
                tokenizer=tokenizer,
                prompt=prompt,
                system_prompt=args.system_prompt,
            )
        return prompt

    t0 = time.time()
    n_pairs_seen = 0
    n_variants_written = 0
    n_skipped_schema = 0

    _empty_cuda_cache()

    f_out = open(args.out_jsonl, "a", encoding="utf-8")
    batch_items: List[Tuple[str, str, str, Dict[str, Any]]] = []

    def flush_batch(items: List[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        nonlocal n_variants_written
        if not items:
            return

        texts = [item[2] for item in items]
        all_vecs = None
        prompt_lens = None
        last_idx = None
        actual_max_length = None
        ids_for_log = [f"{item[0]}:{item[1]}" for item in items]

        attempt_lengths: List[int] = []
        for candidate in [args.max_length, 1024, 768, 512, 384, 256, 192, 128]:
            if candidate > 0 and candidate not in attempt_lengths:
                attempt_lengths.append(candidate)

        for attempt_max_length in attempt_lengths:
            try:
                _empty_cuda_cache()
                all_vecs, prompt_lens, last_idx, _ = extract_batch_vectors_all_layers(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    layer_indices=layer_indices,
                    max_length=attempt_max_length,
                )
                actual_max_length = attempt_max_length
                break
            except Exception as exc:
                if not _is_oom_error(exc):
                    raise
                print(
                    f"[OOM] max_length={attempt_max_length} "
                    f"batch={ids_for_log} — {exc!r}"
                )
                all_vecs = None
                prompt_lens = None
                last_idx = None
                _empty_cuda_cache()
                continue

        if all_vecs is None or prompt_lens is None or last_idx is None or actual_max_length is None:
            print(f"[SKIP] Skipping batch after OOM: {ids_for_log}")
            return

        for i, (uid, variant, _text, meta) in enumerate(items):
            vec_all = all_vecs[i]

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
                    "system_prompt_used": bool(str(args.system_prompt).strip()),
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

            npz_user_ids.append(uid)
            npz_variant.append(variant_to_code(variant, variant_order))
            npz_vectors.append(vec_all)

            n_variants_written += 1

        _empty_cuda_cache()

    for row in iter_jsonl(args.pairs):
        if "user_id" not in row:
            continue

        uid = str(row["user_id"])
        meta = row.get("metadata", {}) or {}

        pair_prompts = get_pair_prompts(row)
        if not pair_prompts:
            n_skipped_schema += 1
            continue

        n_pairs_seen += 1
        if args.limit_users > 0 and n_pairs_seen > args.limit_users:
            break

        for variant, prompt in pair_prompts:
            if (uid, variant) in done:
                continue
            batch_items.append((uid, variant, maybe_wrap(prompt), meta))

        while len(batch_items) >= args.batch_size:
            flush_batch(batch_items[: args.batch_size])
            batch_items = batch_items[args.batch_size:]

        if n_pairs_seen % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"[PROGRESS] pairs_seen={n_pairs_seen} "
                f"variants_written={n_variants_written} "
                f"skipped_schema={n_skipped_schema} "
                f"elapsed_sec={elapsed:.1f}"
            )

        if n_pairs_seen % 100 == 0:
            _empty_cuda_cache()

    while batch_items:
        chunk = batch_items[: args.batch_size]
        batch_items = batch_items[args.batch_size:]
        flush_batch(chunk)

    f_out.close()

    if npz_vectors:
        vec_mat = np.stack(npz_vectors, axis=0)
    else:
        vec_mat = np.zeros((0, 4, 0), dtype=np.float32)

    vec_mat = vec_mat.astype(
        np.float16 if args.save_dtype == "float16" else np.float32
    )

    np.savez_compressed(
        args.out_npz,
        user_id=np.asarray(npz_user_ids),
        variant=np.asarray(npz_variant, dtype=np.int8),
        variant_order=np.asarray(variant_order),
        layer_mode=np.asarray(["quartiles"]),
        relative_layers=np.asarray(relative_layers, dtype=np.float32),
        layer_indices=np.asarray(layer_indices, dtype=np.int32),
        anchor=np.asarray(["last_prompt_token"]),
        vectors=vec_mat,
    )
    print(
        f"[OK] Saved NPZ: {args.out_npz} "
        f"shape={vec_mat.shape} dtype={vec_mat.dtype}"
    )

    elapsed = time.time() - t0
    print(
        f"[OK] Done. pairs_seen={n_pairs_seen} "
        f"variants_written={n_variants_written} "
        f"skipped_schema={n_skipped_schema} "
        f"elapsed_sec={elapsed:.1f}"
    )
    print(f"JSONL: {args.out_jsonl}")
    print(f"NPZ:   {args.out_npz}")


if __name__ == "__main__":
    main()
