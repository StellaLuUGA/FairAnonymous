#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generation.py

Generate LLM recommendation outputs for FairGap counterfactual prompt pairs
using Hugging Face Transformers.

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
- generations.jsonl

Each output row contains:
- user_id
- variant
- prompt
- output_text
- decoding metadata
- run metadata
- pair metadata

Notes
-----
This script requires torch and transformers. Some open-weight models may require
Hugging Face access approval and a cached login token or an HF_TOKEN /
HUGGINGFACE_HUB_TOKEN environment variable.

Example
-------
python scripts/3run_generation.py \
  --pairs data/movielens_smoke/gender/pairs_sample.jsonl \
  --out data/movielens_smoke/gender/generations_sample.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --device_map auto \
  --max_new_tokens 160 \
  --resume
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


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


def normalize_variant_name(x: Any) -> str:
    return str(x).strip().lower()


def load_done_keys(out_path: str) -> Set[Tuple[str, str]]:
    """
    For resume, return set of (user_id, variant) already generated.
    """
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(out_path):
        return done

    with open(out_path, "r", encoding="utf-8") as f:
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
            }
        )
    else:
        info["cuda_available"] = False
    return info


def reset_all_seeds(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pair_seed(seed_base: int, user_id: str, variant: str) -> int:
    """
    Derive a stable per-user, per-variant seed that also works for anonymized
    string user IDs.
    """
    key = f"{seed_base}::{user_id}::{variant}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _from_pretrained_kwargs(dtype: torch.dtype) -> Dict[str, Any]:
    """
    Transformers versions vary:
    - some prefer torch_dtype=
    - some accept dtype=
    """
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    params = sig.parameters
    if "dtype" in params:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def _tokenize_with_or_without_chat_template(
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    prefer_chat_template: bool,
    system_prompt: str = "",
) -> Dict[str, torch.Tensor]:
    use_chat = bool(prefer_chat_template) and hasattr(tokenizer, "apply_chat_template")

    if use_chat:
        try:
            messages = []
            if str(system_prompt).strip():
                messages.append({"role": "system", "content": str(system_prompt).strip()})
            messages.append({"role": "user", "content": prompt})

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
    prefer_chat_template: bool,
    system_prompt: str,
) -> str:
    inputs = _tokenize_with_or_without_chat_template(
        tokenizer=tokenizer,
        prompt=prompt,
        device=model.device,
        prefer_chat_template=prefer_chat_template,
        system_prompt=system_prompt,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    gen_ids = out_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="Input pairs.jsonl")
    ap.add_argument("--out", required=True, help="Output generations.jsonl")

    ap.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local model path.",
    )

    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for loading.",
    )
    ap.add_argument(
        "--device_map",
        default="auto",
        help='Transformers device_map, e.g. "auto" or "cuda:0".',
    )

    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling. If omitted, generation is greedy/deterministic.",
    )
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument(
        "--no_chat_template",
        action="store_true",
        help=(
            "Use raw prompt tokenization. Use this for the simple prompt family "
            "or when the model has no reliable chat template."
        ),
    )

    ap.add_argument(
        "--system_prompt",
        default="",
        help=(
            "System message used when chat template is enabled. For structured "
            "or optimized prompt families, pass the exact system message used "
            "in the paper. Empty string means no separate system message."
        ),
    )

    ap.add_argument(
        "--hf_token",
        default="true",
        help="Use 'true' for cached/env token, explicit token string, or 'false' to disable.",
    )

    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code in from_pretrained when needed.",
    )

    args = ap.parse_args()
    ensure_dir(os.path.dirname(args.out))

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

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
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_kwargs = _from_pretrained_kwargs(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        token=token_arg,
        trust_remote_code=bool(args.trust_remote_code),
        **model_kwargs,
    )
    model.eval()

    device_info = get_device_summary()
    done = load_done_keys(args.out) if args.resume else set()
    prefer_chat_template = not args.no_chat_template

    if not args.do_sample:
        args.temperature = 0.0
        args.top_p = 1.0

    n_pairs = 0
    n_written = 0
    n_skipped_schema = 0
    t0 = time.time()

    with open(args.out, "a", encoding="utf-8") as f_out:
        for row in iter_jsonl(args.pairs):
            if "user_id" not in row:
                continue

            user_id = str(row["user_id"])
            pair_prompts = get_pair_prompts(row)
            if not pair_prompts:
                n_skipped_schema += 1
                continue

            n_pairs += 1
            if args.limit > 0 and n_pairs > args.limit:
                break

            meta = row.get("metadata", {})

            for variant, prompt in pair_prompts:
                key = (user_id, variant)
                if key in done:
                    continue

                seed_used = make_pair_seed(args.seed, user_id, variant) if args.do_sample else args.seed
                reset_all_seeds(seed_used)

                out_text = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    repetition_penalty=args.repetition_penalty,
                    prefer_chat_template=prefer_chat_template,
                    system_prompt=args.system_prompt,
                )

                rec = {
                    "user_id": user_id,
                    "variant": variant,
                    "prompt": prompt,
                    "output_text": out_text,
                    "decoding": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": bool(args.do_sample),
                        "temperature": float(args.temperature),
                        "top_p": float(args.top_p),
                        "repetition_penalty": float(args.repetition_penalty),
                        "prefer_chat_template": bool(prefer_chat_template),
                    },
                    "run": {
                        "model": args.model,
                        "dtype": str(args.dtype),
                        "device_map": args.device_map,
                        "seed_base": args.seed,
                        "seed_used": seed_used,
                        "timestamp_utc": now_iso(),
                        "device_info": device_info,
                    },
                    "pair_metadata": meta,
                }

                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                n_written += 1
                done.add(key)

            if n_pairs % 50 == 0:
                elapsed = time.time() - t0
                print(f"[PROGRESS] pairs={n_pairs} rows_written={n_written} elapsed_sec={elapsed:.1f}")

    elapsed = time.time() - t0
    print(f"[OK] Done. pairs_seen={n_pairs} rows_written={n_written} skipped_schema={n_skipped_schema} elapsed_sec={elapsed:.1f}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
