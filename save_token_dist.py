#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()


def to_cpu_float_list(t: torch.Tensor):
    return [float(x) for x in t.detach().cpu().tolist()]


def main():
    parser = argparse.ArgumentParser(description="Save per-token next-token distributions (top-k) for a prompt.")
    parser.add_argument("--model_id", required=True, help="Hugging Face model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt_file", required=True, help="Path to a text file containing the prompt.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top tokens to store per position.")
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass through to HF loaders.")
    parser.add_argument("--max_length", type=int, default=None, help="Optional truncation length for the prompt (tokens).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code, token=os.getenv("HF_TOKEN"))
    # Ensure pad_token exists (not required, but nice for attention masks)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if device == "cuda" else None,
        token=os.getenv("HF_TOKEN")
    )
    model.eval()

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")

    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    if args.max_length is not None and input_ids.shape[-1] > args.max_length:
        input_ids = input_ids[:, : args.max_length]
        attention_mask = attention_mask[:, : args.max_length]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # logits: [batch=1, seq_len, vocab]
        logits = out.logits  # next-token logits for positions 0..seq_len-1 predicting token at those positions
        # We want the distribution for "next token after prefix of length t".
        # logits[:, t, :] is the distribution for token at position t (given tokens < t).
        probs = torch.softmax(logits.float(), dim=-1)  # convert to float32 for numerical stability

    vocab_size = probs.shape[-1]
    seq_len = probs.shape[1]
    input_ids_list = to_cpu_float_list(input_ids[0])

    # Build per-position top-k records
    k = min(args.top_k, vocab_size)
    per_pos = []
    for t in range(seq_len):
        p = probs[0, t, :]  # [vocab]
        topk = torch.topk(p, k=k, dim=-1)
        top_indices = topk.indices
        top_values = topk.values
        top_sum = float(top_values.sum().item())
        other_prob = max(0.0, 1.0 - top_sum)

        # Map ids to tokens (string) to help inspect later
        top_entries = []
        for idx, val in zip(to_cpu_float_list(top_indices), to_cpu_float_list(top_values)):
            idx_int = int(idx)
            tok_str = tokenizer.decode([idx_int], clean_up_tokenization_spaces=False)
            top_entries.append({
                "id": idx_int,
                "prob": float(val),
                "token": tok_str
            })

        # For clarity, also record the actual next token in the prompt at this position (if exists)
        actual_token_id = None
        if t < len(input_ids_list):
            actual_token_id = int(input_ids_list[t])

        per_pos.append({
            "position": t,
            "topk": top_entries,
            "other_prob": other_prob,
            "actual_token_id": actual_token_id
        })

    result = {
        "model_id": args.model_id,
        "top_k": k,
        "vocab_size": vocab_size,
        "tokenizer_name": tokenizer.name_or_path,
        "prompt_text": prompt_text,
        "input_ids": [int(x) for x in input_ids_list],
        "sequence_length": seq_len,
        "per_position": per_pos,
    }

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote results to: {args.output}")
    print(f"Seq length: {seq_len} | Stored top-{k} + OTHER for each position.")


if __name__ == "__main__":
    main()

