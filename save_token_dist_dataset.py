#!/usr/bin/env python3
"""
Save per-token next-token distributions (top-k) for randomly sampled MMLU-Pro items.

- Loads MMLU-Pro via Hugging Face Datasets (default: TIGER-Lab/MMLU-Pro, split="test").
- Randomly samples n items (without replacement).
- For each item, builds a multiple-choice prompt and runs a single forward pass.
- Saves, for every prompt token position, the top-k next-token probabilities + an "OTHER" bucket.

Output format: a single JSON file containing an array of result objects.
Writes to <output>.tmp first, fsyncs, then atomically renames to <output>.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# ---------- Prompt templating ----------

DEFAULT_TEMPLATE = """You are given a multiple-choice question. Select the single best answer.
Question: {question}
Options:
{options_block}
Answer:"""

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt_from_record(rec: dict, template: str = DEFAULT_TEMPLATE) -> Tuple[str, List[str]]:
    """
    Extracts question + options from an MMLU-Pro record and builds a prompt.
    Returns (prompt_text, option_labels_list).
    """
    question = rec.get("question") or rec.get("stem") or rec.get("prompt") or ""
    options = (
        rec.get("options")
        or rec.get("choices")
        or rec.get("answers")            # some variants
        or rec.get("candidate_answers")   # some variants
    )

    # Some variants put options under single-letter keys "A","B","C","D"
    if options is None:
        letter_keys = [k for k in rec.keys() if isinstance(k, str) and len(k) == 1 and k.isalpha()]
        letter_keys = sorted(letter_keys)
        if letter_keys:
            options = [rec[k] for k in letter_keys]

    if not isinstance(options, list) or len(options) == 0:
        raise ValueError("Could not find a list of options/choices in the record.")

    labels = [LETTER_ALPH[i] for i in range(len(options))]
    lines = [f"{labels[i]}. {options[i]}" for i in range(len(options))]
    options_block = "\n".join(lines)

    prompt = template.format(question=question, options_block=options_block)
    return prompt, labels


# ---------- Utilities ----------

def to_cpu_float_list(t: torch.Tensor) -> List[float]:
    return [float(x) for x in t.detach().cpu().tolist()]


def atomic_write_json(path: Path, obj, *, indent: int = 2, ensure_ascii: bool = False):
    """
    Write JSON to a temp file in the same directory, fsync, then atomically replace.
    Ensures the final file is either old or fully new; never corrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Save per-token next-token distributions for random MMLU-Pro samples.")
    parser.add_argument("--model_id", required=True, help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of random MMLU-Pro items to sample.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top tokens to store per position.")
    parser.add_argument("--output", required=True, help="Output JSON file (array of result objects).")
    parser.add_argument("--dataset_name", default="TIGER-Lab/MMLU-Pro", help="HF datasets path for MMLU-Pro.")
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for sampling.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass through to HF loaders.")
    parser.add_argument("--prompt_template_file", default=None,
                        help="Optional path to a custom template with {question} and {options_block}.")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Optional truncation length for the prompt (tokens).")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} split={args.split}")
    ds = load_dataset(args.dataset_name, split=args.split, token=os.getenv("HF_TOKEN"))

    if len(ds) < args.n_samples:
        raise ValueError(f"Requested n_samples={args.n_samples} exceeds dataset size {len(ds)}")

    rnd = random.Random(args.seed)
    indices = list(range(len(ds)))
    rnd.shuffle(indices)
    indices = indices[: args.n_samples]

    template = DEFAULT_TEMPLATE
    if args.prompt_template_file:
        template = Path(args.prompt_template_file).read_text(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code, token=os.getenv("HF_TOKEN"))
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

    results: List[dict] = []

    for count, idx in enumerate(indices, start=1):
        rec = ds[int(idx)]
        try:
            prompt_text, option_labels = build_prompt_from_record(rec, template)
        except Exception as e:
            print(f"[SKIP] idx={idx} due to prompt build error: {e}")
            continue

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
            logits = out.logits  # [1, seq_len, vocab]
            probs = torch.softmax(logits.float(), dim=-1)

        vocab_size = probs.shape[-1]
        seq_len = probs.shape[1]
        ids_list = [int(x) for x in to_cpu_float_list(input_ids[0])]

        k = min(args.top_k, vocab_size)
        per_pos = []
        for t in range(seq_len):
            p = probs[0, t, :]  # [vocab]
            topk = torch.topk(p, k=k, dim=-1)
            top_indices = topk.indices
            top_values = topk.values
            top_sum = float(top_values.sum().item())
            other_prob = max(0.0, 1.0 - top_sum)

            top_entries = []
            for idx_i, val in zip(to_cpu_float_list(top_indices), to_cpu_float_list(top_values)):
                idx_int = int(idx_i)
                tok_str = tokenizer.decode([idx_int], clean_up_tokenization_spaces=False)
                top_entries.append({"id": idx_int, "prob": float(val), "token": tok_str})

            actual_token_id = ids_list[t] if t < len(ids_list) else None

            per_pos.append({
                "position": t,
                "topk": top_entries,
                "other_prob": other_prob,
                "actual_token_id": actual_token_id
            })

        subject = rec.get("subject") or rec.get("category") or rec.get("task", None)
        gold = rec.get("answer") or rec.get("label") or rec.get("gold", None)

        obj = {
            "source": "MMLU-Pro",
            "dataset_name": args.dataset_name,
            "split": args.split,
            "dataset_index": int(idx),
            "model_id": args.model_id,
            "top_k": k,
            "vocab_size": vocab_size,
            "tokenizer_name": tokenizer.name_or_path,
            "sequence_length": seq_len,
            "prompt_text": prompt_text,
            "input_ids": ids_list,
            "per_position": per_pos,
            "subject": subject,
            "option_count": len(option_labels),
            "gold_field": gold,
        }

        results.append(obj)
        print(f"[{count}/{args.n_samples}] processed item idx={idx} | seq_len={seq_len} | top-{k}")

    # Final atomic write
    out_path = Path(args.output)
    atomic_write_json(out_path, results, indent=2, ensure_ascii=False)
    print(f"Done. Output written to: {out_path.resolve()} (items: {len(results)})")


if __name__ == "__main__":
    main()
