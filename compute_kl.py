#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def load_result(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Basic integrity checks
    assert "per_position" in data and isinstance(data["per_position"], list)
    return data


def build_union_support(p_entry, q_entry):
    """
    Returns:
      ids: sorted list of token ids comprising union of P top-k and Q top-k
      maps: (p_map, q_map) of id->prob for the ids
      p_other, q_other: their respective OTHER masses
    """
    p_other = float(p_entry["other_prob"])
    q_other = float(q_entry["other_prob"])

    p_map = {int(it["id"]): float(it["prob"]) for it in p_entry["topk"]}
    q_map = {int(it["id"]): float(it["prob"]) for it in q_entry["topk"]}

    ids = sorted(set(p_map.keys()) | set(q_map.keys()))
    return ids, p_map, q_map, p_other, q_other


def kl_divergence(p_vec, q_vec, epsilon=1e-12):
    """
    Compute KL(P‖Q) for discrete distributions given as lists of probs on same support.
    Adds epsilon for numerical safety (and renormalizes).
    """
    # Add epsilon and renormalize
    p = [max(epsilon, float(x)) for x in p_vec]
    q = [max(epsilon, float(x)) for x in q_vec]
    p_sum = sum(p)
    q_sum = sum(q)
    p = [x / p_sum for x in p]
    q = [x / q_sum for x in q]

    kl = 0.0
    for pi, qi in zip(p, q):
        kl += pi * math.log(pi / qi)
    return kl


def main():
    parser = argparse.ArgumentParser(description="Compare two saved token distributions via per-position KL (P‖Q).")
    parser.add_argument("-p", required=True, help="JSON results from save_token_dists.py (treated as P).")
    parser.add_argument("-q", required=True, help="JSON results from save_token_dists.py (treated as Q).")
    parser.add_argument("--csv_out", default=None, help="Optional CSV path to dump per-position KLs.")
    args = parser.parse_args()

    P = load_result(args.p)
    Q = load_result(args.q)

    # Require the same tokenized prompt length to make positions comparable.
    lenP = int(P["sequence_length"])
    lenQ = int(Q["sequence_length"])
    if lenP != lenQ:
        raise ValueError(f"Sequence length mismatch: P={lenP}, Q={lenQ}. "
                         f"Ensure both ran on the same prompt and truncation settings.")

    n = lenP
    kl_per_pos = []

    for t in range(n):
        p_entry = P["per_position"][t]
        q_entry = Q["per_position"][t]

        ids, p_map, q_map, p_other, q_other = build_union_support(p_entry, q_entry)

        # Build vectors over union + OTHER bucket
        p_vec = [p_map.get(tok, 0.0) for tok in ids] + [p_other]
        q_vec = [q_map.get(tok, 0.0) for tok in ids] + [q_other]

        kl_t = kl_divergence(p_vec, q_vec)
        kl_per_pos.append(kl_t)

    # Summary stats
    kl_min = min(kl_per_pos) if kl_per_pos else float("nan")
    kl_med = median(kl_per_pos) if kl_per_pos else float("nan")
    kl_mean = mean(kl_per_pos) if kl_per_pos else float("nan")
    kl_max = max(kl_per_pos) if kl_per_pos else float("nan")

    print("KL(P‖Q) per position:")
    for i, v in enumerate(kl_per_pos):
        print(f"pos {i:4d}: {v:.6f}")

    print("\nSummary:")
    print(f"min   : {kl_min:.6f}")
    print(f"median: {kl_med:.6f}")
    print(f"mean  : {kl_mean:.6f}")
    print(f"max   : {kl_max:.6f}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["position", "kl_p_q"])
            for i, v in enumerate(kl_per_pos):
                w.writerow([i, v])
        print(f"Wrote CSV to: {args.csv_out}")


if __name__ == "__main__":
    main()
