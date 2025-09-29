#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Dict, List, Tuple


# ---------- JSON loader (JSON only) ----------

def load_json_records(path: str) -> List[dict]:
    """Load a JSON file that is either a single object or an array of objects."""
    txt = Path(path).read_text(encoding="utf-8")
    obj = json.loads(txt)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"{path} must be a JSON object or array of objects.")


# ---------- Keying & alignment ----------

def rec_key(rec: dict) -> Tuple:
    """
    Alignment key priority:
      1) (dataset_name, split, dataset_index)
      2) prompt_text
      3) input_ids tuple
      4) fallback unique marker
    """
    ds = rec.get("dataset_name")
    sp = rec.get("split")
    idx = rec.get("dataset_index")
    if ds is not None and sp is not None and idx is not None:
        return ("dsidx", str(ds), str(sp), int(idx))
    pt = rec.get("prompt_text")
    if isinstance(pt, str):
        return ("prompt", pt)
    ids = rec.get("input_ids")
    if isinstance(ids, list):
        try:
            return ("ids", tuple(int(x) for x in ids))
        except Exception:
            pass
    return ("anon", id(rec))


def label_for(rec: dict) -> str:
    k = rec_key(rec)
    if k[0] == "dsidx":
        return f"{k[1]}:{k[2]}:{k[3]}"
    if k[0] == "prompt":
        return f"prompt:{(rec.get('prompt_text') or '')[:60].replace('\\n',' ')}"
    if k[0] == "ids":
        return f"ids:{len(rec.get('input_ids', []))}tok"
    return "item"


def align_records(P_list: List[dict], Q_list: List[dict]) -> List[Tuple[dict, dict, str]]:
    """
    Returns list of (P_rec, Q_rec, label). Alignment strategy:
      - If both singletons: pair them.
      - Else build maps by key and use intersection.
      - If no usable keys exist (all 'anon'), fall back to index-wise pairing up to min length (warn).
    """
    if len(P_list) == 1 and len(Q_list) == 1:
        recP, recQ = P_list[0], Q_list[0]
        return [(recP, recQ, label_for(recP))]

    def to_map(lst: List[dict]) -> Dict[Tuple, dict]:
        m = {}
        for r in lst:
            k = rec_key(r)
            if k not in m:  # keep first if duplicates
                m[k] = r
        return m

    P_map = to_map(P_list)
    Q_map = to_map(Q_list)

    keysP = set(P_map.keys())
    keysQ = set(Q_map.keys())
    common = [k for k in keysP & keysQ if k[0] != "anon"]

    if common:
        common.sort(key=lambda x: (x[0], str(x)))
        out = []
        for k in common:
            recP = P_map[k]; recQ = Q_map[k]
            out.append((recP, recQ, label_for(recP)))
        skipped_p = len(P_map) - len(common)
        skipped_q = len(Q_map) - len(common)
        if skipped_p or skipped_q:
            print(f"[INFO] Aligned {len(common)} items by keys (skipped P={skipped_p}, Q={skipped_q}).")
        return out

    # Fallback: index-wise pairing
    n = min(len(P_list), len(Q_list))
    if n == 0:
        return []
    print("[WARN] No alignment keys found; falling back to index-wise pairing.")
    return [(P_list[i], Q_list[i], f"idx:{i}") for i in range(n)]


# ---------- KL utilities ----------

def build_union_support(p_entry: dict, q_entry: dict):
    p_other = float(p_entry["other_prob"])
    q_other = float(q_entry["other_prob"])
    p_map = {int(it["id"]): float(it["prob"]) for it in p_entry["topk"]}
    q_map = {int(it["id"]): float(it["prob"]) for it in q_entry["topk"]}
    ids = sorted(set(p_map.keys()) | set(q_map.keys()))
    return ids, p_map, q_map, p_other, q_other


def kl_divergence(p_vec: List[float], q_vec: List[float], eps: float = 1e-12) -> float:
    p = [max(eps, float(x)) for x in p_vec]
    q = [max(eps, float(x)) for x in q_vec]
    ps = sum(p); qs = sum(q)
    p = [x / ps for x in p]
    q = [x / qs for x in q]
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))


def per_position_kl(P: dict, Q: dict, truncate_to_min: bool) -> List[float]:
    lenP = int(P.get("sequence_length", len(P["per_position"])))
    lenQ = int(Q.get("sequence_length", len(Q["per_position"])))
    if lenP != lenQ:
        if not truncate_to_min:
            raise ValueError(f"Sequence length mismatch: P={lenP}, Q={lenQ}. Use --truncate_to_min to allow min().")
        n = min(lenP, lenQ)
    else:
        n = lenP

    kls: List[float] = []
    for t in range(n):
        p_entry = P["per_position"][t]
        q_entry = Q["per_position"][t]
        ids, p_map, q_map, p_other, q_other = build_union_support(p_entry, q_entry)
        p_vec = [p_map.get(tok, 0.0) for tok in ids] + [p_other]
        q_vec = [q_map.get(tok, 0.0) for tok in ids] + [q_other]
        kls.append(kl_divergence(p_vec, q_vec))
    return kls


# ---------- Percentiles helper ----------

def pctiles(values: List[float]) -> Tuple[float, float, float]:
    """
    Returns (p25, p95, p99). Uses statistics.quantiles with 'inclusive' method.
    For very small lists (<2), falls back to the single value.
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = values[0]
        return v, v, v
    qs = quantiles(values, n=100, method="inclusive")  # 99 cut points for 1..99%
    p25 = qs[24]   # 25th percentile
    p95 = qs[94]   # 95th percentile
    p99 = qs[98]   # 99th percentile
    return p25, p95, p99


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Compute per-position KL(P‖Q) from JSON outputs (no JSONL).")
    ap.add_argument("--p_file", required=True, help="Path to P results (JSON object or array).")
    ap.add_argument("--q_file", required=True, help="Path to Q results (JSON object or array).")
    ap.add_argument("--truncate_to_min", action="store_true",
                    help="If lengths differ, compare up to min length instead of erroring.")
    ap.add_argument("--csv_out", default=None, help="Optional CSV path (item, position, kl_p_q).")
    ap.add_argument("--quiet_positions", action="store_true", help="Hide per-position prints.")
    args = ap.parse_args()

    P_list = load_json_records(args.p_file)
    Q_list = load_json_records(args.q_file)

    pairs = align_records(P_list, Q_list)
    if not pairs:
        raise SystemExit("No comparable items found between the two JSON files.")

    global_vals: List[float] = []
    csv_rows: List[Tuple[str, int, float]] = []

    for P, Q, label in pairs:
        kls = per_position_kl(P, Q, truncate_to_min=args.truncate_to_min)
        if not args.quiet_positions:
            print(f"\nKL(P‖Q) per position [{label}]:")
            for i, v in enumerate(kls):
                print(f"pos {i:4d}: {v:.6f}")

        _min, _med, _mean, _max = min(kls), median(kls), mean(kls), max(kls)
        p25, p95, p99 = pctiles(kls)
        print(
            f"Item summary [{label}]: "
            f"min={_min:.6f} p25={p25:.6f} median={_med:.6f} mean={_mean:.6f} "
            f"p95={p95:.6f} p99={p99:.6f} max={_max:.6f}"
        )

        global_vals.extend(kls)
        csv_rows.extend([(label, i, v) for i, v in enumerate(kls)])

    if global_vals:
        gmin, gmed, gmean, gmax = min(global_vals), median(global_vals), mean(global_vals), max(global_vals)
        gp25, gp95, gp99 = pctiles(global_vals)
        print("\n=== Global summary across all compared positions ===")
        print(
            f"min   : {gmin:.6f}\n"
            f"p25   : {gp25:.6f}\n"
            f"median: {gmed:.6f}\n"
            f"mean  : {gmean:.6f}\n"
            f"p95   : {gp95:.6f}\n"
            f"p99   : {gp99:.6f}\n"
            f"max   : {gmax:.6f}"
        )

    if args.csv_out and csv_rows:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "position", "kl_p_q"])
            w.writerows(csv_rows)
        print(f"Wrote CSV to: {args.csv_out}")


if __name__ == "__main__":
    main()
