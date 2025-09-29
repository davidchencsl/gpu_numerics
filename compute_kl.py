#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple, Any, Optional


# ---------- I/O helpers ----------

def _read_json_or_jsonl(path: str) -> List[dict]:
    """
    Returns a list of result objects.
    - JSON (single object) -> [obj]
    - JSONL (one object per line) -> [obj1, obj2, ...]
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # Heuristic: if multiple lines and each line looks like a JSON object, treat as JSONL
    if "\n" in txt:
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        objs = []
        for i, ln in enumerate(lines, 1):
            try:
                objs.append(json.loads(ln))
            except json.JSONDecodeError:
                # Fallback: if any line isn't valid JSON, try whole-file JSON
                objs = [json.loads(txt)]
                break
        return objs
    else:
        return [json.loads(txt)]


def load_results(path: str) -> List[dict]:
    objs = _read_json_or_jsonl(path)
    if not objs:
        return []
    # Basic integrity filtering
    cleaned = []
    for obj in objs:
        if "per_position" in obj and isinstance(obj["per_position"], list):
            cleaned.append(obj)
        else:
            # Allow older single-JSON that wraps in top-level fields
            # (already handled above) — here we only keep entries with per_position
            pass
    return cleaned


# ---------- Keying & alignment ----------

def _key_for_record(rec: dict) -> Tuple:
    """
    Construct an alignment key.
    Priority:
      1) (dataset_name, split, dataset_index) if present (MMLU-Pro path)
      2) prompt_text (exact)
      3) input_ids tuple (last resort)
    """
    ds = rec.get("dataset_name")
    sp = rec.get("split")
    idx = rec.get("dataset_index")
    if ds is not None and sp is not None and idx is not None:
        return ("dsidx", str(ds), str(sp), int(idx))
    if "prompt_text" in rec and isinstance(rec["prompt_text"], str):
        return ("prompt", rec["prompt_text"])
    if "input_ids" in rec and isinstance(rec["input_ids"], list):
        try:
            return ("ids", tuple(int(x) for x in rec["input_ids"]))
        except Exception:
            pass
    # Totally anonymous; fall back to object id
    return ("anon", id(rec))


def index_by_key(records: List[dict]) -> Dict[Tuple, dict]:
    return {_key_for_record(r): r for r in records}


# ---------- KL utils ----------

def build_union_support(p_entry: dict, q_entry: dict):
    p_other = float(p_entry["other_prob"])
    q_other = float(q_entry["other_prob"])
    p_map = {int(it["id"]): float(it["prob"]) for it in p_entry["topk"]}
    q_map = {int(it["id"]): float(it["prob"]) for it in q_entry["topk"]}
    ids = sorted(set(p_map.keys()) | set(q_map.keys()))
    return ids, p_map, q_map, p_other, q_other


def kl_divergence(p_vec: List[float], q_vec: List[float], epsilon: float = 1e-12) -> float:
    # Add epsilon and renormalize to ensure proper distributions
    p = [max(epsilon, float(x)) for x in p_vec]
    q = [max(epsilon, float(x)) for x in q_vec]
    ps = sum(p)
    qs = sum(q)
    p = [x / ps for x in p]
    q = [x / qs for x in q]
    kl = 0.0
    for pi, qi in zip(p, q):
        kl += pi * math.log(pi / qi)
    return kl


def per_position_kl(P: dict, Q: dict, truncate_to_min: bool = False) -> List[float]:
    lenP = int(P.get("sequence_length", len(P["per_position"])))
    lenQ = int(Q.get("sequence_length", len(Q["per_position"])))
    if lenP != lenQ:
        if not truncate_to_min:
            raise ValueError(f"Sequence length mismatch for item key: P={lenP}, Q={lenQ}. "
                             f"Use --truncate_to_min to allow min(lenP,lenQ).")
        n = min(lenP, lenQ)
    else:
        n = lenP

    out = []
    for t in range(n):
        p_entry = P["per_position"][t]
        q_entry = Q["per_position"][t]
        ids, p_map, q_map, p_other, q_other = build_union_support(p_entry, q_entry)
        p_vec = [p_map.get(tok, 0.0) for tok in ids] + [p_other]
        q_vec = [q_map.get(tok, 0.0) for tok in ids] + [q_other]
        out.append(kl_divergence(p_vec, q_vec))
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Compare per-position KL(P‖Q) for outputs from save_token_dists*.py")
    ap.add_argument("--p_file", required=True, help="Path to P results (JSON or JSONL).")
    ap.add_argument("--q_file", required=True, help="Path to Q results (JSON or JSONL).")
    ap.add_argument("--csv_out", default=None, help="Optional CSV path to dump per-position KLs (across items).")
    ap.add_argument("--truncate_to_min", action="store_true",
                    help="If lengths differ, compare up to min length instead of erroring.")
    ap.add_argument("--quiet_positions", action="store_true",
                    help="Suppress per-position prints; only show per-item + global summaries.")
    args = ap.parse_args()

    P_list = load_results(args.p_file)
    Q_list = load_results(args.q_file)

    if not P_list or not Q_list:
        raise SystemExit("One of the inputs has no valid records (missing 'per_position').")

    # Handle formats:
    # - If both have 1 record: compare directly.
    # - If both have many: align by key.
    # - If one has 1 and the other has many: try to align the singleton to a matching key; else error.
    global_kls: List[float] = []
    rows_for_csv: List[Tuple[str, int, float]] = []

    def label_from_rec(rec: dict) -> str:
        key = _key_for_record(rec)
        if key and key[0] == "dsidx":
            return f"{key[1]}:{key[2]}:{key[3]}"
        if key and key[0] == "prompt":
            return f"prompt:{(rec.get('prompt_text') or '')[:60].replace('\\n',' ')}"
        if key and key[0] == "ids":
            return f"ids:{len(rec.get('input_ids', []))}tok"
        return "item"

    if len(P_list) == 1 and len(Q_list) == 1:
        P = P_list[0]; Q = Q_list[0]
        item_label = label_from_rec(P)
        kls = per_position_kl(P, Q, truncate_to_min=args.truncate_to_min)
        if not args.quiet_positions:
            print(f"KL(P‖Q) per position ({item_label}):")
            for i, v in enumerate(kls):
                print(f"pos {i:4d}: {v:.6f}")
        _min, _med, _mean, _max = min(kls), median(kls), mean(kls), max(kls)
        print(f"\nItem summary [{item_label}]: min={_min:.6f} median={_med:.6f} mean={_mean:.6f} max={_max:.6f}")
        global_kls.extend(kls)
        rows_for_csv.extend([(item_label, i, v) for i, v in enumerate(kls)])
    else:
        P_idx = index_by_key(P_list)
        Q_idx = index_by_key(Q_list)

        # If one side is singleton, see if its key exists on the other side.
        if len(P_list) == 1 and len(Q_list) > 1:
            k = _key_for_record(P_list[0])
            if k in Q_idx:
                P_idx = {k: P_list[0]}
                Q_idx = {k: Q_idx[k]}
            else:
                raise SystemExit("Single-record P does not match any record in Q (by dataset index or prompt).")
        elif len(Q_list) == 1 and len(P_list) > 1:
            k = _key_for_record(Q_list[0])
            if k in P_idx:
                P_idx = {k: P_idx[k]}
                Q_idx = {k: Q_list[0]}
            else:
                raise SystemExit("Single-record Q does not match any record in P (by dataset index or prompt).")

        # Intersect keys
        common = list(P_idx.keys() & Q_idx.keys())
        if not common:
            raise SystemExit("No matching records between files. "
                             "Ensure both are JSONL from the same dataset indices or identical prompts.")

        skipped_p = len(P_idx) - len(common)
        skipped_q = len(Q_idx) - len(common)
        if skipped_p or skipped_q:
            print(f"[INFO] Aligning {len(common)} common items "
                  f"(skipped: P={skipped_p}, Q={skipped_q}).")

        for k in sorted(common, key=lambda x: (str(x), x)):
            P = P_idx[k]; Q = Q_idx[k]
            item_label = label_from_rec(P)
            try:
                kls = per_position_kl(P, Q, truncate_to_min=args.truncate_to_min)
            except Exception as e:
                print(f"[SKIP] {item_label}: {e}")
                continue

            if not args.quiet_positions:
                print(f"\nKL(P‖Q) per position [{item_label}]:")
                for i, v in enumerate(kls):
                    print(f"pos {i:4d}: {v:.6f}")

            _min, _med, _mean, _max = min(kls), median(kls), mean(kls), max(kls)
            print(f"Item summary [{item_label}]: min={_min:.6f} median={_med:.6f} mean={_mean:.6f} max={_max:.6f}")

            global_kls.extend(kls)
            rows_for_csv.extend([(item_label, i, v) for i, v in enumerate(kls)])

    # Global summary
    if global_kls:
        gmin, gmed, gmean, gmax = min(global_kls), median(global_kls), mean(global_kls), max(global_kls)
        print("\n=== Global summary across all compared positions ===")
        print(f"min   : {gmin:.6f}")
        print(f"median: {gmed:.6f}")
        print(f"mean  : {gmean:.6f}")
        print(f"max   : {gmax:.6f}")
    else:
        print("No KL values computed (nothing aligned).")

    if args.csv_out and rows_for_csv:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "position", "kl_p_q"])
            w.writerows(rows_for_csv)
        print(f"Wrote CSV to: {args.csv_out}")


if __name__ == "__main__":
    main()
