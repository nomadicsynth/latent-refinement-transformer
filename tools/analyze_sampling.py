#!/usr/bin/env python
import argparse
import csv
import json
from collections import defaultdict

GROUP_KEYS = [
    "temperature",
    "top_p",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "max_new_tokens",
]

def parse_float(d, k, default=None):
    try:
        return float(d[k])
    except Exception:
        return default

def parse_int(d, k, default=None):
    try:
        return int(float(d[k]))
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser(description="Analyze sampling sweep CSV and select top decoding configs.")
    ap.add_argument("--csv", required=True, help="Path to results/sampling_sweep.csv")
    ap.add_argument("--top-k", type=int, default=10, help="How many configs to print")
    ap.add_argument("--out-json", default=None, help="If set, write best config as JSON here")
    ap.add_argument("--min-len", type=int, default=0, help="Ignore rows with tokens < min-len")
    ap.add_argument("--sort-by", default="score", choices=["score", "ppl"], help="Primary ranking metric (avg)")
    ap.add_argument("--descending", action="store_true", default=True, help="Sort descending (True for score)")
    args = ap.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Basic sanity filters
            tokens = parse_int(r, "tokens", 0) or 0
            if tokens < args.min_len:
                continue
            # Require score and ppl fields
            if r.get("score") is None or r.get("ppl") is None:
                continue
            rows.append(r)

    if not rows:
        print("No rows after filtering. Check CSV path and content.")
        return

    agg = defaultdict(lambda: {"score_sum": 0.0, "ppl_sum": 0.0, "len_sum": 0.0, "n": 0, "example": None})
    for r in rows:
        key = tuple(
            r.get(k)
            for k in GROUP_KEYS
        )
        agg[key]["score_sum"] += parse_float(r, "score", 0.0) or 0.0
        agg[key]["ppl_sum"] += parse_float(r, "ppl", 0.0) or 0.0
        agg[key]["len_sum"] += parse_int(r, "tokens", 0) or 0
        agg[key]["n"] += 1
        if agg[key]["example"] is None and r.get("prompt") and r.get("completion"):
            agg[key]["example"] = {"prompt": r["prompt"], "completion": r["completion"]}

    def key_to_cfg(k_tuple):
        d = {GROUP_KEYS[i]: k_tuple[i] for i in range(len(GROUP_KEYS))}
        # cast to proper types
        d["temperature"] = float(d["temperature"])
        d["top_p"] = float(d["top_p"])
        d["repetition_penalty"] = float(d["repetition_penalty"])
        d["no_repeat_ngram_size"] = int(float(d["no_repeat_ngram_size"]))
        d["max_new_tokens"] = int(float(d["max_new_tokens"]))
        return d

    summary = []
    for k, v in agg.items():
        if v["n"] == 0:
            continue
        avg_score = v["score_sum"] / v["n"]
        avg_ppl = v["ppl_sum"] / v["n"]
        avg_len = v["len_sum"] / v["n"]
        summary.append(
            {
                **key_to_cfg(k),
                "avg_score": avg_score,
                "avg_ppl": avg_ppl,
                "avg_tokens": avg_len,
                "trials": v["n"],
                "example": v["example"],
            }
        )

    if not summary:
        print("No aggregated configs found.")
        return

    # Determine sort order
    if args.sort_by == "score":
        summary.sort(key=lambda d: d["avg_score"], reverse=args.descending)
    else:  # ppl (lower is better)
        summary.sort(key=lambda d: d["avg_ppl"], reverse=not args.descending)

    top = summary[: args.top_k]

    print(f"\nTop {len(top)} configs (sorted by avg_{args.sort_by}):")
    for i, s in enumerate(top, 1):
        print(
            f"{i:2d}. T={s['temperature']:.2f} top_p={s['top_p']:.2f} "
            f"rp={s['repetition_penalty']:.2f} ng={s['no_repeat_ngram_size']} "
            f"max_new={s['max_new_tokens']} | avg_score={s['avg_score']:.4f} "
            f"avg_ppl={s['avg_ppl']:.3f} avg_tokens={s['avg_tokens']:.1f} trials={s['trials']}"
        )
        if s.get("example"):
            ex = s["example"]
            # Print a short preview of the example completion
            completion = (ex["completion"][:160] + "…") if len(ex["completion"]) > 160 else ex["completion"]
            print(f"    eg: {completion}")

    best = top[0]
    if args.out_json:
        best_payload = {
            "temperature": best["temperature"],
            "top_p": best["top_p"],
            "repetition_penalty": best["repetition_penalty"],
            "no_repeat_ngram_size": best["no_repeat_ngram_size"],
            "max_new_tokens": best["max_new_tokens"],
            "do_sample": True,
            "use_cache": False,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, indent=2)
        print(f"\nSaved best config to {args.out_json}")

if __name__ == "__main__":
    main()