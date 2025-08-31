#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Local model class
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM


@dataclass
class DecodeConfig:
    temperature: float
    top_p: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    max_new_tokens: int = 192
    do_sample: bool = True
    use_cache: bool = False  # ACT model: keep cache off


def parse_list(s: str, cast):
    return [cast(x) for x in s.split(",") if x.strip() != ""]


def grid_configs(
    temperatures: List[float],
    top_ps: List[float],
    repetition_penalties: List[float],
    no_repeat_ngram_sizes: List[int],
    max_new_tokens: int,
) -> List[DecodeConfig]:
    cfgs: List[DecodeConfig] = []
    for t in temperatures:
        for p in top_ps:
            for rp in repetition_penalties:
                for ng in no_repeat_ngram_sizes:
                    cfgs.append(
                        DecodeConfig(
                            temperature=t,
                            top_p=p,
                            repetition_penalty=rp,
                            no_repeat_ngram_size=ng,
                            max_new_tokens=max_new_tokens,
                        )
                    )
    return cfgs


def build_prompt(topic: str, wikitext: bool = False) -> str:
    topic = topic.strip()
    if not topic:
        return ""
    if wikitext:
        return f"'''{topic}''' is"
    else:
        return f"{topic}\n{topic} is"


def tokenize_for_ppl(tokenizer, prompt: str, completion: str, device) -> Dict[str, torch.Tensor]:
    # Measure NLL over completion tokens only; mask prompt tokens
    text = prompt + completion
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0]
    # Find split point by tokenizing prompt alone
    pre = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    cut = pre.shape[-1]
    labels = ids.clone()
    labels[:cut] = -100
    return {
        "input_ids": ids.unsqueeze(0).to(device),
        "attention_mask": torch.ones_like(ids).unsqueeze(0).to(device),
        "labels": labels.unsqueeze(0).to(device),
    }


def ngram_counts(tokens: List[int], n: int) -> Dict[Tuple[int, ...], int]:
    cnt: Dict[Tuple[int, ...], int] = {}
    for i in range(0, len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        cnt[ng] = cnt.get(ng, 0) + 1
    return cnt


def repetition_metrics(token_ids: List[int]) -> Dict[str, float]:
    res = {}
    for n in (2, 3, 4):
        if len(token_ids) >= n:
            c = ngram_counts(token_ids, n)
            uniq = len(c)
            total = max(1, len(token_ids) - n + 1)
            res[f"uniq_ng{n}_ratio"] = uniq / total
            repeats = sum(v - 1 for v in c.values() if v > 1)
            res[f"repeat_ng{n}"] = float(repeats)
        else:
            res[f"uniq_ng{n}_ratio"] = 1.0
            res[f"repeat_ng{n}"] = 0.0
    return res


def score_sample(
    ppl: float,
    rep: Dict[str, float],
    length: int,
    target_len: int,
    weights: Dict[str, float],
) -> float:
    # Lower perplexity is better; higher uniq_ng ratios are better; length close to target
    len_pen = abs(length - target_len) / max(1, target_len) if target_len > 0 else 0.0
    # Convert to bounded terms
    fluency = -math.log(max(ppl, 1e-6))  # higher is better
    diversity = 0.5 * rep.get("uniq_ng3_ratio", 1.0) + 0.5 * rep.get("uniq_ng4_ratio", 1.0)  # in [0,1]
    repetition = -(0.5 * rep.get("repeat_ng3", 0.0) + 0.5 * rep.get("repeat_ng4", 0.0))  # more repeats -> lower score
    length_term = -len_pen

    return (
        weights.get("fluency", 1.0) * fluency
        + weights.get("diversity", 0.5) * diversity
        + weights.get("repetition", 0.5) * repetition
        + weights.get("length", 0.1) * length_term
    )


def main():
    ap = argparse.ArgumentParser(description="Optimize decoding hyperparameters via local heuristics.")
    ap.add_argument("--model-dir", required=True, help="Path to checkpoint dir (with config.json and weights).")
    ap.add_argument("--prompts-file", required=True, help="File with one topic per line.")
    ap.add_argument("--out-csv", default="sampling_sweep.csv")
    ap.add_argument("--wikitext", action="store_true", help="Use wikitext-style prompts (triple quotes).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Grid definitions (comma-separated lists)
    ap.add_argument("--temperatures", default="0.5,0.6,0.7,0.8")
    ap.add_argument("--top-ps", default="0.85,0.9,0.92,0.95")
    ap.add_argument("--repetition-penalties", default="1.05,1.1,1.15")
    ap.add_argument("--no-repeat-ngram-sizes", default="3,4,5")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--per-config-samples", type=int, default=1, help="Samples per prompt per config.")
    ap.add_argument("--target-len", type=int, default=120, help="Target output tokens (approx).")
    ap.add_argument("--batch-size", type=int, default=8, help="Max generation batch size (may be smaller).")

    # Scoring weights
    ap.add_argument("--w-fluency", type=float, default=1.0)
    ap.add_argument("--w-diversity", type=float, default=0.5)
    ap.add_argument("--w-repetition", type=float, default=0.5)
    ap.add_argument("--w-length", type=float, default=0.1)

    # Optional separate judge LM for perplexity (defaults to same model)
    ap.add_argument("--judge-model-name-or-path", default=None)

    args = ap.parse_args()

    # Set random seed for reproducibility
    if args.seed != 0:
        set_seed(args.seed)

    device = torch.device(args.device)
    model = RecursiveHaltingMistralForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]

    temps = parse_list(args.temperatures, float)
    tps = parse_list(args.top_ps, float)
    reps = parse_list(args.repetition_penalties, float)
    ngrs = parse_list(args.no_repeat_ngram_sizes, int)
    cfgs = grid_configs(temps, tps, reps, ngrs, args.max_new_tokens)

    weights = {
        "fluency": args.w_fluency,
        "diversity": args.w_diversity,
        "repetition": args.w_repetition,
        "length": args.w_length,
    }

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = [
        "temperature",
        "top_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "max_new_tokens",
        "do_sample",
        "use_cache",
        "prompt",
        "completion",
        "tokens",
        "ppl",
        "score",
        "uniq_ng2_ratio",
        "uniq_ng3_ratio",
        "uniq_ng4_ratio",
        "repeat_ng2",
        "repeat_ng3",
        "repeat_ng4",
    ]

    best = None  # (score, config_dict)
    rows = []

    # Pipeline Stage 1: prepare all prompts per config
    per_samp = max(1, args.per_config_samples)
    all_batches = []  # list of tuples (cfg_d, prompts: List[str])
    total_samples = 0
    expected_pre_samples = len(cfgs) * len(topics) * per_samp
    with tqdm(total=expected_pre_samples, desc="Preparing", unit="sample", dynamic_ncols=True) as pbar0:
        for cfg in cfgs:
            cfg_d = asdict(cfg)
            prompts: List[str] = []
            for topic in topics:
                prompt = build_prompt(topic, wikitext=args.wikitext)
                if not prompt:
                    continue
                prompts.extend([prompt] * per_samp)
            if prompts:
                all_batches.append((cfg_d, prompts))
                total_samples += len(prompts)
                pbar0.update(len(prompts))

    # Pipeline Stage 2: batched generation per config
    samples = []  # will hold dicts with cfg and raw outputs for scoring later
    with tqdm(total=total_samples, desc="Generating", unit="sample", dynamic_ncols=True) as pbar:
        for cfg_d, prompts in all_batches:
            for i in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[i : i + args.batch_size]
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in enc.items()}

                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=cfg_d["max_new_tokens"],
                        do_sample=cfg_d["do_sample"],
                        temperature=cfg_d["temperature"],
                        top_p=cfg_d["top_p"],
                        repetition_penalty=cfg_d["repetition_penalty"],
                        no_repeat_ngram_size=cfg_d["no_repeat_ngram_size"],
                        length_penalty=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=cfg_d["use_cache"],
                    )

                # Decode each sample in the batch
                decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                for prompt_text, full_text in zip(batch_prompts, decoded):
                    completion = full_text[len(prompt_text) :] if full_text.startswith(prompt_text) else full_text
                    samples.append(
                        {
                            "cfg": cfg_d,
                            "prompt": prompt_text,
                            "completion": completion,
                        }
                    )
                pbar.update(len(batch_prompts))

    # Pipeline Stage 3: scoring/analysis
    judge_model = model
    judge_tokenizer = tokenizer
    if args.judge_model_name_or_path:
        # Unload the original model
        del model
        torch.cuda.empty_cache()

        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model_name_or_path,
            torch_dtype=torch.bfloat16,
        ).eval().to(device)
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_name_or_path)
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    with tqdm(total=len(samples), desc="Scoring", unit="sample", dynamic_ncols=True) as pbar2:
        for s in samples:
            prompt = s["prompt"]
            completion = s["completion"]
            cfg_d = s["cfg"]

            judge_in = tokenize_for_ppl(judge_tokenizer, prompt, completion, device)
            with torch.no_grad():
                out = judge_model(**judge_in, return_dict=True)
                loss = float(out.loss.detach().cpu())
                ppl = float(math.exp(min(20.0, max(-20.0, loss))))

            comp_ids = judge_tokenizer(completion, return_tensors="pt")["input_ids"][0].tolist()
            rep = repetition_metrics(comp_ids)

            score = score_sample(ppl=ppl, rep=rep, length=len(comp_ids), target_len=args.target_len, weights=weights)

            row = {
                **cfg_d,
                "prompt": prompt,
                "completion": completion,
                "tokens": len(comp_ids),
                "ppl": ppl,
                "score": score,
                **rep,
            }
            rows.append(row)
            pbar2.update(1)

    # Aggregate by config
    agg: Dict[Tuple, Dict[str, float]] = {}
    counts: Dict[Tuple, int] = {}
    for r in rows:
        key = (
            r["temperature"],
            r["top_p"],
            r["repetition_penalty"],
            r["no_repeat_ngram_size"],
            r["max_new_tokens"],
        )
        agg.setdefault(key, {"score_sum": 0.0, "ppl_sum": 0.0, "len_sum": 0.0})
        agg[key]["score_sum"] += r["score"]
        agg[key]["ppl_sum"] += r["ppl"]
        agg[key]["len_sum"] += r["tokens"]
        counts[key] = counts.get(key, 0) + 1

    best_key = None
    best_avg = -1e9
    for key, sums in agg.items():
        avg = sums["score_sum"] / max(1, counts[key])
        if avg > best_avg:
            best_avg, best_key = avg, key

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out_csv} with {len(rows)} rows.")
    if best_key is not None:
        t, p, rp, ng, mnt = best_key
        print("Best (avg score):")
        print(
            json.dumps(
                {
                    "temperature": t,
                    "top_p": p,
                    "repetition_penalty": rp,
                    "no_repeat_ngram_size": ng,
                    "max_new_tokens": mnt,
                    "avg_score": best_avg,
                    "trials": counts[best_key],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
