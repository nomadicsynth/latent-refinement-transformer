#!/usr/bin/env python
import argparse
import math
import os
import sys
import time
import wandb
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM
from trl import SFTConfig, SFTTrainer


@dataclass
class TrialResult:
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    params: int
    max_steps: int
    eval_loss: Optional[float]
    eval_ppl: Optional[float]
    train_runtime_s: float
    status: str
    error: Optional[str] = None
    output_dir: Optional[str] = None


def parse_int_list(csv: str) -> List[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Grid search for attention heads and KV heads on custom Mistral config.")
    parser.add_argument("--dataset-path", default="././preprocessed_dataset_2184_227", help="Path to datasets.load_from_disk folder containing 'train' and 'test'.")
    parser.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF tokenizer name or path.")
    parser.add_argument("--model-config-name", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF config name or path to load base config.")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3688)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=parse_int_list, default="8,12,16,24,32", help="Comma-separated list of num_attention_heads to try.")
    parser.add_argument("--kv-heads", type=parse_int_list, default="1,2,4,8,16", help="Comma-separated list of num_key_value_heads to try.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    # WandB toggles
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", default="science-llm")
    parser.add_argument("--group", default="heads-kv-search")
    parser.add_argument("--output-root", default="./results/hparam_heads_kv", help="Root dir for run outputs.")
    parser.add_argument("--csv-path", default="./results/hparam_heads_kv/results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument("--attn-impl", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"], help="Attention implementation to use.")

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_WATCH"] = "false"
        os.environ["WANDB_LOG_MODEL"] = "false"
        # Group runs for easy comparison
        os.environ["WANDB_RUN_GROUP"] = args.group
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Build candidate grid with constraints
    def valid_combo(h: int, kv: int) -> bool:
        if h <= 0 or kv <= 0:
            return False
        if args.hidden_size % h != 0:
            return False
        if h % kv != 0:
            return False
        head_dim = args.hidden_size // h
        # Flash-Attn prefers multiples of 8 (or 16 on some GPUs)
        if head_dim % 8 != 0:
            return False
        return True

    combos = [(h, kv) for h in args.heads for kv in args.kv_heads if valid_combo(h, kv)]
    if not combos:
        print("No valid (num_attention_heads, num_key_value_heads) combos after applying constraints.", file=sys.stderr)
        sys.exit(2)

    print(f"Testing {len(combos)} combos: {combos}")

    results: List[TrialResult] = []

    for (h, kv) in combos:
        run_name = f"h{h}-kv{kv}"
        output_dir = os.path.join(args.output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Starting trial: {run_name}")
        print("=" * 80)
        start = time.time()

        try:
            config = MistralConfig.from_pretrained(args.model_config_name)
            config.hidden_size = args.hidden_size
            config.intermediate_size = args.intermediate_size
            config.num_hidden_layers = args.num_layers
            config.num_attention_heads = h
            config.num_key_value_heads = kv
            config._attn_implementation = args.attn_impl

            model = MistralForCausalLM(config).to(device=device, dtype=torch.bfloat16 if args.bf16 else torch.float32)
            params = model.num_parameters()
            head_dim = args.hidden_size // h

            training_args = SFTConfig(
                output_dir=output_dir,
                seed=args.seed,
                eval_strategy="steps",
                eval_steps=args.eval_steps,
                eval_on_start=True,
                learning_rate=args.learning_rate,
                lr_scheduler_type="cosine",
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=True,
                max_steps=args.max_steps,
                weight_decay=args.weight_decay,
                completion_only_loss=False,
                bf16=args.bf16,
                bf16_full_eval=args.bf16,
                max_length=args.max_length,
                logging_strategy="steps",
                logging_steps=args.logging_steps,
                save_strategy="no",
                report_to=("wandb" if args.use_wandb else "none"),
                dataset_num_proc=args.dataset_num_proc,
                eos_token=tokenizer.eos_token,
                pad_token=tokenizer.pad_token,
                packing=getattr(args, "packing", True),
                dataset_kwargs={"skip_preprocessing": True},
                use_liger_kernel=True,
                run_name=run_name,
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                processing_class=tokenizer,
            )

            trainer.train()
            metrics = trainer.evaluate()
            eval_loss = metrics.get("eval_loss")
            eval_ppl = float(math.exp(eval_loss)) if eval_loss is not None else None

            result = TrialResult(
                num_attention_heads=h,
                num_key_value_heads=kv,
                head_dim=head_dim,
                params=params,
                max_steps=args.max_steps,
                eval_loss=(float(eval_loss) if eval_loss is not None else None),
                eval_ppl=eval_ppl,
                train_runtime_s=time.time() - start,
                status="ok",
                output_dir=output_dir,
            )
            results.append(result)

        except RuntimeError as e:
            # Handle OOM or Flash-Attn issues gracefully
            err = str(e)
            print(f"Error in trial {run_name}: {err}", file=sys.stderr)
            results.append(
                TrialResult(
                    num_attention_heads=h,
                    num_key_value_heads=kv,
                    head_dim=(args.hidden_size // h),
                    params=-1,
                    max_steps=args.max_steps,
                    eval_loss=None,
                    eval_ppl=None,
                    train_runtime_s=time.time() - start,
                    status="error",
                    error=err,
                    output_dir=output_dir,
                )
            )
        finally:
            try:
                # Finish wandb run
                wandb.finish(0)
            except Exception as e:
                print(f"Error finishing wandb run: {e}", file=sys.stderr)

            # Free VRAM between trials
            try:
                del trainer
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()

    # Write CSV summary
    import csv

    fieldnames = [
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "params",
        "max_steps",
        "eval_loss",
        "eval_ppl",
        "train_runtime_s",
        "status",
        "error",
        "output_dir",
    ]

    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Print top-5 by eval_loss
    ok_results = [r for r in results if r.status == "ok" and r.eval_loss is not None]
    ok_results.sort(key=lambda r: r.eval_loss)

    print("\nSearch complete.")
    if ok_results:
        print("Top results (by eval_loss):")
        for r in ok_results[:5]:
            print(
                f"h={r.num_attention_heads:>2} kv={r.num_key_value_heads:>2} head_dim={r.head_dim:>3} "
                f"params={r.params:,} eval_loss={r.eval_loss:.4f} ppl={r.eval_ppl:.2f} "
                f"time={r.train_runtime_s:.1f}s dir={r.output_dir}"
            )
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()
