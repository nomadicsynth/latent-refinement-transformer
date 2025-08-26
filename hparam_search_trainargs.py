#!/usr/bin/env python
import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM
from trl import SFTConfig, SFTTrainer


@dataclass
class TrialResult:
    learning_rate: float
    grad_accum: int
    epochs: int
    params: int
    steps_trained: Optional[int]
    eval_loss: Optional[float]
    eval_ppl: Optional[float]
    train_runtime_s: float
    status: str
    error: Optional[str] = None
    output_dir: Optional[str] = None


def parse_float_list(csv_vals: str) -> List[float]:
    return [float(x.strip()) for x in csv_vals.split(",") if x.strip()]


def parse_int_list(csv_vals: str) -> List[int]:
    return [int(x.strip()) for x in csv_vals.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Grid search for LR, grad accumulation, and epochs on fixed Mistral config.")

    # Dataset/model setup (mirrors train.py defaults)
    parser.add_argument("--dataset-path", default="./preprocessed_dataset_2184_227", help="Path to datasets.load_from_disk folder containing 'train' and 'test'.")
    parser.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF tokenizer name or path.")
    parser.add_argument("--model-config-name", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF config name or path to load base config.")

    # Fixed small model config from train.py
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3688)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--attn-impl", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"], help="Attention implementation to use.")

    # Search grid
    parser.add_argument("--lrs", type=parse_float_list, default="1e-4,2e-4,5e-4")
    parser.add_argument("--grad-accum", type=parse_int_list, default="2,4,8")
    parser.add_argument("--epochs", type=parse_int_list, default="3,4,5,6")

    # Trainer base args
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--packing", action="store_true", default=True)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-strategy", default="epoch", choices=["epoch", "steps"], help="Evaluation strategy.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Used when eval-strategy=steps.")
    parser.add_argument("--save-strategy", default="no", choices=["no", "epoch", "steps"], help="Checkpoint saving strategy.")

    # Limits to make sweeps faster (optional)
    parser.add_argument("--train-samples", type=int, default=512, help="Optional cap on number of train rows (None for all).")
    parser.add_argument("--eval-samples", type=int, default=128, help="Optional cap on number of eval rows (None for all).")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max training steps cap (overrides epochs if set).")

    # Output and logging
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", default="science-llm")
    parser.add_argument("--group", default="trainargs-sweep")
    parser.add_argument("--output-root", default="./results/hparam_trainargs", help="Root dir for run outputs.")
    parser.add_argument("--csv-path", default="./results/hparam_trainargs/results.csv")

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # WandB config
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_WATCH"] = "false"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_RUN_GROUP"] = args.group
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Optional subsampling to accelerate sweeps
    train_ds = dataset["train"]
    eval_ds = dataset.get("test", None)
    if args.train_samples is not None and len(train_ds) > args.train_samples:
        train_ds = train_ds.select(range(args.train_samples))
        print(f"Selected first {len(train_ds)} train samples for the sweep.")
    if eval_ds is not None and args.eval_samples is not None and len(eval_ds) > args.eval_samples:
        eval_ds = eval_ds.select(range(args.eval_samples))
        print(f"Selected first {len(eval_ds)} eval samples for the sweep.")

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    results: List[TrialResult] = []
    total_trials = len(args.lrs) * len(args.grad_accum) * len(args.epochs)
    print(f"Total trials: {total_trials}")

    trial_idx = 0
    for epochs in args.epochs:
        for ga in args.grad_accum:
            for lr in args.lrs:
                trial_idx += 1
                run_name = f"lr{lr:g}-ga{ga}-ep{epochs}"
                output_dir = os.path.join(args.output_root, run_name)
                os.makedirs(output_dir, exist_ok=True)

                print("\n" + "=" * 80)
                print(f"[{trial_idx}/{total_trials}] Starting trial: {run_name}")
                print("=" * 80)
                start = time.time()

                model = None
                trainer = None
                try:
                    # Build model config (fixed arch from train.py)
                    config = MistralConfig.from_pretrained(args.model_config_name)
                    config.hidden_size = args.hidden_size
                    config.intermediate_size = args.intermediate_size
                    config.num_hidden_layers = args.num_layers
                    config.num_attention_heads = args.num_attention_heads
                    config.num_key_value_heads = args.num_kv_heads
                    config._attn_implementation = args.attn_impl

                    model = MistralForCausalLM(config).to(device=device, dtype=(torch.bfloat16 if args.bf16 else torch.float32))
                    params = model.num_parameters()

                    training_args = SFTConfig(
                        output_dir=output_dir,
                        seed=args.seed,
                        eval_strategy=args.eval_strategy,
                        eval_steps=(args.eval_steps if args.eval_strategy == "steps" else None),
                        eval_on_start=True,
                        learning_rate=lr,
                        lr_scheduler_type="cosine",
                        per_device_train_batch_size=args.per_device_train_batch_size,
                        per_device_eval_batch_size=args.per_device_eval_batch_size,
                        gradient_accumulation_steps=ga,
                        gradient_checkpointing=True,
                        num_train_epochs=(epochs if args.max_steps is None else 0),
                        max_steps=(args.max_steps if args.max_steps is not None else -1),
                        weight_decay=args.weight_decay,
                        completion_only_loss=False,
                        bf16=args.bf16,
                        bf16_full_eval=args.bf16,
                        max_length=args.max_length,
                        logging_strategy="steps",
                        logging_steps=args.logging_steps,
                        save_strategy=args.save_strategy,
                        report_to=("wandb" if args.use_wandb else "none"),
                        dataset_num_proc=args.dataset_num_proc,
                        eos_token=tokenizer.eos_token,
                        pad_token=tokenizer.pad_token,
                        packing=args.packing,
                        dataset_kwargs={"skip_preprocessing": True},
                        use_liger_kernel=True,
                        run_name=run_name,
                    )

                    trainer = SFTTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_ds,
                        eval_dataset=eval_ds,
                        processing_class=tokenizer,
                    )

                    train_output = trainer.train()
                    steps_trained = None
                    try:
                        steps_trained = int(train_output.metrics.get("train_runtime", 0))  # fallback if step metric missing
                    except Exception:
                        steps_trained = None

                    metrics = trainer.evaluate() if eval_ds is not None else {}
                    eval_loss = metrics.get("eval_loss")
                    eval_ppl = float(math.exp(eval_loss)) if (eval_loss is not None) else None

                    results.append(
                        TrialResult(
                            learning_rate=lr,
                            grad_accum=ga,
                            epochs=epochs,
                            params=params,
                            steps_trained=steps_trained,
                            eval_loss=(float(eval_loss) if eval_loss is not None else None),
                            eval_ppl=eval_ppl,
                            train_runtime_s=time.time() - start,
                            status="ok",
                            output_dir=output_dir,
                        )
                    )

                except RuntimeError as e:
                    # Handle OOM or Flash-Attn issues gracefully
                    err = str(e)
                    print(f"Error in trial {run_name}: {err}", file=sys.stderr)
                    results.append(
                        TrialResult(
                            learning_rate=lr,
                            grad_accum=ga,
                            epochs=epochs,
                            params=(-1 if model is None else model.num_parameters()),
                            steps_trained=None,
                            eval_loss=None,
                            eval_ppl=None,
                            train_runtime_s=time.time() - start,
                            status="error",
                            error=err,
                            output_dir=output_dir,
                        )
                    )
                finally:
                    # Gracefully end WandB run if enabled
                    try:
                        if os.environ.get("WANDB_MODE") != "disabled":
                            import wandb
                            wandb.finish(quiet=True)
                    except Exception as e:
                        print(f"Warning: wandb.finish failed: {e}", file=sys.stderr)

                    # Cleanup VRAM between trials
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
    fieldnames = [
        "learning_rate",
        "grad_accum",
        "epochs",
        "params",
        "steps_trained",
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

    ok_results = [r for r in results if r.status == "ok" and r.eval_loss is not None]
    ok_results.sort(key=lambda r: r.eval_loss)

    print("\nSearch complete.")
    if ok_results:
        print("Top results (by eval_loss):")
        for r in ok_results[:5]:
            print(
                f"lr={r.learning_rate:g} ga={r.grad_accum} ep={r.epochs} params={r.params:,} "
                f"eval_loss={r.eval_loss:.4f} ppl={(r.eval_ppl if r.eval_ppl is not None else float('nan')):.2f} "
                f"time={r.train_runtime_s:.1f}s dir={r.output_dir}"
            )
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()
