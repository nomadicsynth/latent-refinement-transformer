#!/usr/bin/env python
import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    MistralConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import WandbCallback
from trl import SFTConfig, SFTTrainer

from datasets import load_from_disk
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM
try:
    # tqdm for a nice progress bar over the sweep
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - best-effort fallback if tqdm isn't installed
    def tqdm(iterable, **kwargs):
        return iterable


class HaltingStatsCallback(TrainerCallback):
    """Injects ACT halting telemetry into the Trainer logs so W&B picks them up.

    We avoid direct W&B calls and rely on the Trainer logging API:
    - During training logs, we add: `act_inner_steps`, `act_expected_steps`.
    - During evaluation logs (identified by keys prefixed with "eval_"), we add
      `eval_act_inner_steps`, `eval_act_expected_steps`.
    """

    @staticmethod
    def _get_act_stats(model):
        inner = getattr(model, "_last_inner_steps", None)
        exp_mean = getattr(model, "_last_expected_steps_mean", None)
        # Normalize/validate types
        inner_v = int(inner) if inner is not None else None
        exp_v = float(exp_mean) if exp_mean is not None else None
        return inner_v, exp_v

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or logs is None:
            return control

        try:
            inner_v, exp_v = self._get_act_stats(model)
            if inner_v is None and exp_v is None:
                return control

            # Determine if this is eval logging (Transformers indicates eval with 'eval_' keys)
            is_eval = any(isinstance(k, str) and k.startswith("eval_") for k in logs.keys())

            # Use both underscore (Trainer-agnostic) and slash-prefixed (for W&B history display)
            if is_eval:
                if inner_v is not None:
                    logs["eval_act_inner_steps"] = inner_v
                    # logs["eval/act_inner_steps"] = inner_v
                if exp_v is not None:
                    logs["eval_act_expected_steps"] = exp_v
                    # logs["eval/act_expected_steps"] = exp_v
            else:
                if inner_v is not None:
                    logs["act_inner_steps"] = inner_v
                    # logs["train/act_inner_steps"] = inner_v
                if exp_v is not None:
                    logs["act_expected_steps"] = exp_v
                    # logs["train/act_expected_steps"] = exp_v
        except Exception:
            # Never break training due to telemetry
            pass

        return control

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        # Also inject ACT metrics into the eval metrics dict
        model = kwargs.get("model")
        if metrics is None or model is None:
            return control
        try:
            inner_v, exp_v = self._get_act_stats(model)
            if inner_v is not None:
                metrics["eval_act_inner_steps"] = inner_v
                # metrics["eval/act_inner_steps"] = inner_v
            if exp_v is not None:
                metrics["eval_act_expected_steps"] = exp_v
                # metrics["eval/act_expected_steps"] = exp_v
        except Exception:
            pass
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Try to inject final train ACT metrics into train summary if metrics dict is available.
        model = kwargs.get("model")
        metrics = kwargs.get("metrics")
        if model is None or not isinstance(metrics, dict):
            return control
        try:
            inner_v, exp_v = self._get_act_stats(model)
            if inner_v is not None:
                metrics["train_act_inner_steps"] = inner_v
            if exp_v is not None:
                metrics["train_act_expected_steps"] = exp_v
        except Exception:
            pass
        return control


class ACTWandbCallback(WandbCallback):
    """W&B callback that injects ACT metrics into logs before WandB processes them."""

    @staticmethod
    def _get_act_stats(model):
        inner = getattr(model, "_last_inner_steps", None)
        exp_mean = getattr(model, "_last_expected_steps_mean", None)
        inner_v = int(inner) if inner is not None else None
        exp_v = float(exp_mean) if exp_mean is not None else None
        return inner_v, exp_v

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        model = kwargs.get("model")
        if logs is not None and model is not None:
            try:
                inner_v, exp_v = self._get_act_stats(model)
                if inner_v is not None or exp_v is not None:
                    is_eval = any(isinstance(k, str) and k.startswith("eval_") for k in logs.keys())
                    if is_eval:
                        if inner_v is not None:
                            logs["eval_act_inner_steps"] = inner_v
                        if exp_v is not None:
                            logs["eval_act_expected_steps"] = exp_v
                    else:
                        if inner_v is not None:
                            logs["act_inner_steps"] = inner_v
                        if exp_v is not None:
                            logs["act_expected_steps"] = exp_v
            except Exception:
                pass
        return super().on_log(args, state, control, logs=logs, **kwargs)


@dataclass
class TrialResult:
    k_max: int
    tau: float
    lambda_ponder: float
    halting_mass_scale: float
    use_step_film: bool
    film_rank: int
    lambda_deep_supervision: float
    learning_rate: float
    steps_trained: Optional[int]
    eval_loss: Optional[float]
    eval_ppl: Optional[float]
    eval_mean_token_accuracy: Optional[float]
    best_eval_loss: Optional[float]
    best_eval_step: Optional[int]
    best_eval_mean_token_accuracy: Optional[float]
    best_eval_acc_step: Optional[int]
    best_train_loss: Optional[float]
    best_train_step: Optional[int]
    best_train_mean_token_accuracy: Optional[float]
    best_train_acc_step: Optional[int]
    train_runtime_s: float
    status: str
    error: Optional[str] = None
    output_dir: Optional[str] = None


def parse_float_list(csv_vals: str) -> List[float]:
    return [float(x.strip()) for x in csv_vals.split(",") if x.strip()]


def parse_int_list(csv_vals: str) -> List[int]:
    return [int(x.strip()) for x in csv_vals.split(",") if x.strip()]


def parse_bool_list(csv_vals: str) -> List[bool]:
    return [x.strip().lower() == "true" for x in csv_vals.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Sweep ACT hyperparameters (K_max, tau, lambda, LR).")
    p.add_argument("--dataset-path", default="./preprocessed_dataset_2184_227")
    p.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-Instruct-v0.3")

    # Base config (same small arch)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--intermediate-size", type=int, default=3688)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-attention-heads", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--attn-impl", default="flash_attention_2", choices=["sdpa", "eager", "flash_attention_2"])

    # Grid
    p.add_argument("--kmax", type=parse_int_list, default="2,4,6")
    p.add_argument("--tau", type=parse_float_list, default="0.95,0.98,0.99")
    p.add_argument("--lambda-ponder", type=parse_float_list, default="0.001,0.003,0.01")
    p.add_argument("--halting-mass-scale", type=parse_float_list, default="1.0")
    p.add_argument("--use-step-film", type=parse_bool_list, default="true")
    p.add_argument("--film-rank", type=parse_int_list, default="128")
    p.add_argument("--lambda-deep-supervision", type=parse_float_list, default="0.0")
    p.add_argument("--lrs", type=parse_float_list, default="1e-4,5e-4,1e-3")

    # Trainer
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--dataset-num-proc", type=int, default=4)
    p.add_argument(
        "--grad-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (slower but saves VRAM)",
    )
    p.add_argument("--compile", action="store_true", default=False, help="Use torch.compile for potential speedups")
    p.add_argument("--early-stopping", action="store_true", default=False, help="Enable early stopping on eval_loss")
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--early-stopping-threshold", type=float, default=0.0)
    p.add_argument("--packing", action="store_true", default=True)
    p.add_argument("--no-packing", dest="packing", action="store_false")

    # Data caps
    p.add_argument("--train-samples", type=int, default=30000)
    p.add_argument("--eval-samples", type=int, default=128)

    # Randomly sample a subset of the grid to cut total trials
    p.add_argument(
        "--sample-trials",
        type=int,
        default=0,
        help="If >0, randomly sample this many unique trial combos from the full grid",
    )

    # Output/logging
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb-project", default="science-llm")
    p.add_argument("--group", default="act-sweep")
    p.add_argument("--output-root", default="./results/hparam_act")
    p.add_argument("--csv-path", default="./results/hparam_act/results.csv")

    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_WATCH"] = "true"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_RUN_GROUP"] = args.group
    else:
        os.environ["WANDB_MODE"] = "disabled"

    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    train_ds = dataset["train"]
    eval_ds = dataset.get("test")
    if len(train_ds) > args.train_samples:
        train_ds = train_ds.select(range(args.train_samples))
    if eval_ds is not None and len(eval_ds) > args.eval_samples:
        eval_ds = eval_ds.select(range(args.eval_samples))

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tok.pad_token = tok.eos_token

    # Build full grid
    full_grid = [
        (K, t, l, hs, sf, fr, ds, lr)
        for K in args.kmax
        for t in args.tau
        for l in args.lambda_ponder
        for hs in args.halting_mass_scale
        for sf in args.use_step_film
        for fr in args.film_rank
        for ds in args.lambda_deep_supervision
        for lr in args.lrs
    ]
    print(f"Initial full grid size: {len(full_grid)}")

    # Remove items where `sf` is False and (`fr` is not 128 or ds is not 0.0)
    full_grid = [x for x in full_grid if not (x[4] == False and (x[5] != 128 or x[6] != 0.0))]
    print(f"Total grid size: {len(full_grid)}")

    # Randomly sample a subset of the grid to cut total trials
    if args.sample_trials and args.sample_trials > 0 and args.sample_trials < len(full_grid):
        grid = random.sample(full_grid, args.sample_trials)
    else:
        grid = full_grid
    total = len(grid)
    print(f"Total trials: {total}")

    results: List[TrialResult] = []
    for idx, (K, tau, lam, hs, sf, fr, ds, lr) in enumerate(
        tqdm(grid, total=total, desc="ACT sweep", dynamic_ncols=True), start=1
    ):
        run_name = f"K{K}-tau{tau}-lam{lam}-hs{hs}-sf{sf}-fr{fr}-ds{ds}-lr{lr:g}"
        out_dir = os.path.join(args.output_root, run_name)
        os.makedirs(out_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"[{idx}/{total}] {run_name}")
        print("=" * 80)
        start = time.time()
        try:
            # Initialize WandB run early with config, so custom hparams appear in the run config panel
            if args.use_wandb:
                import wandb

                wandb.init(
                    project=args.wandb_project,
                    group=args.group,
                    name=run_name,
                    reinit=True,
                    config={
                        "k_max": K,
                        "tau": tau,
                        "lambda_ponder": lam,
                        "halting_mass_scale": hs,
                        "use_step_film": sf,
                        "film_rank": fr,
                        "lambda_deep_supervision": ds,
                        "learning_rate": lr,
                        "hidden_size": args.hidden_size,
                        "intermediate_size": args.intermediate_size,
                        "num_layers": args.num_layers,
                        "num_attention_heads": args.num_attention_heads,
                        "num_kv_heads": args.num_kv_heads,
                        "attn_impl": args.attn_impl,
                        "per_device_train_batch_size": args.per_device_train_batch_size,
                        "per_device_eval_batch_size": args.per_device_eval_batch_size,
                        "grad_accum": args.grad_accum,
                        "warmup_ratio": args.warmup_ratio,
                        "max_grad_norm": args.max_grad_norm,
                        "weight_decay": args.weight_decay,
                        "bf16": args.bf16,
                        "max_length": args.max_length,
                        "eval_steps": args.eval_steps,
                        "logging_steps": args.logging_steps,
                        "max_steps": args.max_steps,
                        "packing": args.packing,
                        "train_samples": args.train_samples,
                        "eval_samples": args.eval_samples,
                        "dataset_path": args.dataset_path,
                    },
                )
            cfg = MistralConfig.from_pretrained(args.tokenizer_name)
            cfg.hidden_size = args.hidden_size
            cfg.intermediate_size = args.intermediate_size
            cfg.num_hidden_layers = args.num_layers
            cfg.num_attention_heads = args.num_attention_heads
            cfg.num_key_value_heads = args.num_kv_heads
            cfg._attn_implementation = args.attn_impl

            model = RecursiveHaltingMistralForCausalLM(
                cfg,
                k_max=K,
                tau=tau,
                lambda_ponder=lam,
                halting_mass_scale=hs,
                use_step_film=sf,
                film_rank=fr,
                lambda_deep_supervision=ds,
            ).to(device=device, dtype=(torch.bfloat16 if args.bf16 else torch.float32))
            if args.compile and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                except Exception:
                    pass

            sft = SFTConfig(
                output_dir=out_dir,
                eval_strategy="steps",
                eval_steps=args.eval_steps,
                eval_on_start=False,
                learning_rate=lr,
                lr_scheduler_type="cosine",
                warmup_ratio=args.warmup_ratio,
                max_grad_norm=args.max_grad_norm,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                gradient_accumulation_steps=args.grad_accum,
                gradient_checkpointing=args.grad_checkpointing,
                max_steps=args.max_steps,
                weight_decay=args.weight_decay,
                completion_only_loss=False,
                bf16=args.bf16,
                bf16_full_eval=args.bf16,
                max_length=args.max_length,
                logging_strategy="steps",
                logging_steps=args.logging_steps,
                save_strategy="no",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                # We'll attach WandB via explicit callback to control ordering
                report_to=("none" if args.use_wandb else "none"),
                dataset_num_proc=args.dataset_num_proc,
                eos_token=tok.eos_token,
                pad_token=tok.pad_token,
                packing=args.packing,
                dataset_kwargs={"skip_preprocessing": True},
                use_liger_kernel=False,
                run_name=run_name,
            )

            callbacks = []
            if args.early_stopping:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=args.early_stopping_patience,
                        early_stopping_threshold=args.early_stopping_threshold,
                    )
                )
            # Always add ACT halting stats logger
            callbacks.append(HaltingStatsCallback())
            # If using W&B, add our ACT-aware WandB callback
            if args.use_wandb:
                callbacks.append(ACTWandbCallback())

            trainer = SFTTrainer(
                model=model,
                args=sft,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                processing_class=tok,
                callbacks=callbacks if callbacks else None,
            )
            # Placeholders to record ACT telemetry snapshots
            train_act_inner = None
            train_act_expected = None
            eval_act_inner = None
            eval_act_expected = None

            out = trainer.train()

            train_runtime = time.time() - start

            steps_trained = None
            try:
                steps_trained = int(out.metrics.get("train_steps", 0))
            except Exception:
                pass
            # Snapshot ACT stats at end of training
            try:
                train_act_inner = int(getattr(model, "_last_inner_steps", None))
            except Exception:
                train_act_inner = None
            try:
                v = getattr(model, "_last_expected_steps_mean", None)
                train_act_expected = float(v) if v is not None else None
            except Exception:
                train_act_expected = None

            metrics = trainer.evaluate() if eval_ds is not None else {}
            # Snapshot ACT stats at end of evaluation
            try:
                eval_act_inner = int(getattr(model, "_last_inner_steps", None))
            except Exception:
                eval_act_inner = None
            try:
                v = getattr(model, "_last_expected_steps_mean", None)
                eval_act_expected = float(v) if v is not None else None
            except Exception:
                eval_act_expected = None
            loss = metrics.get("eval_loss")
            ppl = float(math.exp(loss)) if loss is not None else None
            eval_mean_token_accuracy = metrics.get("eval_mean_token_accuracy")

            # Scan log history for best eval/train losses and their steps
            best_eval_loss = None
            best_eval_step = None
            best_eval_mean_token_accuracy = None
            best_eval_acc_step = None
            best_train_loss = None
            best_train_step = None
            best_train_mean_token_accuracy = None
            best_train_acc_step = None
            try:
                for rec in trainer.state.log_history:
                    if isinstance(rec, dict):
                        if "eval_loss" in rec:
                            v = rec["eval_loss"]
                            s = rec.get("step", rec.get("global_step"))
                            if v is not None and (best_eval_loss is None or v < best_eval_loss):
                                best_eval_loss = float(v)
                                best_eval_step = int(s) if s is not None else None
                        # training loss is usually under 'loss'
                        if "loss" in rec:
                            v = rec["loss"]
                            s = rec.get("step", rec.get("global_step"))
                            if v is not None and (best_train_loss is None or v < best_train_loss):
                                best_train_loss = float(v)
                                best_train_step = int(s) if s is not None else None
                        if "eval_mean_token_accuracy" in rec:
                            v = rec["eval_mean_token_accuracy"]
                            s = rec.get("step", rec.get("global_step"))
                            if v is not None and (
                                best_eval_mean_token_accuracy is None or v > best_eval_mean_token_accuracy
                            ):
                                best_eval_mean_token_accuracy = float(v)
                                best_eval_acc_step = int(s) if s is not None else None
                        if "mean_token_accuracy" in rec:
                            v = rec["mean_token_accuracy"]
                            s = rec.get("step", rec.get("global_step"))
                            if v is not None and (
                                best_train_mean_token_accuracy is None or v > best_train_mean_token_accuracy
                            ):
                                best_train_mean_token_accuracy = float(v)
                                best_train_acc_step = int(s) if s is not None else None
            except Exception:
                pass

            # Log summaries to WandB if enabled
            try:
                import wandb

                if os.environ.get("WANDB_MODE") != "disabled":
                    wandb.run.summary["final_eval_loss"] = loss if loss is not None else float("nan")
                    wandb.run.summary["best_eval_loss"] = best_eval_loss if best_eval_loss is not None else float("nan")
                    wandb.run.summary["best_eval_step"] = best_eval_step if best_eval_step is not None else -1
                    if best_eval_mean_token_accuracy is not None:
                        wandb.run.summary["best_eval_mean_token_accuracy"] = best_eval_mean_token_accuracy
                    if best_eval_acc_step is not None:
                        wandb.run.summary["best_eval_acc_step"] = best_eval_acc_step
                    if best_train_loss is not None:
                        wandb.run.summary["best_train_loss"] = best_train_loss
                    if best_train_step is not None:
                        wandb.run.summary["best_train_step"] = best_train_step
                    if best_train_mean_token_accuracy is not None:
                        wandb.run.summary["best_train_mean_token_accuracy"] = best_train_mean_token_accuracy
                    if best_train_acc_step is not None:
                        wandb.run.summary["best_train_acc_step"] = best_train_acc_step
                    # Add ACT metrics to summary so they appear in the final Run summary
                    if train_act_inner is not None:
                        wandb.run.summary["train_act_inner_steps"] = train_act_inner
                    if train_act_expected is not None:
                        wandb.run.summary["train_act_expected_steps"] = train_act_expected
                    if eval_act_inner is not None:
                        wandb.run.summary["eval_act_inner_steps"] = eval_act_inner
                    if eval_act_expected is not None:
                        wandb.run.summary["eval_act_expected_steps"] = eval_act_expected
            except Exception:
                pass

            results.append(
                TrialResult(
                    k_max=K,
                    tau=tau,
                    lambda_ponder=lam,
                    halting_mass_scale=hs,
                    use_step_film=sf,
                    film_rank=fr,
                    lambda_deep_supervision=ds,
                    learning_rate=lr,
                    steps_trained=steps_trained,
                    eval_loss=(float(loss) if loss is not None else None),
                    eval_ppl=ppl,
                    eval_mean_token_accuracy=eval_mean_token_accuracy,
                    best_eval_loss=best_eval_loss,
                    best_eval_step=best_eval_step,
                    best_eval_mean_token_accuracy=best_eval_mean_token_accuracy,
                    best_eval_acc_step=best_eval_acc_step,
                    best_train_loss=best_train_loss,
                    best_train_step=best_train_step,
                    train_runtime_s=train_runtime,
                    status="ok",
                    output_dir=out_dir,
                )
            )
        except RuntimeError as e:
            err = str(e)
            print(f"Error in {run_name}: {err}", file=sys.stderr)
            results.append(
                TrialResult(
                    k_max=K,
                    tau=tau,
                    lambda_ponder=lam,
                    halting_mass_scale=hs,
                    use_step_film=sf,
                    film_rank=fr,
                    lambda_deep_supervision=ds,
                    learning_rate=lr,
                    steps_trained=None,
                    eval_loss=None,
                    eval_ppl=None,
                    eval_mean_token_accuracy=None,
                    best_eval_loss=None,
                    best_eval_step=None,
                    best_eval_mean_token_accuracy=None,
                    best_eval_acc_step=None,
                    best_train_loss=None,
                    best_train_step=None,
                    best_train_mean_token_accuracy=None,
                    best_train_acc_step=None,
                    train_runtime_s=train_runtime,
                    status="error",
                    error=err,
                    output_dir=out_dir,
                )
            )
        finally:
            try:
                import wandb

                if os.environ.get("WANDB_MODE") != "disabled":
                    wandb.finish(quiet=True)
            except Exception:
                pass
            try:
                del trainer
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()

    # Write CSV
    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k_max",
                "tau",
                "lambda_ponder",
                "halting_mass_scale",
                "use_step_film",
                "film_rank",
                "lambda_deep_supervision",
                "learning_rate",
                "steps_trained",
                "eval_loss",
                "eval_ppl",
                "eval_mean_token_accuracy",
                "best_eval_loss",
                "best_eval_step",
                "best_eval_mean_token_accuracy",
                "best_eval_acc_step",
                "best_train_loss",
                "best_train_step",
                "best_train_mean_token_accuracy",
                "best_train_acc_step",
                "train_runtime_s",
                "status",
                "error",
                "output_dir",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    oks = [r for r in results if r.status == "ok" and r.eval_loss is not None]
    oks.sort(key=lambda r: r.eval_loss)
    print("\nSearch complete.")
    for r in oks[:5]:
        print(
            f"K={r.k_max} tau={r.tau} lam={r.lambda_ponder} halting_mass_scale={r.halting_mass_scale} "
            f"use_step_film={r.use_step_film} film_rank={r.film_rank} lambda_deep_supervision={r.lambda_deep_supervision} "
            f"lr={r.learning_rate:g} "
            f"eval_loss={r.eval_loss:.4f} ppl={(r.eval_ppl if r.eval_ppl is not None else float('nan')):.2f} "
            f"eval_acc={(r.eval_mean_token_accuracy if r.eval_mean_token_accuracy is not None else float('nan')):.4f} "
            f"time={r.train_runtime_s:.1f}s"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting early due to Ctrl-C")
        try:
            import wandb

            if os.environ.get("WANDB_MODE") != "disabled":
                wandb.finish(quiet=True)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(0)
