#!/usr/bin/env python
"""Single-run trainer for ACT model."""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    MistralConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

from models.recursive_halting_mistral import (
    RecursiveHaltingMistralForCausalLM,
)
from transformers.integrations.integration_utils import WandbCallback


class HaltingStatsCallback(TrainerCallback):
    """Inject ACT halting telemetry into Trainer logs so report_to backends (e.g., W&B) see them."""

    @staticmethod
    def _get_act_stats(model):
        inner = getattr(model, "_last_inner_steps", None)
        exp_mean = getattr(model, "_last_expected_steps_mean", None)
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
class RunSummary:
    eval_loss: Optional[float]
    eval_ppl: Optional[float]
    eval_mean_token_accuracy: Optional[float]
    train_runtime_s: float
    steps_trained: Optional[int]
    act_inner_steps_train: Optional[int]
    act_expected_steps_train: Optional[float]
    act_inner_steps_eval: Optional[int]
    act_expected_steps_eval: Optional[float]


def _last_metric_from_logs(log_history, key: str):
    for rec in reversed(log_history):
        if isinstance(rec, dict) and key in rec and rec[key] is not None:
            try:
                return float(rec[key])
            except Exception:
                return rec[key]
    return None


def main():
    p = argparse.ArgumentParser(description="Train a single ACT configuration (no sweep).")
    # Data/model
    p.add_argument("--dataset-path", default="./preprocessed_dataset_2184_227")
    p.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--output-dir", default="./results/act_single")

    # Small model config (same as sweep defaults)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--intermediate-size", type=int, default=3688)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-attention-heads", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--attn-impl", default="flash_attention_2", choices=["sdpa", "eager", "flash_attention_2"])

    # ACT params
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.999)
    p.add_argument("--lambda-ponder", type=float, default=0.00251)
    p.add_argument("--halting-mass-scale", type=float, default=1.0)
    p.add_argument("--use-step-film", action="store_true", default=True)
    p.add_argument("--no-use-step-film", dest="use_step_film", action="store_false")
    p.add_argument("--film-rank", type=int, default=512)
    p.add_argument("--lambda-deep-supervision", type=float, default=0.06)

    # Trainer
    p.add_argument("--learning-rate", type=float, default=9e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1000, help="Set 0 to use epochs instead.")
    p.add_argument("--num-train-epochs", type=int, default=0, help="Used only if --max-steps=0")
    p.add_argument("--dataset-num-proc", type=int, default=4)
    p.add_argument("--grad-checkpointing", action="store_true", default=False)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--early-stopping", action="store_true", default=False)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--early-stopping-threshold", type=float, default=0.0)
    p.add_argument("--packing", action="store_true", default=True)
    p.add_argument("--no-packing", dest="packing", action="store_false")

    # Data caps
    p.add_argument("--train-samples", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=0)

    # W&B
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb-project", default="science-llm")
    p.add_argument("--group", default="act-single")
    p.add_argument("--run-name", default=None)

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_WATCH"] = "true"
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_RUN_GROUP"] = args.group
        # We'll attach our custom WandB callback manually to control ordering/metrics
        report_to = "none"
    else:
        os.environ["WANDB_MODE"] = "disabled"
        report_to = "none"

    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    train_ds = dataset["train"]
    eval_ds = dataset.get("test")
    if args.train_samples > 0 and len(train_ds) > args.train_samples:
        train_ds = train_ds.select(range(args.train_samples))
    if args.eval_samples > 0 and eval_ds is not None and len(eval_ds) > args.eval_samples:
        eval_ds = eval_ds.select(range(args.eval_samples))

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tok.pad_token = tok.eos_token

    cfg = MistralConfig.from_pretrained(args.tokenizer_name)
    cfg.hidden_size = args.hidden_size
    cfg.intermediate_size = args.intermediate_size
    cfg.num_hidden_layers = args.num_layers
    cfg.num_attention_heads = args.num_attention_heads
    cfg.num_key_value_heads = args.num_kv_heads
    cfg._attn_implementation = args.attn_impl

    model = RecursiveHaltingMistralForCausalLM(
        cfg,
        k_max=args.k_max,
        tau=args.tau,
        lambda_ponder=args.lambda_ponder,
        halting_mass_scale=args.halting_mass_scale,
        use_step_film=args.use_step_film,
        film_rank=args.film_rank,
        lambda_deep_supervision=args.lambda_deep_supervision,
    ).to(device=device, dtype=dtype)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    num_params = getattr(model, "num_parameters", None)
    if callable(num_params):
        trainable_params = num_params()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params >= 1e9:
        trainable_params_hr = f"{trainable_params / 1e9:.0f}B"
    elif trainable_params >= 1e6:
        trainable_params_hr = f"{trainable_params / 1e6:.0f}M"
    elif trainable_params >= 1e3:
        trainable_params_hr = f"{trainable_params / 1e3:.0f}K"
    else:
        trainable_params_hr = str(trainable_params)
    print(f"Trainable parameters: {trainable_params} ({trainable_params_hr})")

    save_strategy = "no"
    sft = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        eval_on_start=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_checkpointing,
        max_steps=(args.max_steps if args.max_steps and args.max_steps > 0 else -1),
        num_train_epochs=(args.num_train_epochs if (not args.max_steps) and args.num_train_epochs > 0 else 3),
        weight_decay=args.weight_decay,
        completion_only_loss=False,
        bf16=args.bf16,
        bf16_full_eval=args.bf16,
        max_length=args.max_length,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        dataset_num_proc=args.dataset_num_proc,
        eos_token=tok.eos_token,
        pad_token=tok.pad_token,
        packing=args.packing,
        dataset_kwargs={"skip_preprocessing": True},
        use_liger_kernel=False,  # Causes a crash. Haven't investigated why.
        run_name=(
            args.run_name
            or f"K{args.k_max}-tau{args.tau}-lam{args.lambda_ponder}-hs{args.halting_mass_scale}-sf{args.use_step_film}-fr{args.film_rank}-ds{args.lambda_deep_supervision}-lr{args.learning_rate:g}"
        ),
    )

    callbacks = [HaltingStatsCallback()]
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    if args.use_wandb:
        callbacks.append(ACTWandbCallback())

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=callbacks,
    )

    trainer.train()

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
    if eval_mean_token_accuracy is None:
        eval_mean_token_accuracy = _last_metric_from_logs(trainer.state.log_history, "eval_mean_token_accuracy")

    print("\nTraining complete.")
    if loss is not None:
        print(f"eval_loss={loss:.4f} ppl={(ppl if ppl is not None else float('nan')):.2f}")
    if eval_mean_token_accuracy is not None:
        print(f"eval_mean_token_accuracy={float(eval_mean_token_accuracy):.4f}")
    if train_act_inner is not None or train_act_expected is not None:
        print(f"train_act_inner_steps={train_act_inner} train_act_expected_steps={train_act_expected}")
    if eval_act_inner is not None or eval_act_expected is not None:
        print(f"eval_act_inner_steps={eval_act_inner} eval_act_expected_steps={eval_act_expected}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting early due to Ctrl-C")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(0)
