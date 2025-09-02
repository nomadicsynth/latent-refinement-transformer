#!/usr/bin/env python
"""Single-run trainer for ACT model."""

import argparse
import math
import json
import re
import shutil
import uuid
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
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
try:
    # Optional: Muon optimizer for hidden weights
    from muon import MuonWithAuxAdam  # type: ignore
    _MUON_AVAILABLE = True
except Exception:
    _MUON_AVAILABLE = False


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
    p = argparse.ArgumentParser(description="Train an ACT model.")
    # Model
    p.add_argument("--tokenizer-name", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--output-dir", default="./results/act_single")
    p.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        default=False,
        help="Allow writing into an existing output directory. Without this flag, the script exits if the directory exists.",
    )

    # Dataset
    p.add_argument("--dataset-path", default="./preprocessed_dataset_2184_227")
    p.add_argument("--packing", action="store_true", default=True)
    p.add_argument("--no-packing", dest="packing", action="store_false")
    p.add_argument("--skip-prepare-dataset", action="store_true", default=True)
    p.add_argument("--train-samples", type=int, default=0)
    p.add_argument("--eval-samples", type=int, default=0)

    # Small model config
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--intermediate-size", type=int, default=3688)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-attention-heads", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--attn-impl", default="flash_attention_2", choices=["sdpa", "eager", "flash_attention_2"])

    # ACT params
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.999)
    p.add_argument("--lambda-ponder", type=float, default=0.0025)
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
    p.add_argument("--eval-steps", type=float, default=100)
    p.add_argument("--eval-on-start", action="store_true", default=False)
    p.add_argument("--logging-steps", type=float, default=50)
    p.add_argument("--max-steps", type=int, default=-1, help="If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`. For a dataset smaller than `max_steps`, training is reiterated through the dataset until `max_steps` is reached.")
    p.add_argument("--num-train-epochs", type=float, default=3, help="Used only if --max-steps=-1")
    p.add_argument("--dataset-num-proc", type=int, default=4)
    p.add_argument("--grad-checkpointing", action="store_true", default=False)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--early-stopping", action="store_true", default=False)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--early-stopping-threshold", type=float, default=0.0)
    p.add_argument("--use-liger-kernel", action="store_true", default=False)
    p.add_argument("--no-use-liger-kernel", dest="use_liger_kernel", action="store_false")

    # Muon optimizer
    p.add_argument("--use-muon", action="store_true", default=True, help="Use Muon optimizer for hidden weights.")
    p.add_argument("--no-muon", dest="use_muon", action="store_false")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Learning rate for Muon (hidden weights).")
    p.add_argument("--aux-adam-lr", type=float, default=None, help="Learning rate for auxiliary AdamW groups (embeddings, heads, gains/biases). Defaults to --learning-rate.")
    p.add_argument("--aux-beta1", type=float, default=0.9, help="AdamW beta1 for auxiliary groups.")
    p.add_argument("--aux-beta2", type=float, default=0.95, help="AdamW beta2 for auxiliary groups.")

    # Saving / checkpoints
    p.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="no", help="Checkpoint saving strategy. 'no' disables checkpointing.")
    p.add_argument("--save-steps", type=float, default=100, help="Save a checkpoint every N steps when --save-strategy=steps. If float < 1, it will be treated as a fraction of the total steps.")
    p.add_argument("--save-total-limit", type=int, default=0, help="Maximum number of checkpoints to keep (0 disables limit). Older checkpoints are deleted.")
    p.add_argument("--load-best-model-at-end", action="store_true", default=False, help="After training, load the best checkpoint according to metric_for_best_model.")
    p.add_argument("--metric-for-best-model", default="eval_loss", help="Metric to use for selecting the best model.")
    p.add_argument("--save-final-model", action="store_true", default=True, help="After training/eval, save the final model to output_dir.")
    p.add_argument("--no-save-final-model", dest="save_final_model", action="store_false")

    # W&B
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb-project", default="science-llm")
    p.add_argument("--group", default="act-single")
    p.add_argument("--run-name", default=None)

    # Backups (redundant copy of saved final model)
    p.add_argument("--backup-root", default=os.environ.get("SCIENCE_LLM_BACKUP_ROOT"), help="Root directory where per-run backups are stored (e.g., your NTFS drive mount). Can also be set via SCIENCE_LLM_BACKUP_ROOT.")
    p.add_argument("--backup", dest="backup", action="store_true", default=None, help="Enable automatic backup of the final model to --backup-root. Defaults to True when --backup-root is set.")
    p.add_argument("--no-backup", dest="backup", action="store_false")

    args = p.parse_args()

    # Safety: require explicit consent to write into an existing output directory
    try:
        if os.path.isdir(args.output_dir) and not args.overwrite_output_dir:
            RED_BG = "\033[1;97;41m"  # bold white on red background
            YELLOW = "\033[1;33m"
            RESET = "\033[0m"
            abspath = os.path.abspath(args.output_dir)
            msg = [
                "",
                f"{RED_BG}{' ' * 74}{RESET}",
                f"{RED_BG}  OUTPUT DIRECTORY ALREADY EXISTS — REFUSING TO OVERWRITE.        {RESET}",
                f"{RED_BG}{' ' * 74}{RESET}",
                "",
                f"{YELLOW}Directory: {abspath}{RESET}",
                "To proceed and allow writing into this directory, pass --overwrite-output-dir.",
                "Alternatively, choose a new path via --output-dir.",
                "",
            ]
            try:
                print("\n".join(msg), file=sys.stderr)
            except Exception:
                print(
                    "\n".join(
                        [
                            "",
                            "*** OUTPUT DIRECTORY ALREADY EXISTS — REFUSING TO OVERWRITE. ***",
                            f"Directory: {abspath}",
                            "Pass --overwrite-output-dir to proceed, or choose a new --output-dir.",
                            "",
                        ]
                    ),
                    file=sys.stderr,
                )
            sys.exit(2)
    except Exception:
        # If the check fails for some reason, default to creating the directory safely below
        pass

    # Create the directory if it doesn't exist yet (or when overwrite was explicitly allowed)
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

    # Determine whether backups are active
    backup_root = args.backup_root
    backups_enabled = (args.backup if args.backup is not None else bool(backup_root)) and bool(backup_root)

    # Loud warning if backups are not enabled
    if not backups_enabled:
        RED_BG = "\033[1;97;41m"  # bold white on red background
        YELLOW = "\033[1;33m"
        RESET = "\033[0m"
        msg = [
            "",
            f"{RED_BG}{' ' * 74}{RESET}",
            f"{RED_BG}  WARNING: AUTO-BACKUP IS DISABLED — YOUR FINAL MODEL MAY NOT BE COPIED!  {RESET}",
            f"{RED_BG}{' ' * 74}{RESET}",
            "",
            f"{YELLOW}Set SCIENCE_LLM_BACKUP_ROOT or pass --backup-root <path> to enable backups.",
            f"Current output_dir: {os.path.abspath(args.output_dir)}{RESET}",
            f"To silence this warning explicitly, pass --no-backup.",
            "",
        ]
        try:
            print("\n".join(msg), file=sys.stderr)
        except Exception:
            # Fallback without ANSI
            print("\n".join([
                "",
                "*** WARNING: AUTO-BACKUP IS DISABLED — YOUR FINAL MODEL MAY NOT BE COPIED! ***",
                f"Set SCIENCE_LLM_BACKUP_ROOT or pass --backup-root <path> to enable backups.",
                f"Current output_dir: {os.path.abspath(args.output_dir)}",
                "To silence this warning explicitly, pass --no-backup.",
                "",
            ]), file=sys.stderr)

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

    # Persist ACT hyperparameters to the config so they serialize with checkpoints
    cfg.k_max = args.k_max
    cfg.tau = args.tau
    cfg.lambda_ponder = args.lambda_ponder
    cfg.halting_mass_scale = args.halting_mass_scale
    cfg.use_step_film = args.use_step_film
    cfg.film_rank = args.film_rank
    cfg.lambda_deep_supervision = args.lambda_deep_supervision

    # Disable KV cache because the model doesn't use it, yet...
    cfg.use_cache = False

    model = RecursiveHaltingMistralForCausalLM(cfg).to(device=device, dtype=dtype)

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

    # Save/Checkpoint strategy
    save_strategy = args.save_strategy

    dataset_kwargs = {}
    if args.skip_prepare_dataset:
        dataset_kwargs["skip_prepare_dataset"] = True

    # Build SFTConfig kwargs with conditional population
    sft_kwargs = {
        "output_dir": args.output_dir,
        "eval_steps": args.eval_steps,
        "eval_on_start": args.eval_on_start,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_checkpointing": args.grad_checkpointing,
        "weight_decay": args.weight_decay,
        "completion_only_loss": False,
        "bf16": args.bf16,
        "bf16_full_eval": args.bf16,
        "max_length": args.max_length,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": save_strategy,
        "save_steps": args.save_steps,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": False,
        "report_to": report_to,
        "dataset_num_proc": args.dataset_num_proc,
        "eos_token": tok.eos_token,
        "pad_token": tok.pad_token,
        "packing": args.packing,
        "eval_packing": args.packing,
        "dataset_kwargs": dataset_kwargs,
        "use_liger_kernel": args.use_liger_kernel,
    }

    # Conditional: evaluation strategy depends on eval dataset presence
    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = "steps"
    else:
        sft_kwargs["eval_strategy"] = "no"

    # Conditional: max steps vs epochs
    if args.max_steps and args.max_steps > 0:
        sft_kwargs["max_steps"] = args.max_steps
    else:
        sft_kwargs["max_steps"] = -1

    if args.max_steps == -1 and args.num_train_epochs > 0:
        sft_kwargs["num_train_epochs"] = args.num_train_epochs
    else:
        sft_kwargs["num_train_epochs"] = 3

    # Conditional: save_total_limit may be disabled with 0/None
    if args.save_total_limit and args.save_total_limit > 0:
        sft_kwargs["save_total_limit"] = args.save_total_limit
    else:
        sft_kwargs["save_total_limit"] = None

    # Conditional: run name provided or constructed
    if args.run_name:
        sft_kwargs["run_name"] = args.run_name
    else:
        sft_kwargs["run_name"] = (
            f"K{args.k_max}-tau{args.tau}-lam{args.lambda_ponder}-hs{args.halting_mass_scale}-"
            f"sf{args.use_step_film}-fr{args.film_rank}-ds{args.lambda_deep_supervision}-lr{args.learning_rate:g}"
        )

    sft = SFTConfig(**sft_kwargs)

    callbacks = []
    # Skip ACT telemetry when using Liger kernels (it breaks inner/expected steps)
    if not args.use_liger_kernel:
        callbacks.append(HaltingStatsCallback())

    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    if args.use_wandb:
        if not args.use_liger_kernel:
            callbacks.append(ACTWandbCallback())
        else:
            # Use plain WandB logging without ACT metric injection
            callbacks.append(WandbCallback())

    # Ensure torch.distributed is initialized for Muon if needed (Muon uses dist.get_world_size())
    if args.use_muon and _MUON_AVAILABLE and dist.is_available() and not dist.is_initialized():
        try:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            init_file = f"/tmp/muon_pg_{os.getpid()}"
            # Use a file-store to avoid requiring env:// vars in single-process
            dist.init_process_group(backend=backend, init_method=f"file://{init_file}", rank=0, world_size=1)
            print("Initialized torch.distributed (single-process) for Muon.")
        except Exception as e:
            print(f"WARNING: Could not initialize torch.distributed for Muon: {e}")

    # Optionally build a Muon optimizer with auxiliary AdamW for non-hidden params
    optimizers = (None, None)
    if args.use_muon:
        if not _MUON_AVAILABLE:
            print("WARNING: --use-muon set but Muon library not available. Falling back to default optimizer.")
        else:
            # Derive aux Adam LR if not provided
            aux_lr = args.aux_adam_lr if args.aux_adam_lr is not None else args.learning_rate

            # Define parameter groups:
            # - Hidden weights: tensors in the transformer body (model.model) with ndim >= 2 -> Muon
            # - Hidden gains/biases: tensors in the transformer body with ndim < 2 -> AdamW
            # - Non-hidden: embeddings, lm_head, ACT-specific heads/film/gates -> AdamW
            body = getattr(model, "model", None)
            if body is None:
                print("WARNING: Could not locate model body; skipping Muon.")
            else:
                def distinct(params):
                    # Remove Nones and preserve order while de-duplicating by id
                    seen = set()
                    out = []
                    for p in params:
                        if p is None:
                            continue
                        pid = id(p)
                        if pid not in seen:
                            seen.add(pid)
                            out.append(p)
                    return out

                body_params = list(body.parameters())
                hidden_weights = [p for p in body_params if getattr(p, "ndim", 0) >= 2 and p.requires_grad]
                hidden_gains_biases = [p for p in body_params if getattr(p, "ndim", 0) < 2 and p.requires_grad]

                nonhidden_params = []
                # Embeddings
                try:
                    nonhidden_params += list(body.embed_tokens.parameters())
                except Exception:
                    pass
                # LM head
                try:
                    nonhidden_params += list(model.lm_head.parameters())
                except Exception:
                    pass
                # ACT-specific heads and controls
                try:
                    nonhidden_params += list(model.stop_head.parameters())
                except Exception:
                    pass
                try:
                    if getattr(model, "step_film", None) is not None:
                        nonhidden_params += list(model.step_film.parameters())
                except Exception:
                    pass
                # step_gates is a Parameter
                try:
                    if hasattr(model, "step_gates") and isinstance(model.step_gates, torch.nn.Parameter):
                        nonhidden_params.append(model.step_gates)
                except Exception:
                    pass

                # Ensure we don't double-assign params: remove any nonhidden from hidden groups
                nonhidden_ids = {id(p) for p in nonhidden_params}
                hidden_weights = [p for p in hidden_weights if id(p) not in nonhidden_ids]
                hidden_gains_biases = [p for p in hidden_gains_biases if id(p) not in nonhidden_ids]

                hidden_weights = distinct(hidden_weights)
                hidden_gains_biases = distinct(hidden_gains_biases)
                nonhidden_params = distinct([p for p in nonhidden_params if p.requires_grad])

                param_groups = []
                if len(hidden_weights) > 0:
                    param_groups.append(
                        dict(
                            params=hidden_weights,
                            use_muon=True,
                            lr=float(args.muon_lr),
                            weight_decay=float(args.weight_decay),
                        )
                    )
                # Merge gains/biases into the aux Adam group with nonhidden
                aux_params = hidden_gains_biases + nonhidden_params
                if len(aux_params) > 0:
                    param_groups.append(
                        dict(
                            params=aux_params,
                            use_muon=False,
                            lr=float(aux_lr),
                            betas=(float(args.aux_beta1), float(args.aux_beta2)),
                            weight_decay=float(args.weight_decay),
                        )
                    )

                if len(param_groups) > 0:
                    try:
                        optimizer = MuonWithAuxAdam(param_groups)
                        # Pass optimizer; let Trainer build the LR scheduler from args
                        optimizers = (optimizer, None)
                        print(
                            f"Using Muon optimizer: {len(hidden_weights)} hidden-weight tensors with lr={args.muon_lr}, "
                            f"and {len(aux_params)} aux Adam params with lr={aux_lr}."
                        )
                    except Exception as e:
                        print(f"WARNING: Failed to initialize Muon optimizer; falling back. Error: {e}")

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=callbacks,
        optimizers=optimizers,
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
    # Suppress ACT telemetry prints when Liger kernels are enabled
    if not args.use_liger_kernel:
        if train_act_inner is not None or train_act_expected is not None:
            print(f"train_act_inner_steps={train_act_inner} train_act_expected_steps={train_act_expected}")
        if eval_act_inner is not None or eval_act_expected is not None:
            print(f"eval_act_inner_steps={eval_act_inner} eval_act_expected_steps={eval_act_expected}")

    # Optionally save final model (post-eval to include best model if reloaded)
    if args.save_final_model:
        try:
            print(f"Saving final model and tokenizer to {args.output_dir + '/final_model'} ...")
            trainer.save_model(args.output_dir + '/final_model')
            try:
                tok.save_pretrained(args.output_dir + '/final_model')
            except Exception:
                pass
        except Exception as e:
            print(f"WARNING: Failed to save final model: {e}")

    # Optional backup to secondary drive with per-run unique folder and W&B indexing
    def _sanitize_name(name: str) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "run"))
        return (name.strip("-._") or "run")[:120]

    def _maybe_get_wandb_info():
        info = {"enabled": bool(args.use_wandb), "project": args.wandb_project, "group": args.group}
        try:
            import wandb  # type: ignore

            run = wandb.run
            if run is not None:
                info.update({
                    "run_id": getattr(run, "id", None),
                    "name": getattr(run, "name", None),
                    "url": getattr(run, "url", None),
                    "entity": getattr(run, "entity", None),
                    "project": getattr(run, "project", info.get("project")),
                })
            return info, wandb
        except Exception:
            return info, None

    def _write_json(path: str, obj):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"WARNING: Failed to write JSON {path}: {e}")

    if backups_enabled:
        src_dir = args.output_dir

        if os.path.isdir(src_dir) and backup_root:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run_name = getattr(sft, "run_name", None) or args.run_name or "run"
            safe_name = _sanitize_name(str(run_name))
            wb_info, wb_mod = _maybe_get_wandb_info()
            suffix = f"wb-{wb_info.get('run_id')}" if wb_info.get("run_id") else f"uid-{uuid.uuid4().hex[:8]}"
            folder = f"{ts}_{safe_name}__{suffix}"
            backup_dir = os.path.join(backup_root, folder)
            try:
                os.makedirs(backup_dir, exist_ok=True)
                print(f"Backing up model from {src_dir} to {backup_dir} ...")
                # Copy directory tree into backup folder
                shutil.copytree(src_dir, os.path.join(backup_dir, os.path.basename(src_dir)), dirs_exist_ok=True)

                # Persist minimal metadata for indexing
                meta = {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "source_output_dir": os.path.abspath(args.output_dir),
                    "source_final_model_dir": os.path.abspath(src_dir),
                    "backup_root": os.path.abspath(backup_root),
                    "backup_dir": os.path.abspath(backup_dir),
                    "run_name": run_name,
                    "wandb": wb_info,
                    "metrics": {
                        "eval_loss": float(loss) if loss is not None else None,
                        "eval_ppl": float(ppl) if ppl is not None else None,
                        "eval_mean_token_accuracy": float(eval_mean_token_accuracy) if eval_mean_token_accuracy is not None else None,
                        "train_act_inner_steps": train_act_inner,
                        "train_act_expected_steps": train_act_expected,
                        "eval_act_inner_steps": eval_act_inner,
                        "eval_act_expected_steps": eval_act_expected,
                    },
                    "trainer": {
                        "global_step": getattr(trainer.state, "global_step", None),
                        "train_runtime": getattr(trainer.state, "train_runtime", None),
                    },
                    "args": vars(args),
                }
                _write_json(os.path.join(backup_dir, "backup_meta.json"), meta)

                # Also log the backup path to W&B summary
                if wb_mod is not None and wb_info.get("enabled"):
                    try:
                        wb_mod.run.summary["backup_dir"] = os.path.abspath(backup_dir)
                        wb_mod.run.summary["backup_folder_name"] = folder
                        wb_mod.run.summary["backup_final_model_subdir"] = os.path.basename(src_dir)
                        wb_mod.run.summary["backup_eval_loss"] = meta["metrics"]["eval_loss"]
                        wb_mod.run.summary["backup_eval_ppl"] = meta["metrics"]["eval_ppl"]
                    except Exception:
                        pass

                print(f"Backup complete: {backup_dir}")
            except Exception as e:
                print(f"WARNING: Backup failed to {backup_root}: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting early due to Ctrl-C")
    finally:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
