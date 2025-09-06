#!/usr/bin/env python
"""Train RecursiveHaltingMistralForCausalLM on MNIST classification via label token.

Classification framing:
  sequence: <bos> + 784 pixel tokens + <label-token>
  We feed the whole sequence except final label token for prediction (standard causal LM).
  label token ids: 259..268 mapping to digits 0..9.

Usage (quick test):
  python train_mnist_act.py --quick-test

Full run example:
  python train_mnist_act.py \
    --output-dir mnist-act \
    --epochs 3 \
    --per-device-train-batch-size 64 \
    --learning-rate 3e-4

Requires torchvision (will attempt import and raise helpful error if missing).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import MistralConfig, Trainer, TrainingArguments, set_seed

# Local model import
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM

# ACT telemetry / W&B callbacks (shared with main training script)
try:  # pragma: no cover - best effort import
    from train import (
        HaltingStatsCallback,  # type: ignore
        ACTWandbCallback,      # type: ignore
        WandbConfigUpdateCallback,  # type: ignore
    )
except Exception:  # Fallbacks if symbols missing
    HaltingStatsCallback = None  # type: ignore
    ACTWandbCallback = None  # type: ignore
    WandbConfigUpdateCallback = None  # type: ignore

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258  # not used for classification, reserved
LABEL_BASE_ID = 259  # 259..268 inclusive
VOCAB_SIZE = 269
MAX_SEQ_LEN = 1 + 28 * 28 + 1  # bos + pixels + label


def require_torchvision():
    try:
        import torchvision  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("torchvision required. Install with: pip install torchvision") from e


def image_to_tokens(img: torch.Tensor) -> List[int]:
    # img: [1,28,28], float 0..1
    pix = (img.squeeze() * 255).to(torch.uint8).flatten().tolist()
    return pix


class MNISTSeqClassification(Dataset):
    def __init__(self, split: str = "train", limit: int | None = None, cache: bool = True):
        require_torchvision()
        import torchvision
        import torchvision.transforms as T

        train_flag = split == "train"
        self.ds = torchvision.datasets.MNIST(root="data", train=train_flag, download=True, transform=T.ToTensor())
        self.items: List[torch.Tensor] = []
        self.labels: List[int] = []
        for i, (x, y) in enumerate(self.ds):
            if limit is not None and i >= limit:
                break
            tokens = [BOS_ID] + image_to_tokens(x) + [LABEL_BASE_ID + int(y)]
            self.items.append(torch.tensor(tokens, dtype=torch.long))
            self.labels.append(int(y))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.items[idx]
        return {
            "input_ids": ids,
            "labels": ids.clone(),  # LM loss will predict label token
            "attention_mask": torch.ones_like(ids),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    ibs, labs, masks = [], [], []
    for ex in batch:
        ids = ex["input_ids"]
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            pad = torch.full((pad_len,), PAD_ID, dtype=ids.dtype)
            ids_p = torch.cat([ids, pad])
        else:
            ids_p = ids
        ibs.append(ids_p)
        labs.append(torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=ids.dtype)]))
        masks.append((ids_p != PAD_ID).long())
    return {
        "input_ids": torch.stack(ibs),
        "labels": torch.stack(labs),
        "attention_mask": torch.stack(masks),
    }


@dataclass
class EvalMetrics:
    accuracy: float
    loss: float


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to only the tiny slice needed for accuracy metric to save VRAM.

    Hugging Face Trainer will call this on each eval step before storing predictions.
    We return only the 10-way digit logits for the (sequence_length-2) position (the
    timestep that predicts the final label token). This avoids keeping the full
    [batch, seq_len, vocab] tensor in memory across the entire evaluation loop.
    Returned tensor is already moved to CPU and detached.
    """
    import torch

    if isinstance(logits, tuple):  # unwrap potential tuple
        logits = logits[0]
    # logits: [B, L, V]; prediction for last label is at position L-2 (due to shift)
    with torch.no_grad():
        # Slice before moving to CPU to minimize transfer size
        needed = logits[:, -2, LABEL_BASE_ID : LABEL_BASE_ID + 10].detach().to("cpu")  # [B,10]
    return needed


def compute_metrics(eval_pred):
    """Compute accuracy given compact predictions.

    Supports two formats:
      1. New (with preprocess_logits_for_metrics): predictions shape [N,10] (numpy) of digit logits.
      2. Fallback legacy: predictions shape [N,L,V] full logits (will down-project on CPU).
    """
    import numpy as np

    predictions, labels = eval_pred

    # labels: full token ids sequences (numpy) shape [N, L]
    if isinstance(predictions, tuple):  # just in case
        predictions = predictions[0]

    if predictions.ndim == 2 and predictions.shape[1] == 10:
        # Already reduced digit logits
        digit_logits = predictions  # [N,10]
        # final label token id is last non -100 in each label sequence (should be last element)
        label_tokens = labels[:, -1]
    else:
        # Legacy path: full logits were provided. Reduce now (more memory hungry!).
        # logits shape [N,L,V]; need second-to-last position, digit slice
        digit_logits = predictions[:, -2, LABEL_BASE_ID : LABEL_BASE_ID + 10]
        label_tokens = labels[:, -1]

    gold_digits = label_tokens - LABEL_BASE_ID  # 0..9
    pred_digits = np.argmax(digit_logits, axis=-1)
    mask_valid = (label_tokens >= LABEL_BASE_ID) & (label_tokens < LABEL_BASE_ID + 10)
    if mask_valid.sum() == 0:
        acc = 0.0
    else:
        acc = float((gold_digits[mask_valid] == pred_digits[mask_valid]).mean())
    return {"accuracy": acc}


def build_model(args) -> RecursiveHaltingMistralForCausalLM:
    config = MistralConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size or args.hidden_size * 4,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=getattr(args, "kv_heads", None) or args.heads,
        max_position_embeddings=MAX_SEQ_LEN,
    )
    # ACT parameters
    config.k_max = args.k_max
    config.tau = args.tau
    config.lambda_ponder = args.lambda_ponder
    config.use_step_film = not args.no_step_film
    config.film_rank = args.film_rank
    config.lambda_deep_supervision = args.lambda_deep_supervision
    config.halting_mass_scale = args.halting_mass_scale
    model = RecursiveHaltingMistralForCausalLM(config)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/mnist-act")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-device-train-batch-size", type=int, default=64)
    p.add_argument("--per-device-eval-batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=None, help="Number of key/value heads (defaults to --heads if not set)")
    p.add_argument("--intermediate-size", type=int, default=None)

    # ACT hyperparameters
    p.add_argument("--k-max", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--lambda-ponder", type=float, default=1e-3)
    p.add_argument("--lambda-deep-supervision", type=float, default=0.0)
    p.add_argument("--film-rank", type=int, default=64)
    p.add_argument("--no-step-film", action="store_true")
    p.add_argument("--halting-mass-scale", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=None, help="Limit train examples (debug)")
    p.add_argument("--limit-eval", type=int, default=None, help="Limit eval examples (debug)")
    p.add_argument("--quick-test", action="store_true", help="Run tiny setup to verify script")

    # Memory / perf options
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if available")
    p.add_argument("--auto-batch-reduce", action="store_true", help="On OOM, halve batch and retry once")

    # W&B logging
    p.add_argument("--use-wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb-project", default="science-llm")
    p.add_argument("--wandb-group", default="mnist-act")
    p.add_argument("--wandb-run-name", default=None)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.quick_test:
        # Tiny configuration for fast CPU sanity check
        args.epochs = 1
        args.limit_train = 64
        args.limit_eval = 64
        args.logging_steps = 5
        args.eval_steps = 20
        args.save_steps = 10_000  # effectively disable
        args.hidden_size = 96
        args.layers = 2
        args.heads = 4
        args.k_max = 2
        args.tau = 0.85
        args.lambda_ponder = 1e-3
        if args.max_steps is None:
            args.max_steps = 30

    train_ds = MNISTSeqClassification("train", limit=args.limit_train)
    eval_ds = MNISTSeqClassification("test", limit=args.limit_eval)

    model = build_model(args)

    # Optional W&B setup (environment vars before Trainer instantiation)
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        os.environ["WANDB_WATCH"] = "true"  # log gradients/params summary
        os.environ["WANDB_LOG_MODEL"] = "false"  # we handle artifacts manually if desired
        report_to = ["wandb"]
    else:
        os.environ["WANDB_MODE"] = "disabled"
        report_to = ["none"]

    if args.grad_checkpoint:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:  # pragma: no cover
            print("Gradient checkpointing not supported on model instance.")

    if not args.no_compile:
        try:
            model = torch.compile(model)  # type: ignore
            print("Model compiled with torch.compile")
        except Exception as e:  # pragma: no cover
            print("torch.compile failed (continuing):", e)

    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        report_to=report_to,
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        label_names=["labels"],
        run_name=args.wandb_run_name,
    )
    training_args = TrainingArguments(**ta_kwargs)

    # Build callbacks
    callbacks = []
    # Always include halting telemetry if available
    if HaltingStatsCallback is not None:
        try:
            callbacks.append(HaltingStatsCallback())
        except Exception:
            pass
    # If W&B enabled, add ACT-specific W&B callback + config push
    if args.use_wandb:
        if ACTWandbCallback is not None:
            try:
                callbacks.append(ACTWandbCallback())
            except Exception:
                pass
        if WandbConfigUpdateCallback is not None:
            extra_cfg = {
                # Core model sizing
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "heads": args.heads,
                "kv_heads": args.kv_heads if args.kv_heads is not None else args.heads,
                # ACT hyperparams
                "k_max": args.k_max,
                "tau": args.tau,
                "lambda_ponder": args.lambda_ponder,
                "halting_mass_scale": args.halting_mass_scale,
                "film_rank": args.film_rank,
                "use_step_film": not args.no_step_film,
                "lambda_deep_supervision": args.lambda_deep_supervision,
                # Training knobs
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "batch_size_train": args.per_device_train_batch_size,
                "batch_size_eval": args.per_device_eval_batch_size,
                "grad_accum": args.gradient_accumulation_steps,
            }
            try:
                callbacks.append(WandbConfigUpdateCallback(extra_cfg))
            except Exception:
                pass

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_batch,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Eval:", eval_metrics)

    # If ACT stats available, print a concise summary
    if HaltingStatsCallback is not None:
        inner_v = getattr(model, "_last_inner_steps", None)
        exp_v = getattr(model, "_last_expected_steps_mean", None)
        if inner_v is not None or exp_v is not None:
            print(f"ACT summary: inner_steps={inner_v} expected_steps_mean={exp_v}")

    # Show a few predictions in quick test mode
    if args.quick_test:
        model.eval()
        with torch.no_grad():
            ex = eval_ds[0]["input_ids"][:-1]  # drop label token
            device = next(model.parameters()).device
            ex_dev = ex.to(device)
            attn = torch.ones_like(ex_dev)
            out = model(input_ids=ex_dev.unsqueeze(0), attention_mask=attn.unsqueeze(0))
            next_logits = out.logits[0, -1]
            label_slice = next_logits[LABEL_BASE_ID : LABEL_BASE_ID + 10]
            pred = int(torch.argmax(label_slice).item())
            print("Quick test sample predicted digit:", pred)
            print("Model ACT stats steps_mean=", getattr(model, "_last_expected_steps_mean", None))


if __name__ == "__main__":  # pragma: no cover
    main()
