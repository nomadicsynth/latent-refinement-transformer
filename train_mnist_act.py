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
    --output-dir results/mnist-act \
    --epochs 3 \
    --per-device-train-batch-size 64 \
    --learning-rate 3e-4

Requires torchvision (will attempt import and raise helpful error if missing).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List
import math
import os
import random
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import MistralConfig, Trainer, TrainingArguments, set_seed, TrainerCallback

# Local model import
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM
from models.image_encoders import ConvStemEncoder, PatchEmbeddingEncoder, WaveletEncoder, SIRENImplicitEncoder
from models.neon_vision_processor import NeonVisionProcessor, NeonVisionConfig

try:
    # Optional: Muon optimizer for hidden weights
    from muon import MuonWithAuxAdam  # type: ignore
    _MUON_AVAILABLE = True
except Exception:
    _MUON_AVAILABLE = False

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
TOKENS_PER_SAMPLE = MAX_SEQ_LEN  # 1 + 784 + 1 = 786


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


class DynamicAugmentedMNISTSeqClassification(Dataset):
    """On-the-fly MNIST -> token sequence with internal multiplicity calculation.

    Pass token_param_ratio > 0 along with param_count to auto-compute multiplicity:
        multiplicity = ceil( ratio * param_count / (base_len * TOKENS_PER_SAMPLE) )
    Clamped to max_multiplicity. If an explicit override_multiplicity is given, it wins.
    """

    def __init__(
        self,
        *,
        split: str = "train",
        limit: int | None = None,
        enable_aug: bool = True,
        aug_degrees: float = 10.0,
        aug_translate: float = 0.10,
        # multiplicity control
        token_param_ratio: float = 0.0,
        param_count: int | None = None,
        override_multiplicity: int | None = None,
        max_multiplicity: int = 64,
        aug_prob: float = 1.0,
    ):
        require_torchvision()
        import torchvision

        self.split = split
        base_flag = split == "train"
        self.enable_aug = enable_aug and base_flag
        self.aug_degrees = aug_degrees
        self.aug_translate = aug_translate
        self.limit = limit
        self.aug_prob = aug_prob  # mutable by scheduler
        self._global_step = 0

        self.base = torchvision.datasets.MNIST(
            root="data",
            train=base_flag,
            download=True,
            transform=None,
        )
        full_len = len(self.base)
        if self.limit is not None:
            full_len = min(full_len, self.limit)
        self._base_len = full_len

        # Compute multiplicity
        if override_multiplicity is not None:
            multiplicity = max(1, int(override_multiplicity))
            reason = "override"
        else:
            if token_param_ratio > 0 and param_count is not None:
                target_tokens = token_param_ratio * param_count
                base_tokens = self._base_len * TOKENS_PER_SAMPLE
                multiplicity = int(math.ceil(target_tokens / base_tokens))
                if multiplicity < 1:
                    multiplicity = 1
                if multiplicity > max_multiplicity:
                    print(f"[dataset] Auto multiplicity {multiplicity} > max {max_multiplicity}; clamping.")
                    multiplicity = max_multiplicity
                reason = "auto-ratio"
            else:
                multiplicity = 1
                reason = "default"
        self.multiplicity = multiplicity
        print(f"[dataset] dynamic split={split} base_len={self._base_len} multiplicity={self.multiplicity} reason={reason} ratio={token_param_ratio} param_count={param_count}")

    def __len__(self):
        return self._base_len * self.multiplicity

    def _build_transform(self, seed: int):
        import torchvision.transforms as T
        import torch

        rng = random.Random(seed)
        # RandomAffine params sampled by torchvision internally; we only fix degrees & translate.
        transforms = []
        if self.enable_aug:
            transforms.append(
                T.RandomAffine(
                    degrees=self.aug_degrees,
                    translate=(self.aug_translate, self.aug_translate),
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0,
                )
            )
        transforms.append(T.ToTensor())
        return T.Compose(transforms)

    def __getitem__(self, idx: int):
        base_idx = idx % self._base_len
        variant_idx = idx // self._base_len
        img, label = self.base[base_idx]  # PIL image, int label
        # Deterministic seed per (base_idx, variant_idx) for reproducibility
        seed = (base_idx * 1315423911 + variant_idx * 2654435761) & 0xFFFFFFFF
        transform = self._build_transform(seed)
        img_t = transform(img)  # [1,28,28]
        # Optionally drop augmentation stochastically based on aug_prob for non-zero variants
        # We currently only randomize by deciding after transform if we keep augmented or raw.
        if self.enable_aug and variant_idx > 0 and self.aug_prob < 1.0:
            # re-fetch raw if we skip
            r = random.Random(seed ^ 0xABCDEF).random()
            if r > self.aug_prob:
                # Rebuild a transform without augmentation
                no_aug_seed = seed ^ 0x12345678
                no_aug_tf = self._build_transform(no_aug_seed)
                # Temporarily disable aug flags for clean sample
                prev = self.enable_aug
                self.enable_aug = False
                img_t = no_aug_tf(img)
                self.enable_aug = prev
        tokens = [BOS_ID] + image_to_tokens(img_t) + [LABEL_BASE_ID + int(label)]
        ids = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
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
    p.add_argument("--per-device-eval-batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-scheduler-type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=float, default=500)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
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

    # Aug / multiplicity control
    p.add_argument("--aug-degrees", type=float, default=10.0, help="Max rotation degrees (train only)")
    p.add_argument("--aug-translate", type=float, default=0.10, help="Max fractional translation (train only)")
    # Optional severity ramp starts (final targets are the above base values)
    p.add_argument("--aug-degrees-start", type=float, default=None, help="If set, linearly ramp rotation degrees from this value to --aug-degrees")
    p.add_argument("--aug-translate-start", type=float, default=None, help="If set, ramp translation fraction from this value to --aug-translate")
    p.add_argument("--aug-severity-ramp-steps", type=int, default=None, help="Steps over which to ramp augmentation severity (degrees/translate). Defaults to --aug-prob-ramp-steps if unset.")
    p.add_argument("--aug-severity-auto-fraction", type=float, default=None, help="If set and any *-start provided (and steps not set), auto-compute severity ramp steps as fraction * total_train_steps.")
    p.add_argument("--use-aug", action="store_true", default=False, help="Enable stochastic augmentation")
    p.add_argument("--no-use-aug", dest="use_aug", action="store_false", help="Disable stochastic augmentation")
    p.add_argument("--aug-multiplicity", type=int, default=None, help="Explicit dataset multiplicity (overrides ratio calc)")
    p.add_argument("--token-param-ratio", type=float, default=0.0,
                   help="Target tokens:params ratio (e.g. 20 for 20:1). 0 disables automatic expansion.")
    p.add_argument("--max-multiplicity", type=int, default=64,
                   help="Safety cap for automatic multiplicity.")
    p.add_argument("--include-embeddings-in-param-count", action="store_true",
                   help="If set, count all params (including embeddings) for ratio; else exclude lm_head & embeddings if possible.")
    # Per-step augmentation probability ramp (single-epoch friendly)
    p.add_argument("--aug-prob-start", type=float, default=None, help="If set with --aug-prob-end & --aug-prob-ramp-steps, linearly ramp aug probability from start to end over given steps.")
    p.add_argument("--aug-prob-end", type=float, default=None, help="See --aug-prob-start")
    p.add_argument("--aug-prob-ramp-steps", type=int, default=None, help="Total steps over which to ramp aug prob. After that it stays at end value.")
    p.add_argument("--aug-prob-static", type=float, default=1.0, help="Fallback static augmentation probability when ramp args not supplied.")
    p.add_argument("--aug-prob-auto-fraction", type=float, default=None,
                   help="If set (e.g. 0.7) and --aug-prob-start/--aug-prob-end provided but no ramp steps, auto-compute ramp steps as fraction * total_train_steps (after multiplicity).")

    # New image encoder args
    p.add_argument("--image-encoder", default="conv", choices=["conv","patch","wavelet","siren","raw"])
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--encoder-hidden", type=int, default=64)
    p.add_argument("--freeze-encoder", action="store_true")

    # Muon optimizer
    p.add_argument("--use-muon", action="store_true", default=True, help="Use Muon optimizer for hidden weights.")
    p.add_argument("--no-muon", dest="use_muon", action="store_false")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Learning rate for Muon (hidden weights).")
    p.add_argument("--aux-adam-lr", type=float, default=None, help="Learning rate for auxiliary AdamW groups (embeddings, heads, gains/biases). Defaults to --learning-rate.")
    p.add_argument("--aux-beta1", type=float, default=0.9, help="AdamW beta1 for auxiliary groups.")
    p.add_argument("--aux-beta2", type=float, default=0.95, help="AdamW beta2 for auxiliary groups.")

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

    # --- Build model first (need param count for dataset) ---
    model = build_model(args)
    # If using image encoder path (inputs_embeds), inform Trainer which main input to track
    # so token counting & Accelerate hooks don't warn.
    if getattr(args, 'image_encoder', 'raw') != 'raw':
        model.main_input_name = 'inputs_embeds'

    if args.include_embeddings_in_param_count:
        total_params = model.num_parameters()
    else:
        total_params = 0
        for n, p in model.named_parameters():
            if any(x in n for x in ["embed_tokens", "lm_head"]):
                continue
            total_params += p.numel()

    # Training dataset: always use dynamic class now (it will pick multiplicity=1 if ratio disabled & no aug)
    train_ds = DynamicAugmentedMNISTSeqClassification(
        split="train",
        limit=args.limit_train,
        enable_aug=args.use_aug,
        aug_degrees=args.aug_degrees_start if args.aug_degrees_start is not None else args.aug_degrees,
        aug_translate=args.aug_translate_start if args.aug_translate_start is not None else args.aug_translate,
        token_param_ratio=args.token_param_ratio,
        param_count=total_params,
        override_multiplicity=args.aug_multiplicity,
        max_multiplicity=args.max_multiplicity,
        aug_prob=args.aug_prob_static,
    )

    eval_ds = DynamicAugmentedMNISTSeqClassification(
        split="test",
        limit=args.limit_eval,
        enable_aug=False,
        aug_degrees=0.0,
        aug_translate=0.0,
        token_param_ratio=0.0,
        param_count=total_params,
        override_multiplicity=1,
    )

    # --- Auto-compute ramp steps if requested ---
    if (
        args.aug_prob_auto_fraction is not None
        and args.aug_prob_start is not None
        and args.aug_prob_end is not None
        and args.aug_prob_ramp_steps is None
    ):
        # total training examples (after multiplicity & limit)
        effective_train_examples = len(train_ds)
        batch = args.per_device_train_batch_size
        accum = args.gradient_accumulation_steps
        epochs = args.epochs if args.max_steps == -1 or args.max_steps is None else 1
        steps_per_epoch = math.ceil(effective_train_examples / (batch * accum))
        if args.max_steps is not None and args.max_steps > 0:
            total_steps = min(args.max_steps, steps_per_epoch * epochs)
        else:
            total_steps = steps_per_epoch * epochs
        ramp_steps = int(max(1, round(total_steps * args.aug_prob_auto_fraction)))
        args.aug_prob_ramp_steps = ramp_steps
        print(f"[aug] auto ramp steps computed: total_steps={total_steps} fraction={args.aug_prob_auto_fraction} ramp_steps={ramp_steps}")
    else:
        # still compute total_steps for potential severity auto fraction reuse
        effective_train_examples = len(train_ds)
        batch = args.per_device_train_batch_size
        accum = args.gradient_accumulation_steps
        epochs = args.epochs if args.max_steps == -1 or args.max_steps is None else 1
        steps_per_epoch = math.ceil(effective_train_examples / (batch * accum))
        if args.max_steps is not None and args.max_steps > 0:
            total_steps = min(args.max_steps, steps_per_epoch * epochs)
        else:
            total_steps = steps_per_epoch * epochs

    # Auto-compute severity ramp steps if requested
    any_severity_start = any(
        x is not None for x in [args.aug_degrees_start, args.aug_translate_start]
    )
    if (
        any_severity_start
        and args.aug_severity_ramp_steps is None
        and args.aug_severity_auto_fraction is not None
    ):
        sev_steps = int(max(1, round(total_steps * args.aug_severity_auto_fraction)))
        args.aug_severity_ramp_steps = sev_steps
        print(f"[aug] auto severity ramp steps computed: total_steps={total_steps} fraction={args.aug_severity_auto_fraction} ramp_steps={sev_steps}")
    # If no explicit severity ramp steps but severity starts provided, fall back to prob ramp steps
    if any_severity_start and args.aug_severity_ramp_steps is None and args.aug_prob_ramp_steps is not None:
        args.aug_severity_ramp_steps = args.aug_prob_ramp_steps
        print(f"[aug] severity ramp steps defaulting to prob ramp steps: {args.aug_severity_ramp_steps}")

    # === NEW: evaluation telemetry containers (captured by closures) ===
    halting_expected_steps_batches: List[float] = []
    per_eval_halting_snapshots: List[float] = []  # optional history (not strictly needed)

    def preprocess_logits_for_metrics(logits, labels):
        """
        Reduce logits to [B,10] digit slice and record current batch's expected halting steps.
        """
        import torch
        if isinstance(logits, tuple):
            logits = logits[0]
        with torch.no_grad():
            needed = logits[:, -2, LABEL_BASE_ID:LABEL_BASE_ID + 10].detach().to("cpu")
        # Record halting stats (None-safe)
        hs = getattr(model, "_last_expected_steps_mean", None)
        if hs is not None:
            halting_expected_steps_batches.append(float(hs))
        return needed

    def compute_metrics(eval_pred):
        """
        Returns:
          - accuracy (overall)
          - acc_0..acc_9 (per-class)
          - acc_macro (mean over classes with at least one sample)
          - ponder_expected_steps_mean (mean expected halting steps across eval batches)
          - cm_diag_sum / cm_total (implicit via accuracy already)
          - confusion_top (a compact string of top off-diagonal confusions)
        """
        import numpy as np
        predictions, labels = eval_pred  # predictions: [N,10] digit logits (numpy)

        digit_logits = predictions
        label_tokens = labels[:, -1]
        gold_digits = label_tokens - LABEL_BASE_ID  # 0..9
        pred_digits = np.argmax(digit_logits, axis=-1)

        valid_mask = (label_tokens >= LABEL_BASE_ID) & (label_tokens < LABEL_BASE_ID + 10)
        gold_digits_valid = gold_digits[valid_mask]
        pred_digits_valid = pred_digits[valid_mask]

        if gold_digits_valid.size == 0:
            acc = 0.0
        else:
            acc = float((gold_digits_valid == pred_digits_valid).mean())

        # Confusion matrix
        cm = np.zeros((10, 10), dtype=np.int64)
        for g, p in zip(gold_digits_valid, pred_digits_valid):
            if 0 <= g < 10 and 0 <= p < 10:
                cm[g, p] += 1

        per_class_acc = {}
        class_accs = []
        for c in range(10):
            row_sum = cm[c].sum()
            if row_sum > 0:
                acc_c = cm[c, c] / row_sum
                class_accs.append(acc_c)
            else:
                acc_c = 0.0
            per_class_acc[f"acc_{c}"] = float(acc_c)

        acc_macro = float(np.mean(class_accs)) if class_accs else 0.0

        # Compact confusion summary (top 5 off-diagonal pairs by count)
        off_pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] > 0:
                    off_pairs.append((cm[i, j], i, j))
        off_pairs.sort(reverse=True)
        top_confusions = ";".join(f"{i}->{j}:{n}" for n, i, j in off_pairs[:5])

        # Halting expected steps aggregation
        if halting_expected_steps_batches:
            ponder_mean = float(np.mean(halting_expected_steps_batches))
            ponder_median = float(np.median(halting_expected_steps_batches))
        else:
            ponder_mean = 0.0
            ponder_median = 0.0

        # Preserve history (optional)
        if ponder_mean > 0:
            per_eval_halting_snapshots.append(ponder_mean)

        # After consuming, clear for next eval cycle
        halting_expected_steps_batches.clear()

        metrics = {
            "accuracy": acc,
            "acc_macro": acc_macro,
            "ponder_expected_steps_mean": ponder_mean,
            "ponder_expected_steps_median": ponder_median,
            "confusion_top": top_confusions,
        }
        metrics.update(per_class_acc)
        return metrics

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
        # torch.compile can add overhead for very short quick-test runs or custom embed paths
        if getattr(model, 'main_input_name', '') == 'inputs_embeds' and args.quick_test:
            print("[compile] Skipping torch.compile for quick_test with inputs_embeds path.")
        else:
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
        lr_scheduler_type=args.lr_scheduler_type,
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
        include_num_input_tokens_seen=True,
    )
    training_args = TrainingArguments(**ta_kwargs)

    # (Initial callback list construction removed; rebuilt after defining ramp callback.)

    # Augmentation probability scheduler callback (per-step linear ramp)
    class AugRampCallback(TrainerCallback):
        def __init__(self):
            # Probability ramp
            self.prob_start = args.aug_prob_start
            self.prob_end = args.aug_prob_end
            self.prob_total = args.aug_prob_ramp_steps
            self.prob_active = (
                self.prob_start is not None and self.prob_end is not None and self.prob_total is not None and self.prob_total > 0
            )
            if self.prob_active:
                print(f"[aug] prob ramp active: start={self.prob_start} end={self.prob_end} steps={self.prob_total}")
            elif self.prob_start is not None or self.prob_end is not None or self.prob_total is not None:
                print("[aug] incomplete prob ramp args provided; using static aug_prob_static")

            # Severity ramp (degrees / translate)
            self.deg_start = args.aug_degrees_start
            self.deg_end = args.aug_degrees if args.aug_degrees_start is not None else None  # Only ramp if start provided
            self.tr_start = args.aug_translate_start
            self.tr_end = args.aug_translate if args.aug_translate_start is not None else None
            self.sev_total = args.aug_severity_ramp_steps
            # Active if at least one pair start/end provided and steps positive
            self.sev_active = (
                any(v is not None for v in [self.deg_start, self.tr_start])
                and self.sev_total is not None and self.sev_total > 0
            )
            if self.sev_active:
                print(f"[aug] severity ramp active: steps={self.sev_total} deg:{self.deg_start}->{self.deg_end} tr:{self.tr_start}->{self.tr_end}")
            elif any(v is not None for v in [self.deg_start, self.tr_start]) and not self.sev_active:
                print("[aug] severity ramp incomplete (missing steps); using start values static")

        def on_init_end(self, args_, state, control, **kwargs):
            # Initialize augmentation probability
            if self.prob_active:
                train_ds.aug_prob = float(self.prob_start)
            else:
                train_ds.aug_prob = args.aug_prob_static
            # Initialize severity already set at dataset construction (start values) so nothing needed here.
            return control

        def on_step_begin(self, args_, state, control, **kwargs):
            step = state.global_step
            warmup = step < args.warmup_steps
            log_now = (step % max(1, args.logging_steps) == 0)

            # Probability ramp update
            if self.prob_active:
                if warmup:
                    train_ds.aug_prob = 0.0
                else:
                    if step >= self.prob_total:
                        prob = float(self.prob_end)
                    else:
                        alpha_p = step / max(1, self.prob_total)
                        prob = float(self.prob_start + (self.prob_end - self.prob_start) * alpha_p)
                    train_ds.aug_prob = prob

            # Severity ramp update
            if self.sev_active and not warmup:
                alpha_s = 1.0 if step >= self.sev_total else step / max(1, self.sev_total)
                if self.deg_start is not None and self.deg_end is not None:
                    train_ds.aug_degrees = float(self.deg_start + (self.deg_end - self.deg_start) * alpha_s)
                if self.tr_start is not None and self.tr_end is not None:
                    train_ds.aug_translate = float(self.tr_start + (self.tr_end - self.tr_start) * alpha_s)

            if log_now and state.global_step > 0:
                # Trainer can log scalars via control if using self.log; simpler: use callback hook
                if hasattr(trainer, "log"):
                    trainer.log({
                        "train/aug_prob": train_ds.aug_prob,
                        "train/aug_translate": train_ds.aug_translate,
                        "train/global_step": state.global_step
                    })

            return control

    # Insert ramp callback first so other callbacks see updated aug_prob
    ramp_cb = AugRampCallback()
    # Build callbacks
    callbacks = [ramp_cb]
    if HaltingStatsCallback is not None:
        try:
            callbacks.append(HaltingStatsCallback())
        except Exception:
            pass
    if args.use_wandb:
        if ACTWandbCallback is not None:
            try:
                callbacks.append(ACTWandbCallback())
            except Exception:
                pass
        if WandbConfigUpdateCallback is not None:
            extra_cfg = {
                "aug_prob_static": args.aug_prob_static,
                # Muon and aux optimizer knobs
                "use_muon": bool(args.use_muon),
                "muon_lr": args.muon_lr,
                "muon-lr": args.muon_lr,
                "aux_adam_lr": (args.aux_adam_lr if args.aux_adam_lr is not None else args.learning_rate),
                "aux-adam-lr": (args.aux_adam_lr if args.aux_adam_lr is not None else args.learning_rate),
                "aux_beta1": args.aux_beta1,
                "aux-beta1": args.aux_beta1,
                "aux_beta2": args.aux_beta2,
                "aux-beta2": args.aux_beta2,
            }
            try:
                callbacks.append(WandbConfigUpdateCallback(extra_cfg))
            except Exception:
                pass

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
                        # Optional: surface group sizes in W&B summary for quick visibility
                        try:
                            import wandb  # type: ignore
                            if getattr(wandb, "run", None) is not None:
                                wandb.run.summary["muon_hidden_weight_tensors"] = len(hidden_weights)
                                wandb.run.summary["muon_aux_param_count"] = sum(p.numel() for p in aux_params)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"WARNING: Failed to initialize Muon optimizer; falling back. Error: {e}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_batch,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks if callbacks else None,
        optimizers=optimizers,
    )

    # Build NeonVisionProcessor (unified interface) for embedding path if not raw
    if args.image_encoder != "raw":
        vision_cfg = NeonVisionConfig(
            encoder_type=args.image_encoder,
            image_size=28,
            in_channels=1,
            patch_size=args.patch_size,
            hidden_size=model.config.hidden_size,
            conv_hidden=args.encoder_hidden,
            siren_hidden=args.encoder_hidden,
        )
        vision_processor = NeonVisionProcessor(config=vision_cfg)
        if args.freeze_encoder and vision_processor.encoder is not None:
            for p_ in vision_processor.encoder.parameters():
                p_.requires_grad = False
    else:
        vision_processor = None

    # Wrap Trainer data flow: create a custom data_collator producing inputs_embeds
    def collate_batch_enc(batch):
        if vision_processor is None:
            return collate_batch(batch)
        batch_pixels = []
        label_ids = []
        for ex in batch:
            seq = ex["input_ids"]
            img = seq[1:-1].float().reshape(1,28,28) / 255.0
            batch_pixels.append(img)
            label_ids.append(seq[-1])
        imgs = torch.stack(batch_pixels, dim=0)
        proc_out = vision_processor(imgs)
        if "inputs_embeds" not in proc_out:
            # raw path fallback (should not happen when vision_processor is set)
            return {"input_ids": proc_out["input_ids"], "labels": proc_out["input_ids"].clone(), "attention_mask": proc_out["attention_mask"]}
        emb = proc_out["inputs_embeds"]  # (B,L,H) or variable L
        B, L, H = emb.shape
        bos = torch.zeros(B,1,H,dtype=emb.dtype)
        label_placeholder = torch.zeros(B,1,H,dtype=emb.dtype)
        inputs_embeds = torch.cat([bos, emb, label_placeholder], dim=1)
        labels_tensor = torch.full((B, inputs_embeds.size(1)), -100, dtype=torch.long)
        labels_tensor[:, -1] = torch.stack(label_ids)
        attention_mask = torch.ones(B, inputs_embeds.size(1), dtype=torch.long)
        return {"inputs_embeds": inputs_embeds, "labels": labels_tensor, "attention_mask": attention_mask}

    # swap data_collator in Trainer(...) if encoder is not None
    data_collator = collate_batch_enc if vision_processor is not None else collate_batch
    trainer.data_collator = data_collator

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
