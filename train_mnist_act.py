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
from transformers import MistralConfig, Trainer, TrainingArguments

# Local model import
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM

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
        self.ds = torchvision.datasets.MNIST(
            root="data", train=train_flag, download=True, transform=T.ToTensor()
        )
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


def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    # logits shape: [B, L, V]; For causal LM, token at position t (labels[:, t]) is
    # predicted by logits at position t-1 due to the internal shift in loss computation
    # (see standard transformers causal LM implementation: shift_logits = logits[..., :-1, :],
    # shift_labels = labels[..., 1:]). Our label (digit) token is the LAST real token in
    # the sequence, so we must look at logits from the second-to-last position.

    # Remove the final timestep of logits to align with labels[:, 1:]
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    # We only care about predicting the final (digit) label token.
    label_tokens = shifted_labels[:, -1]  # true label token ids (259..268)
    gold_digits = label_tokens - LABEL_BASE_ID

    # Take logits at predictive timestep (second-to-last original position) and slice to label range.
    label_logits = shifted_logits[:, -1, LABEL_BASE_ID: LABEL_BASE_ID + 10]  # [B, 10]
    pred_digits = np.argmax(label_logits, axis=-1)  # 0..9

    # All examples should be valid; keep mask in case of padding anomalies.
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
    num_key_value_heads=getattr(args, 'kv_heads', None) or args.heads,
        max_position_embeddings=MAX_SEQ_LEN,
    )
    # ACT parameters
    config.k_max = args.k_max
    config.tau = args.tau
    config.lambda_ponder = args.lambda_ponder
    config.use_step_film = not args.no_step_film
    config.film_rank = args.film_rank
    config.lambda_deep_supervision = args.lambda_deep_supervision
    model = RecursiveHaltingMistralForCausalLM(config)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="mnist-act")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-train-batch-size", type=int, default=64)
    p.add_argument("--per-device-eval-batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
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

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=None, help="Limit train examples (debug)")
    p.add_argument("--limit-eval", type=int, default=None, help="Limit eval examples (debug)")
    p.add_argument("--quick-test", action="store_true", help="Run tiny setup to verify script")
    p.add_argument("--push-to-hub", action="store_true")
    # Memory / perf options
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if available")
    p.add_argument("--auto-batch-reduce", action="store_true", help="On OOM, halve batch and retry once")

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

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

    if args.grad_checkpoint:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        else:  # pragma: no cover
            print("Gradient checkpointing not supported on model instance.")

    if not args.no_compile:
        try:
            model = torch.compile(model)  # type: ignore
            print("Model compiled with torch.compile")
        except Exception as e:  # pragma: no cover
            print("torch.compile failed (continuing):", e)

    # HuggingFace Trainer expects integer comparison; use -1 to mean 'not set'
    total_train_steps = args.max_steps if args.max_steps is not None else -1

    # Some older transformers versions use 'evaluation_strategy'; if unavailable we fallback to manual eval.
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
        num_train_epochs=args.epochs if total_train_steps is None else 1.0,
        max_steps=total_train_steps,
        report_to=["none"],
        fp16=args.fp16,
        bf16=args.bf16,
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        auto_find_batch_size=args.auto_batch_reduce,
        label_names=["labels"],
    )
    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_batch,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Eval:", eval_metrics)

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
            label_slice = next_logits[LABEL_BASE_ID:LABEL_BASE_ID + 10]
            pred = int(torch.argmax(label_slice).item())
            print("Quick test sample predicted digit:", pred)
            print("Model ACT stats steps_mean=", getattr(model, "_last_expected_steps_mean", None))


if __name__ == "__main__":  # pragma: no cover
    main()
