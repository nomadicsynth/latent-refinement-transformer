#!/usr/bin/env python
import os
import math
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, MistralConfig
from trl import SFTConfig, SFTTrainer

from models.recursive_mistral import RecursiveMistralForCausalLM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.environ["WANDB_MODE"] = "disabled"

    dataset_path = "./preprocessed_dataset_2184_227"
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    train_ds = dataset["train"].select(range(min(512, len(dataset["train"]))))
    eval_ds = dataset["test"].select(range(min(128, len(dataset["test"]))))

    tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Base config (small) – same as train.py, but we’ll use the recursive wrapper.
    config_name = tokenizer_name
    cfg = MistralConfig.from_pretrained(config_name)
    cfg.hidden_size = 768
    cfg.intermediate_size = 3688
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 16
    cfg.num_key_value_heads = 8
    cfg._attn_implementation = "sdpa"

    # Recursive wrapper with K=4 inner steps
    print("Creating RecursiveMistralForCausalLM (K=4)")
    model = RecursiveMistralForCausalLM(cfg, inner_steps=4).to(device=device, dtype=torch.bfloat16)

    training_args = SFTConfig(
        output_dir="./results/overfit_demo",
        eval_strategy="steps",
        eval_steps=50,
        eval_on_start=True,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=25,
        weight_decay=0.0,
        completion_only_loss=False,
        bf16=True,
        bf16_full_eval=True,
        max_length=512,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataset_num_proc=2,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        packing=False,
        dataset_kwargs={"skip_preprocessing": True},
        use_liger_kernel=True,
        run_name="overfit-demo",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting overfit demo training...")
    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval:", metrics)
    if "eval_loss" in metrics:
        print("Eval ppl:", math.exp(metrics["eval_loss"]))


if __name__ == "__main__":
    main()
