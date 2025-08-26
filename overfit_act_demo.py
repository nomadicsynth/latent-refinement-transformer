#!/usr/bin/env python
import os
import math
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, MistralConfig
from trl import SFTConfig, SFTTrainer

from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.environ["WANDB_MODE"] = "disabled"

    dataset_path = "./preprocessed_dataset_2184_227"
    dataset = load_from_disk(dataset_path)
    train_ds = dataset["train"].select(range(min(512, len(dataset["train"]))))
    eval_ds = dataset["test"].select(range(min(128, len(dataset["test"]))))

    tok_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token = tokenizer.eos_token

    cfg = MistralConfig.from_pretrained(tok_name)
    cfg.hidden_size = 768
    cfg.intermediate_size = 3688
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 16
    cfg.num_key_value_heads = 8
    cfg._attn_implementation = "sdpa"

    print("Creating RecursiveHaltingMistralForCausalLM (K_max=4, tau=0.99, lambda=0.03)")
    model = RecursiveHaltingMistralForCausalLM(cfg, k_max=4, tau=0.99, lambda_ponder=0.03).to(device=device, dtype=torch.bfloat16)

    args = SFTConfig(
        output_dir="./results/overfit_act_demo",
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
        num_train_epochs=10,
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
        use_liger_kernel=False,
        run_name="overfit-act-demo",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Training ACT demo...")
    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval:", metrics)
    if "eval_loss" in metrics:
        print("Eval ppl:", math.exp(metrics["eval_loss"]))


if __name__ == "__main__":
    main()
