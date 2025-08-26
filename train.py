import os

import torch
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM
from trl import SFTConfig, SFTTrainer

from datasets import load_from_disk

print("Training wikipedia model")

# Train: 10000 Test: 1000
# dataset_name = "./preprocessed_dataset_562_55"
# Train: 20000 Test: 2000
# dataset_name = "./preprocessed_dataset_1108_113"
# Train: 40000 Test: 4000
dataset_name = "./preprocessed_dataset_2184_227"
# Train: 80000 Test: 8000
# dataset_name = "./preprocessed_dataset_4360_446"
# Train: 160000 Test: 16000
# dataset_name = "./preprocessed_dataset_8770_888"
# Train: 320000 Test: 32000
# dataset_name = "./preprocessed_dataset_17510_1759"
# Train: 640000 Test: 64000
# dataset_name = "./preprocessed_dataset_34954_3492"
# Train: 1280000 Test: 128000
# dataset_name = "./preprocessed_dataset_69668_7001"
# Train: 1530000 Test: 170000 - Full dataset of 1.7M rows split at 10%
# dataset_name = "./preprocessed_dataset_83208_9318"

tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_config_name = tokenizer_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "science-llm"
print(f"Logging to wandb project: {os.environ['WANDB_PROJECT']}")

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "false"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

# Load the English Wikipedia dataset from the latest dump
print(f"Loading dataset from: {dataset_name}")
dataset = load_from_disk(dataset_name)
print(dataset)

# Load and configure the tokeniser
print(f"Loading tokenizer from: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
print(f"Setting pad token to eos token")
tokenizer.pad_token = tokenizer.eos_token

print(f"Create config from: {model_config_name}")
config = MistralConfig.from_pretrained(model_config_name)

# Modify the config for 70M params
config.hidden_size = 768
config.intermediate_size = 3688
config.num_hidden_layers = 2
config.num_attention_heads = 16
config.num_key_value_heads = 8
config._attn_implementation = "flash_attention_2"

# Create the model
print(f"Creating model on: {device}")
model = MistralForCausalLM(config).to(device=device, dtype=torch.bfloat16)

trainable_params = model.num_parameters()
trainable_params_hr = 0
if trainable_params >= 1e9:
    trainable_params_hr = f"{trainable_params / 1e9:.0f}B"
elif trainable_params >= 1e6:
    trainable_params_hr = f"{trainable_params / 1e6:.0f}M"
elif trainable_params >= 1e3:
    trainable_params_hr = f"{trainable_params / 1e3:.0f}K"

# Print the number of trainable parameters
print(f"Trainable parameters: {trainable_params} ({trainable_params_hr})")

print("Creating and configuring Trainer")
training_args = SFTConfig(
    output_dir="./results",
    eval_strategy="epoch",
    eval_on_start=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    num_train_epochs=4,
    weight_decay=0.01,
    completion_only_loss=False,
    bf16=True,
    bf16_full_eval=True,
    max_length=32768,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",
    report_to="wandb",
    # report_to="none",
    dataset_num_proc=4,
    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.pad_token,
    packing=True,
    dataset_kwargs={"skip_preprocessing": True},
    use_liger_kernel=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()
