import torch
from transformers import AutoTokenizer, MistralForCausalLM
import os

# Path to the checkpoint directory
checkpoint_dir = "./results/checkpoint-15483"

# Model and tokenizer name (should match training)
tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load tokenizer
print(f"Loading tokenizer from: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model from checkpoint
print(f"Loading model from checkpoint: {checkpoint_dir}")
model = MistralForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)
model.eval()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example prompt for inference
prompt = "Doctor Who is"
print(f"Prompt: {prompt}")

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print result
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Response: {response}")
