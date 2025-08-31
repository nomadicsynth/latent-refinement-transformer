import torch
from transformers import AutoTokenizer
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM
import argparse

parser = argparse.ArgumentParser(description="Inference script for Recursive Halting Mistral")
parser.add_argument("--checkpoint_dir", type=str, help="Path to the checkpoint directory")
args = parser.parse_args()

if args.checkpoint_dir is None:
    raise ValueError("Checkpoint directory must be specified.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
print(f"Loading tokenizer from: {args.checkpoint_dir}")
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
tokenizer.pad_token = tokenizer.eos_token

# Load model from checkpoint
print(f"Loading model from checkpoint: {args.checkpoint_dir}")
model = RecursiveHaltingMistralForCausalLM.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16)
model.eval()

# Move model to device
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
