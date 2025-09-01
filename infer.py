import torch
from transformers import AutoTokenizer
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM
from transformers import set_seed
import argparse
import random
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Inference script for Recursive Halting Mistral")
parser.add_argument("--checkpoint_dir", type=str, help="Path to the checkpoint directory")
parser.add_argument("--output_file", type=str, default="./results/output.txt", help="Path to the output file")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
args = parser.parse_args()

if args.checkpoint_dir is None:
    raise ValueError("Checkpoint directory must be specified.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility (optional but helpful when comparing settings)
seed = 42
set_seed(seed)

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
topics = ["Doctor Who", "Quantum Computing", "Photosynthesis", "Alan Turing", "Black Dog Institute", "Albert Einstein", "Michael Stevens", "Darth Vader"]

prompts = []
for topic in topics:
    # Title line + lead stub
    prompt = f"{topic}\n{topic} is"
    prompts.append(prompt)

print(f"Prompts: {prompts}")

# Batch generation respecting --batch_size
all_responses = []
num_prompts = len(prompts)
bsz = max(1, int(args.batch_size))
num_batches = (num_prompts + bsz - 1) // bsz
print(f"Running inference in {num_batches} batch(es) of up to {bsz} prompts each…")

with torch.no_grad():
    for start in tqdm(range(0, num_prompts, bsz), total=num_batches, desc="Generating", unit="batch"):
        end = min(start + bsz, num_prompts)
        batch_prompts = prompts[start:end]

        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate for this batch
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            top_p=0.90,
            repetition_penalty=1.061,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Cache not supported yet
        )

        # Decode and collect
        batch_responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_responses.extend(batch_responses)

# Print results in original order
print("Responses:")
for i, resp in enumerate(all_responses):
    print("-" * 10, f"Response {i+1}", "-" * 10)
    print(resp)

# Write responses to output file
with open(args.output_file, "w", encoding="utf-8") as f:
    for i, resp in enumerate(all_responses):
        f.write("-" * 10 + f"Response {i+1}" + "-" * 10 + "\n")
        f.write(resp + "\n")
print(f"Responses written to: {args.output_file}")
