import argparse
import os
from datetime import datetime

import torch
import shutil
import subprocess

from models.recursive_halting_mistral import (
    RecursiveHaltingMistralForCausalLM,
)


def build_tiny_model(
	hidden_size: int,
	intermediate_size: int,
	num_layers: int,
	num_heads: int,
	num_kv_heads: int,
	k_max: int,
	tau: float,
	lambda_ponder: float,
):
	"""Construct a very small Mistral config model for fast visualization."""
	# Import here to avoid heavy imports if not needed elsewhere
	from transformers import MistralConfig

	cfg = MistralConfig(
		vocab_size=256,
		hidden_size=hidden_size,
		intermediate_size=intermediate_size,
		num_hidden_layers=num_layers,
		num_attention_heads=num_heads,
		num_key_value_heads=num_kv_heads,
		# Keep default rotary/rope & others from HF
	)
	model = RecursiveHaltingMistralForCausalLM(
		cfg, k_max=k_max, tau=tau, lambda_ponder=lambda_ponder
	)
	return model


def make_inputs(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
	x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
	y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
	attn = torch.ones(batch_size, seq_len, device=device)
	return x, y, attn


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def _find_core_and_layers(model: torch.nn.Module):
	"""Best-effort: find the core transformer module and its layer list."""
	core = getattr(model, "model", model)  # HF models usually nest under .model
	for attr in ("layers", "blocks", "h"):
		if hasattr(core, attr):
			layers = getattr(core, attr)
			try:
				return core, list(layers)
			except Exception:
				pass
	# Fallback: collect immediate children that look like layers
	children = list(core.named_children())
	layer_like = [m for name, m in children if name.isdigit() or "layer" in name.lower() or "block" in name.lower()]
	return core, layer_like


def render_architecture_diagram(
	model: torch.nn.Module,
	out_base: str,
	fmt: str,
	k_max: int,
	tau: float,
	lambda_ponder: float,
):
	"""Emit a compact, paper-style module diagram via Graphviz DOT."""
	core, layers = _find_core_and_layers(model)
	num_layers = len(layers)

	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_dir = os.path.dirname(out_base) or "."
	os.makedirs(out_dir, exist_ok=True)
	filename = os.path.basename(out_base) + f"_{ts}"

	# Build DOT text (compact, with clusters)
	lines = []
	lines.append("digraph G {")
	lines.append('  rankdir=LR;')
	lines.append('  graph [fontsize=10, fontname="Helvetica"];')
	lines.append('  node  [shape=record, fontsize=10, fontname="Helvetica"];')
	lines.append('  edge  [fontsize=9,  fontname="Helvetica"];')

	# Annotations for ACT (two-line note)
	act_label = f"ACT (k_max={k_max}, tau={tau}, lambda={lambda_ponder})\nControls dynamic loop"

	# High-level pipeline
	lines.append('  input_ids [label="Input IDs"];')
	lines.append('  attn_mask [label="Attention Mask"];')
	lines.append('  embed [label="Token+Pos Embeddings"];')

	# Transformer block cluster (collapsed N)
	lines.append('  subgraph cluster_transformer {')
	lines.append('    label="Transformer Core";')
	lines.append('    style="rounded"; color="#bbbbbb";')
	lines.append(f'    block [label="{{x | N={num_layers} layers}}"];')
	lines.append(f'    act [label="{act_label}", shape=note, color="#6666aa"];')
	# Optional: draw a self-loop to indicate dynamic unrolling within the transformer
	if k_max and k_max > 1:
		lines.append(f'    block -> block [label="dynamic loop (min=1, max=k_max)", style=dashed];')
	lines.append('  }')

	lines.append('  norm [label="Final Norm"];')
	lines.append('  lm_head [label="LM Head"];')
	lines.append('  loss [label="Loss (xent)"];')

	# Edges (simple flow)
	lines.append('  input_ids -> embed;')
	lines.append('  attn_mask -> block [style=dashed, label="mask"];')
	lines.append('  embed -> block;')
	lines.append('  block -> norm;')
	lines.append('  norm -> lm_head;')
	lines.append('  {rank=same; input_ids; attn_mask}')
	lines.append('  lm_head -> loss [label="with labels"];')
	lines.append("}")

	dot_source = "\n".join(lines)
	dot_path = os.path.join(out_dir, f"{filename}.dot")
	with open(dot_path, "w", encoding="utf-8") as f:
		f.write(dot_source)

	# Try rendering if `dot` exists
	dot_bin = shutil.which("dot")
	if dot_bin:
		out_path = os.path.join(out_dir, f"{filename}.{fmt}")
		try:
			subprocess.run([dot_bin, f"-T{fmt}", dot_path, "-o", out_path], check=True)
			# Optionally remove the .dot after success
			# os.remove(dot_path)
			print(f"Architecture diagram: {out_path}")
			return out_path
		except subprocess.CalledProcessError as e:
			print("dot render failed; keeping .dot:", dot_path, "error:", e)
	else:
		print("Graphviz `dot` not found; wrote DOT file:", dot_path)
	return dot_path


def main():
	ap = argparse.ArgumentParser(description="Visualize RecursiveHaltingMistral with torchviz")
	ap.add_argument("--out", default="results/model_graph", help="Output path (without extension)")
	ap.add_argument("--format", default="png", choices=["png", "svg", "pdf"], help="Graph format")
	ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
	ap.add_argument("--mode", default="autograd", choices=["autograd", "modules"], help="autograd graph vs. module-level diagram")
	ap.add_argument("--batch-size", type=int, default=1)
	ap.add_argument("--seq-len", type=int, default=16)
	ap.add_argument("--vocab-size", type=int, default=256)
	# Tiny model dims (safe defaults)
	ap.add_argument("--hidden-size", type=int, default=64)
	ap.add_argument("--intermediate-size", type=int, default=256)
	ap.add_argument("--layers", type=int, default=2)
	ap.add_argument("--heads", type=int, default=8)
	ap.add_argument("--kv-heads", type=int, default=2)
	# ACT params
	ap.add_argument("--k-max", type=int, default=2)
	ap.add_argument("--tau", type=float, default=0.99)
	ap.add_argument("--lambda-ponder", type=float, default=1e-3)
	args = ap.parse_args()

	# Defer import for autograd mode only
	if args.mode == "autograd":
		from torchviz import make_dot

	device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

	# Build model and data
	model = build_tiny_model(
		hidden_size=args.hidden_size,
		intermediate_size=args.intermediate_size,
		num_layers=args.layers,
		num_heads=args.heads,
		num_kv_heads=args.kv_heads,
		k_max=args.k_max,
		tau=args.tau,
		lambda_ponder=args.lambda_ponder,
	)
	model.eval()
	model.to(device=device, dtype=torch.float32)

	if args.mode == "modules":
		render_architecture_diagram(
			model,
			out_base=args.out,
			fmt=args.format,
			k_max=args.k_max,
			tau=args.tau,
			lambda_ponder=args.lambda_ponder,
		)
	else:
		x, y, attn = make_inputs(args.batch_size, args.seq_len, args.vocab_size, device)
		with torch.no_grad():
			_ = model(input_ids=x, attention_mask=attn, labels=y, return_dict=True)
		model.zero_grad(set_to_none=True)
		out = model(input_ids=x, attention_mask=attn, labels=y, return_dict=True)
		loss = out.loss
		# Build the autograd graph
		dot = make_dot(loss, params=dict(model.named_parameters()))
		dot.format = args.format
		base_out = args.out
		out_dir = os.path.dirname(base_out) or "."
		ensure_dir(out_dir)
		ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = os.path.basename(base_out) + f"_{ts}"
		try:
			output_path = dot.render(filename=filename, directory=out_dir, cleanup=True)
			print(f"Autograd graph: {output_path}")
		except Exception as e:
			dot_path = os.path.join(out_dir, f"{filename}.dot")
			with open(dot_path, "w", encoding="utf-8") as f:
				f.write(dot.source)
			print("Graphviz render failed; wrote DOT file instead:", dot_path, "\nError:", e)

if __name__ == "__main__":
	main()

