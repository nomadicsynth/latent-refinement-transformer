# science-llm

Experimental playground for Adaptive Computation Time (ACT) / recursive halting variants of a compact **Mistral-style causal LM** plus lightweight vision tokenization experiments on MNIST. The codebase focuses on:

* A recursively halting language model `RecursiveHaltingMistralForCausalLM` (wraps HF Mistral) with per‑step FiLM, ponder (expected steps) loss, and deep supervision.
* A unified mini vision processor (`NeonVisionProcessor`) supporting raw pixel tokens or learned encoders (conv, patch, wavelet, siren) feeding embeddings directly into the LM (for MNIST classification via label token prediction).
* Training pipelines for (a) general token datasets (`train.py`) and (b) MNIST ACT classification with dynamic augmentation & token:parameter ratio scaling (`train_mnist_act.py`).
* Muon optimizer integration (hidden weights on Muon, auxiliary AdamW for biases / heads) + W&B telemetry injection of ACT stats.
* Hyperparameter sweep configs (W&B) targeting attention heads, FiLM rank, model config, ACT parameters, and learning rate.
* Lightweight architecture & autograd graph visualizers (`model_vis.py`, `graphfx/`).

> Status: research / experimental. Interfaces may change. Use pinned `requirements.txt`.

---

## Features

* **Recursive Halting Transformer**: ACT loop with maximum `k_max` inner steps, halting threshold `tau`, convex combination of per-step logits weighted by halting mass.
* **Ponder / Expected Steps Loss**: Adds `lambda_ponder * E[steps]` during training for compute regularization.
* **Deep Supervision**: Optional per-step CE mixture (`lambda_deep_supervision`).
* **Per-Step FiLM Modulation**: Tiny rank‑`r` FiLM layer conditioning inner iterations (`--use-step-film`, `--film-rank`).
* **Gated Residual Refinement**: Learnable scalar gate per inner step (smooth refinement instead of hard replace).
* **Vision Embedding Path**: Replace token ids with `inputs_embeds` from conv / patch / wavelet / siren encoders (MNIST demo) via `NeonVisionProcessor`.
* **Dynamic Dataset Multiplicity**: For MNIST classification, automatically expand dataset to target token:parameter ratio (e.g. `--token-param-ratio 20`).
* **Augmentation Ramps**: Linear ramp of augmentation probability and severity (degrees / translate) over training steps.
* **Muon Optimizer Hybrid**: Hidden weight tensors trained with Muon, auxiliary params with AdamW.
* **W&B Enhanced Telemetry**: Custom callbacks inject `act_inner_steps`, `act_expected_steps` in train/eval logs.
* **Sweep Configs**: Reusable YAMLs for W&B sweeps (`sweep-*.yaml`).
* **Graph & Module Visualization**: Generate DOT/PNG architecture diagrams and torchviz autograd graphs.

---

## Installation

```bash
# (Recommended) create environment
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

# Optional: install Muon optimizer (already via git URL in requirements)
# Optional: Graphviz for diagrams
sudo apt-get update && sudo apt-get install -y graphviz
```

GPU with bfloat16 support (A100 / recent) recommended; otherwise pass `--no-bf16` / avoid `--bf16`.

---

## Quick Start: ACT Language Model Training

Preprocess / load a HF dataset to disk (example assumes a preprocessed folder already exists):

```bash
python train.py \
  --dataset-path ./preprocessed_dataset_2184_227 \
  --output-dir ./results/act_single \
  --num-train-epochs 3 \
  --hidden-size 768 --intermediate-size 3688 \
  --num-layers 2 --num-attention-heads 16 --num-kv-heads 8 \
  --k-max 8 --tau 0.999 --lambda-ponder 0.0025 \
  --film-rank 512 --lambda-deep-supervision 0.06 \
  --per-device-train-batch-size 2 --grad-accum 8 \
  --use-wandb --group act-demo --run-name demo-run
```

Key artifacts: final model in `output_dir/final_model`, optional backups if `SCIENCE_LLM_BACKUP_ROOT` set or `--backup-root` provided.

---

## MNIST ACT Classification Demo

Each 28×28 grayscale image -> 784 pixel tokens + BOS + label token (digits 0–9 mapped to IDs 259..268). Model predicts final label token.

Fast smoke test (CPU acceptable):

```bash
python train_mnist_act.py --quick-test
```

Full example with augmentation + Muon:

```bash
python train_mnist_act.py \
  --output-dir results/mnist-act \
  --epochs 3 --per-device-train-batch-size 64 \
  --learning-rate 3e-4 --use-muon \
  --k-max 4 --tau 0.9 --lambda-ponder 1e-3 \
  --use-aug --aug-prob-start 0.2 --aug-prob-end 1.0 --aug-prob-auto-fraction 0.6 \
  --token-param-ratio 20 \
  --image-encoder conv --encoder-hidden 64 --save-vision-processor
```

Vision encoder choices: `--image-encoder {raw,conv,patch,wavelet,siren}`. Use `--freeze-encoder` to stop updates.

Dataset multiplicity auto-computed: `multiplicity = ceil(ratio * param_count / (base_len * tokens_per_sample))`.

---

## Inference

Given a trained checkpoint directory (containing model + tokenizer config):

```bash
python infer.py \
  --checkpoint_dir ./results/act_single/final_model \
  --output_file ./results/output.txt \
  --max_new_tokens 256 --temperature 0.7 --top_p 0.9
```

Optional JSON overrides for sampling:

```bash
python infer.py --checkpoint_dir ckpt --sampling_config sampling_overrides.json
```

---

## Hyperparameter Sweeps (Weights & Biases)

Sweep YAMLs (grid / bayes) live at repo root:

| File | Purpose |
|------|---------|
| `sweep.yaml` | ACT core hyperparams (ponder, FiLM, deep supervision) via `hparam_search_act.py` |
| `sweep-attn.yaml` | Vary attention / KV heads |
| `sweep-film.yaml` | FiLM rank grid |
| `sweep-model-config.yaml` | Model depth/width + k_max search |
| `sweep-lr.yaml` | Muon learning rate search |

Example launch (W&B CLI):

```bash
wandb sweep sweep-attn.yaml
# then agent:
wandb agent <entity>/<project>/<sweep_id>
```

---

## Visualization

Architecture / module diagram:

```bash
python model_vis.py --mode modules --out results/arch
```

Autograd graph (may be slow without tiny dims):

```bash
python model_vis.py --mode autograd --hidden-size 64 --layers 2 --k-max 2
```

GraphFX mini graph execution (example graph provided):

```bash
python -c "from graphfx.core import run_demo; run_demo('graphfx/example_graph.json')"
```

Generated DOT/PNG files stored in `results/`.

---

## Project Structure

```text
models/                    Core model + vision encoders
  recursive_halting_mistral.py  (ACT transformer wrapper)
  neon_vision_processor.py      (Unified vision processor)
  image_encoders.py             (Conv, patch, wavelet, SIREN encoders)
train.py                  Main ACT LM trainer (TRL SFTTrainer)
train_mnist_act.py        MNIST classification w/ ACT & vision embedding path
infer.py                  Sampling inference over prompts
model_vis.py              Diagram & autograd graph utilities
graphfx/                  Minimal graph-defined model executor
sweep-*.yaml              W&B sweep configuration files
preprocessed_dataset_*/   On-disk Hugging Face dataset shards
results/                  Outputs, diagrams, exported CSVs
```

---

## Key Arguments (selected)

`train.py`:

* Model: `--hidden-size`, `--intermediate-size`, `--num-layers`, `--num-attention-heads`, `--num-kv-heads`, `--attn-impl`
* ACT: `--k-max`, `--tau`, `--lambda-ponder`, `--film-rank`, `--lambda-deep-supervision`, `--halting-mass-scale`, `--use-step-film`
* Optimizer: `--use-muon`, `--muon-lr`, `--aux-adam-lr`, `--aux-beta1`, `--aux-beta2`
* Training: `--learning-rate`, `--per-device-train-batch-size`, `--grad-accum`, `--max-steps` / `--num-train-epochs`, `--packing`, `--bf16`
* Checkpointing: `--save-strategy`, `--save-steps`, `--save-final-model`, `--save-total-limit`
* W&B: `--use-wandb`, `--group`, `--run-name`

`train_mnist_act.py` adds:

* Vision: `--image-encoder`, `--patch-size`, `--encoder-hidden`, `--freeze-encoder`
* Augmentation: `--use-aug`, ramp args `--aug-prob-start/end`, `--aug-prob-auto-fraction`, severity ramps `--aug-degrees-start`, etc.
* Dataset scaling: `--token-param-ratio`, `--aug-multiplicity`, `--max-multiplicity`
* ACT (MNIST-specific defaults): `--k-max`, `--tau`, `--lambda-ponder`, `--film-rank`, `--lambda-deep-supervision`

---

## Muon Optimizer Notes

When `--use-muon`:

* Hidden weight tensors (ndim≥2 in transformer body) -> Muon group (learning rate `--muon-lr`).
* Embeddings, lm_head, gains/biases, ACT heads & FiLM -> auxiliary AdamW (lr `--aux-adam-lr` or fallback to `--learning-rate`).
* Requires distributed init even single process; script auto-initializes a 1-rank process group.

If Muon not installed, a warning is emitted and standard optimizer used.

---

## Environment & Performance Tips

* Enable `--grad-checkpointing` to reduce memory (slower).
* Use `--compile` (PyTorch 2) after initial debugging; may skip when using `inputs_embeds` quick tests.
* Disable backups with `--no-backup` if writing frequently; otherwise set `SCIENCE_LLM_BACKUP_ROOT` for safe archiving.
* For very long contexts consider reducing `--k-max` or increasing `--tau` to encourage earlier halting.

---

## Extending

* Add new vision encoder: implement module returning `(B,N,D)` embeddings and register inside `NeonVisionProcessor._build_encoder`.
* Add ACT variant: modify loop in `recursive_halting_mistral.py` (keep telemetry attributes for callbacks).
* New sweep: copy a `sweep-*.yaml`, adjust `parameters:` and target metric.

---

## Roadmap (Indicative)

* [ ] Mixed-modality (text + vision) sequences.
* [ ] Causal cache integration for incremental generation with halting.
* [ ] More robust dataset preprocessing script & docs.
* [ ] ACT policy learning (separate controller network).
* [ ] Export to ONNX / TorchScript for inference benchmarking.

---

## Citation

If this codebase informs your research, you can cite generically:

```text
@software{science_llm_act_2025,
  title = {science-llm: Recursive Halting / ACT Mistral Experiments},
  author = {Contributors},
  year = {2025},
  url = {https://github.com/nomadicsynth/science-llm}
}
```

---

## License

Apache 2.0 (choose suitable license if not yet declared).

---

## Contributing

Issues / PRs welcome. Please:

* Run formatting / lint (if added) & include a concise summary.
* Describe experimental knobs changed and rationale.

---

## Acknowledgements

* Hugging Face Transformers / TRL
* Muon optimizer authors
* TorchVision (MNIST)
* Graphviz & torchviz for visualization

---
Happy halting.
