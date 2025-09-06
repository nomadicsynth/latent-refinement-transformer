#!/usr/bin/env python
"""Compute confusion matrix for MNIST ACT model.
Usage:
  python confusion_mnist_act.py --model-dir mnist-act-2layer-act32
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch
from train_mnist_act import MNISTSeqClassification, LABEL_BASE_ID, PAD_ID
from transformers import MistralConfig
from models.recursive_halting_mistral import RecursiveHaltingMistralForCausalLM

def _resolve_checkpoint_dir(model_dir: str) -> str:
    cfg_path = os.path.join(model_dir, 'config.json')
    if os.path.isfile(cfg_path):
        return model_dir
    # find latest checkpoint-*
    subs = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir,d))]
    if not subs:
        raise FileNotFoundError(f"No config.json and no checkpoint-* directories found in {model_dir}")
    # sort by step number
    subs.sort(key=lambda x: int(x.split('-')[-1]))
    return os.path.join(model_dir, subs[-1])


def load_model(model_dir: str) -> RecursiveHaltingMistralForCausalLM:
    ckpt_dir = _resolve_checkpoint_dir(model_dir)
    cfg_path = os.path.join(ckpt_dir, 'config.json')
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    # The custom ACT fields may exist; pass through via attribute setting after instantiation
    base_keys = {k: v for k, v in cfg_dict.items() if k in MistralConfig().to_dict()}
    config = MistralConfig(**base_keys)
    # attach supplemental attributes
    for k in [
        'k_max','tau','lambda_ponder','use_step_film','film_rank','lambda_deep_supervision','num_key_value_heads'
    ]:
        if k in cfg_dict:
            setattr(config, k, cfg_dict[k])
    model = RecursiveHaltingMistralForCausalLM(config)
    # state dict
    # Prefer safetensors
    st_path = os.path.join(ckpt_dir, 'model.safetensors')
    if os.path.isfile(st_path):
        from safetensors.torch import load_file
        state = load_file(st_path, device='cpu')
    else:
        pt_path = os.path.join(ckpt_dir, 'pytorch_model.bin')
        if not os.path.isfile(pt_path):
            weights = [p for p in os.listdir(ckpt_dir) if p.startswith('pytorch_model-') and p.endswith('.bin')]
            if not weights:
                raise FileNotFoundError(f"No model weights found in {ckpt_dir}")
            pt_path = os.path.join(ckpt_dir, weights[0])
        state = torch.load(pt_path, map_location='cpu')
    # Handle torch.compile saved weights with '_orig_mod.' prefix
    if all(k.startswith('_orig_mod.') for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            nk = k[len('_orig_mod.') :]
            new_state[nk] = v
        state = new_state
    # Some keys might include step_gates etc already matching model
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print('Missing keys:', missing)
    if unexpected:
        print('Unexpected keys:', unexpected)
    return model

def compute_confusion(model_dir: str, limit: int | None = None, batch_size: int = 64, device: str | None = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_dir).to(device).eval()
    test_ds = MNISTSeqClassification(split='test', limit=limit)
    cm = np.zeros((10,10), dtype=np.int64)
    total = 0
    correct = 0

    def iter_batches():
        batch = []
        for ex in test_ds:
            batch.append(ex)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    with torch.no_grad():
        for batch in iter_batches():
            # build tensor batch without label token for prediction
            max_len = max(x['input_ids'].size(0) - 1 for x in batch)  # exclude label
            ids_list = []
            mask_list = []
            true_digits = []
            for ex in batch:
                full_seq = ex['input_ids']
                true_digits.append(int(full_seq[-1].item() - LABEL_BASE_ID))
                seq_in = full_seq[:-1]
                pad_len = max_len - seq_in.size(0)
                if pad_len > 0:
                    pad = torch.full((pad_len,), PAD_ID, dtype=seq_in.dtype)
                    seq_in_padded = torch.cat([seq_in, pad])
                    mask = torch.cat([torch.ones_like(seq_in), torch.zeros_like(pad)])
                else:
                    seq_in_padded = seq_in
                    mask = torch.ones_like(seq_in)
                ids_list.append(seq_in_padded)
                mask_list.append(mask)
            input_ids = torch.stack(ids_list).to(device)
            attention_mask = torch.stack(mask_list).to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # For each sample, find last non-pad position according to mask
            last_indices = attention_mask.sum(dim=1) - 1  # [B]
            logits = out.logits
            last_logits = logits[torch.arange(logits.size(0)), last_indices]
            label_slice = last_logits[:, LABEL_BASE_ID: LABEL_BASE_ID + 10]
            preds = torch.argmax(label_slice, dim=-1).cpu().numpy()
            true_arr = np.array(true_digits)
            for t, p in zip(true_arr, preds):
                cm[t, p] += 1
            total += true_arr.size
            correct += (true_arr == preds).sum()

    acc = correct / total if total else 0.0
    row_norm = cm / cm.sum(axis=1, keepdims=True)
    return cm, row_norm, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    cm, row_norm, acc = compute_confusion(args.model_dir, args.limit, args.batch_size)
    print('Confusion matrix:\n', cm)
    print('Row-normalized (per true class):\n', np.round(row_norm, 3))
    print(f'Accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
