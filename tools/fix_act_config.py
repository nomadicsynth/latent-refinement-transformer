#!/usr/bin/env python
import argparse
import json
import os
from transformers import AutoConfig


def main():
    p = argparse.ArgumentParser(description="Patch a checkpoint's config.json with ACT/FiLM hyperparameters.")
    p.add_argument("--model-dir", required=True, help="Path to model/checkpoint directory (contains config.json)")

    # ACT/FiLM hyperparameters (required to avoid accidental wrong defaults)
    p.add_argument("--k-max", type=int, required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--lambda-ponder", type=float, required=True)
    p.add_argument("--halting-mass-scale", type=float, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--use-step-film", dest="use_step_film", action="store_true")
    group.add_argument("--no-use-step-film", dest="use_step_film", action="store_false")
    p.add_argument("--film-rank", type=int, required=True)
    p.add_argument("--lambda-deep-supervision", type=float, required=True)

    # Optional: ensure architectures includes our class so AutoModel can pick it up if used
    p.add_argument(
        "--set-architectures",
        action="store_true",
        default=True,
        help="Set architectures to ['RecursiveHaltingMistralForCausalLM'] (default: on)",
    )
    p.add_argument(
        "--no-set-architectures",
        dest="set_architectures",
        action="store_false",
        help="Do not modify architectures field",
    )

    # Optional: ensure use_cache is set to False
    p.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Set use_cache to False (default: off)",
    )

    args = p.parse_args()

    model_dir = args.model_dir
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isdir(model_dir):
        raise SystemExit(f"Model directory not found: {model_dir}")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"config.json not found in: {model_dir}")

    # Load existing config
    cfg = AutoConfig.from_pretrained(model_dir)
    before = cfg.to_dict()

    # Inject ACT/FiLM hyperparameters
    cfg.k_max = int(args.k_max)
    cfg.tau = float(args.tau)
    cfg.lambda_ponder = float(args.lambda_ponder)
    cfg.halting_mass_scale = float(args.halting_mass_scale)
    cfg.use_step_film = bool(args.use_step_film)
    cfg.film_rank = int(args.film_rank)
    cfg.lambda_deep_supervision = float(args.lambda_deep_supervision)
    cfg.use_cache = args.use_cache

    if args.set_architectures:
        cfg.architectures = ["RecursiveHaltingMistralForCausalLM"]

    # Save updated config.json in-place
    cfg.save_pretrained(model_dir)

    # Minimal confirmation print
    after = cfg.to_dict()
    touched = [
        "k_max",
        "tau",
        "lambda_ponder",
        "halting_mass_scale",
        "use_step_film",
        "film_rank",
        "lambda_deep_supervision",
        "use_cache",
        "architectures",
    ]
    delta = {k: {"old": before.get(k), "new": after.get(k)} for k in touched if before.get(k) != after.get(k)}
    print(json.dumps({"saved_to": cfg_path, "updated_fields": delta}, indent=2))


if __name__ == "__main__":
    main()
