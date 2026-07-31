"""Train the first conditional diffusion traffic purifier."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.purifier import load_purifier_config
from dmmp.purifier.training import train_purifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a manifest-driven conditional diffusion purifier.")
    parser.add_argument("--config", default="configs/purifier/diffusion_reconstruction.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lambda-rec", type=float, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--condition-channels", type=int, default=None)
    parser.add_argument("--max-train-sources", type=int, default=None)
    parser.add_argument("--max-validation-sources", type=int, default=None)
    parser.add_argument("--pairing-mode", choices=["correct", "shuffled"], default=None)
    parser.add_argument("--condition-mode", choices=["conditioned", "zero", "unconditional"], default=None)
    parser.add_argument("--condition-source", choices=["defended", "clean", "label"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Small fast run that still exercises train/validation separation.")
    return parser.parse_args()


def _override(cfg, args: argparse.Namespace):
    updates: dict[str, Any] = {}
    mapping = {
        "run_name": args.run_name,
        "output_root": args.output_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lambda_rec": args.lambda_rec,
        "diffusion_steps": args.diffusion_steps,
        "sampling_steps": args.sampling_steps,
        "hidden_channels": args.hidden_channels,
        "condition_channels": args.condition_channels,
        "max_train_sources": args.max_train_sources,
        "max_validation_sources": args.max_validation_sources,
        "pairing_mode": args.pairing_mode,
        "condition_mode": args.condition_mode,
        "condition_source": args.condition_source,
        "device": args.device,
        "seed": args.seed,
        "log_every": args.log_every,
    }
    for key, value in mapping.items():
        if value is not None:
            updates[key] = value
    if args.smoke:
        updates.setdefault("run_name", "purifier_smoke")
        updates.update(
            {
                "epochs": int(args.epochs or 1),
                "batch_size": int(args.batch_size or 16),
                "diffusion_steps": int(args.diffusion_steps or 8),
                "sampling_steps": int(args.sampling_steps or 4),
                "hidden_channels": int(args.hidden_channels or 16),
                "condition_channels": int(args.condition_channels or 16),
                "max_train_sources": int(args.max_train_sources or 128),
                "max_validation_sources": int(args.max_validation_sources or 64),
                "log_every": int(args.log_every or 10),
            }
        )
    return replace(cfg, **updates)


def main() -> None:
    args = parse_args()
    cfg = _override(load_purifier_config(args.config), args)
    result = train_purifier(cfg, overwrite=bool(args.overwrite))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
