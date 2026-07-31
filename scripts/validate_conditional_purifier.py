"""Validate a trained conditional purifier on validation pairs only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.purifier import PairManifestDataset
from dmmp.purifier.config import PurifierConfig
from dmmp.purifier.training import load_purifier_checkpoint, validate_purifier
from dmmp.utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation-only metrics for a purifier checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="", help="Default: <run-dir>/checkpoints/best_checkpoint.pt")
    parser.add_argument("--max-validation-sources", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "checkpoints" / "best_checkpoint.pt"
    device = resolve_device(str(args.device))
    model, payload = load_purifier_checkpoint(checkpoint, device)
    cfg = PurifierConfig.from_mapping(payload["config"])
    cfg = replace(cfg, max_validation_sources=int(args.max_validation_sources or 0), device=str(device))
    dataset = PairManifestDataset(
        cfg.validation_manifest,
        cfg.clean_path,
        expected_split="validation",
        seq_length=int(cfg.seq_length),
        value_scale=float(cfg.value_scale),
        max_sources=int(cfg.max_validation_sources),
        preload_shards=bool(cfg.preload_shards),
        max_open_shards=int(cfg.max_open_shards),
    )
    metrics = validate_purifier(
        model,
        dataset,
        cfg,
        torch.device(device),
        batch_size=int(args.batch_size or cfg.batch_size),
        validation_seed=int(cfg.seed) + 9001,
    )
    output = Path(args.output).resolve() if args.output else run_dir / "validation_metrics.json"
    write_json(output, {"checkpoint": str(checkpoint), "metrics": metrics, "scope": "validation_only"})
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
