"""Run a one-batch gradient-flow check for the purifier."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.purifier import PairManifestDataset, load_purifier_config
from dmmp.purifier.pipeline import build_purifier, gradient_report
from dmmp.utils import resolve_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check purifier gradient flow without attack classifiers.")
    parser.add_argument("--config", default="configs/purifier/diffusion_reconstruction.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-sources", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(
        load_purifier_config(args.config),
        batch_size=int(args.batch_size),
        max_train_sources=int(args.max_train_sources),
        device=str(args.device),
    )
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    dataset = PairManifestDataset(
        cfg.train_manifest,
        cfg.clean_path,
        expected_split="train",
        seq_length=int(cfg.seq_length),
        value_scale=float(cfg.value_scale),
        max_sources=int(cfg.max_train_sources),
        preload_shards=bool(cfg.preload_shards),
        max_open_shards=int(cfg.max_open_shards),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = build_purifier(cfg).to(device)
    clean = batch["clean"].to(device=device, dtype=torch.float32)
    defended = batch["defended"].to(device=device, dtype=torch.float32)
    loss, parts = model.training_losses(clean, defended, lambda_rec=float(cfg.lambda_rec), reconstruction_mode=str(cfg.reconstruction_loss))
    loss.backward()
    report = gradient_report(model)
    report["loss"] = {key: float(value.detach().cpu()) for key, value in parts.items()}
    report["batch_sources"] = [int(value) for value in batch["source_id"].tolist()]
    report["label_passed_to_model"] = False
    output = Path(args.output).resolve() if args.output else Path("gradient_flow_report.json").resolve()
    write_json(output, report)
    print(json.dumps({"output": str(output), "report": report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
