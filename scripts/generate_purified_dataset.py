"""Generate offline purified train/validation/test datasets from a frozen purifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.purifier import PairManifestDataset
from dmmp.purifier.config import PurifierConfig
from dmmp.purifier.training import load_purifier_checkpoint
from dmmp.utils import resolve_device, write_json


MANIFEST_COLUMNS = [
    "source_id",
    "clean_index",
    "defended_index",
    "defended_local_index",
    "variant_id",
    "split",
    "class_id",
    "defended_path",
    "defended_length",
    "purified_path",
    "purified_index",
    "purifier_checkpoint",
    "diffusion_steps",
    "sampling_steps",
    "sampling_seed",
    "representation",
    "legalization_version",
    "output_length_policy",
    "output_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze purifier and generate offline purified datasets.")
    parser.add_argument("--run-dir", required=True, help="Purifier run directory.")
    parser.add_argument("--checkpoint", default="", help="Default: <run-dir>/checkpoints/best_checkpoint.pt")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/purified_datasets")
    parser.add_argument("--manifest-path", default="", help="Default: <run-dir>/manifests/purified_dataset_manifest.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--sampling-steps", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=71000)
    parser.add_argument(
        "--output-length-policy",
        choices=["model", "defended", "clean"],
        default="",
        help="Default: checkpoint config output_length_policy. model=no tail mask; defended=legacy defended length; clean=oracle clean length diagnostic.",
    )
    parser.add_argument("--max-sources-per-split", type=int, default=0, help="Debug guard; 0 means all sources.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _manifest_for_split(cfg: PurifierConfig, split: str) -> str:
    if split == "train":
        return cfg.train_manifest
    if split == "validation":
        return cfg.validation_manifest
    if split == "test":
        return cfg.test_manifest
    raise ValueError(f"Unsupported split {split!r}")


def _tail_lengths_numpy(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D trace batch, got shape {x.shape}")
    positions = np.arange(x.shape[1], dtype=np.int64).reshape(1, -1) + 1
    return np.where(x != 0, positions, 0).max(axis=1).astype(np.int64, copy=False)


def _tail_lengths_tensor(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().cpu()
    positions = torch.arange(x.shape[1], dtype=torch.long).reshape(1, -1) + 1
    return torch.where(x != 0, positions, torch.zeros_like(positions)).max(dim=1).values.long()


def _flush_shard(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    arrays: list[np.ndarray],
    writer: csv.DictWriter,
    checkpoint: Path,
    diffusion_steps: int,
    sampling_steps: int,
    sampling_seed: int,
    representation: str,
    legalization_version: str,
    output_length_policy: str,
) -> int:
    if not rows:
        return 0
    x = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=x,
        y=np.asarray([row["class_id"] for row in rows], dtype=np.int64),
        source_id=np.asarray([row["source_id"] for row in rows], dtype=np.int64),
        clean_index=np.asarray([row["clean_index"] for row in rows], dtype=np.int64),
        defended_index=np.asarray([row["defended_index"] for row in rows], dtype=np.int64),
        defended_local_index=np.asarray([row["defended_local_index"] for row in rows], dtype=np.int64),
        variant_id=np.asarray([row["variant_id"] for row in rows], dtype=np.int64),
        defended_length=np.asarray([row["defended_length"] for row in rows], dtype=np.int64),
        output_length=np.asarray([row["output_length"] for row in rows], dtype=np.int64),
    )
    for local_index, row in enumerate(rows):
        writer.writerow(
            {
                "source_id": int(row["source_id"]),
                "clean_index": int(row["clean_index"]),
                "defended_index": int(row["defended_index"]),
                "defended_local_index": int(row["defended_local_index"]),
                "variant_id": int(row["variant_id"]),
                "split": row["split"],
                "class_id": int(row["class_id"]),
                "defended_path": row["defended_path"],
                "defended_length": int(row["defended_length"]),
                "purified_path": str(output_path.resolve()),
                "purified_index": int(local_index),
                "purifier_checkpoint": str(checkpoint.resolve()),
                "diffusion_steps": int(diffusion_steps),
                "sampling_steps": int(sampling_steps),
                "sampling_seed": int(sampling_seed),
                "representation": representation,
                "legalization_version": legalization_version,
                "output_length_policy": output_length_policy,
                "output_length": int(row["output_length"]),
            }
        )
    return len(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "checkpoints" / "best_checkpoint.pt"
    device = resolve_device(str(args.device))
    model, payload = load_purifier_checkpoint(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    cfg = PurifierConfig.from_mapping(payload["config"])
    sampling_steps = int(args.sampling_steps or cfg.sampling_steps)
    output_length_policy = str(args.output_length_policy or cfg.output_length_policy).strip().lower()
    if output_length_policy not in {"model", "defended", "clean"}:
        raise ValueError("output_length_policy must be one of: model, defended, clean")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "purified_datasets"
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else run_dir / "manifests" / "purified_dataset_manifest.csv"
    if manifest_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    if output_dir.exists() and any(output_dir.glob("**/*.npz")) and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing purified shards under: {output_dir}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    requested_splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    counts: dict[str, dict[str, int]] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.sampling_seed))
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for split in requested_splits:
            dataset = PairManifestDataset(
                _manifest_for_split(cfg, split),
                cfg.clean_path,
                expected_split=split,
                seq_length=int(cfg.seq_length),
                value_scale=float(cfg.value_scale),
                max_sources=int(args.max_sources_per_split),
                preload_shards=bool(cfg.preload_shards),
                max_open_shards=int(cfg.max_open_shards),
                condition_source=str(cfg.condition_source),
                include_clean=output_length_policy == "clean",
            )
            loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
            shard_rows: list[dict[str, Any]] = []
            shard_arrays: list[np.ndarray] = []
            shard_index = 0
            split_count = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(loader, start=1):
                    defended = batch["defended"].to(device=device, dtype=torch.float32, non_blocking=True)
                    labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
                    defended_length = batch["defended_length"]
                    sampled = model.sample(defended, labels=labels, sampling_steps=sampling_steps, generator=generator)
                    output_length: torch.Tensor | None
                    if output_length_policy == "defended":
                        output_length = defended_length
                    elif output_length_policy == "clean":
                        output_length = _tail_lengths_tensor(batch["clean"])
                    else:
                        output_length = None
                    purified = model.decoder.legalize_numpy(sampled, output_length=output_length)
                    actual_output_lengths = _tail_lengths_numpy(purified)
                    rows = []
                    for item in range(purified.shape[0]):
                        rows.append(
                            {
                                "source_id": int(batch["source_id"][item]),
                                "clean_index": int(batch["clean_index"][item]),
                                "defended_index": int(batch["defended_index"][item]),
                                "defended_local_index": int(batch["defended_local_index"][item]),
                                "variant_id": int(batch["variant_id"][item]),
                                "split": split,
                                "class_id": int(batch["label"][item]),
                                "defended_path": batch["defended_path"][item],
                                "defended_length": int(batch["defended_length"][item]),
                                "output_length": int(actual_output_lengths[item]),
                            }
                        )
                    shard_rows.extend(rows)
                    shard_arrays.append(purified)
                    if len(shard_rows) >= int(args.shard_size):
                        output_path = output_dir / split / f"purified_{split}_shard{shard_index:05d}.npz"
                        written = _flush_shard(
                            output_path=output_path,
                            rows=shard_rows,
                            arrays=shard_arrays,
                            writer=writer,
                            checkpoint=checkpoint,
                            diffusion_steps=int(model.diffusion.diffusion_steps),
                            sampling_steps=sampling_steps,
                            sampling_seed=int(args.sampling_seed),
                            representation=str(payload.get("representation", "fixed_length_signed_time_sequence_v1_scaled")),
                            legalization_version=str(model.decoder.legalization_version),
                            output_length_policy=output_length_policy,
                        )
                        split_count += written
                        print(f"[purified generation] split={split} shard={shard_index} rows={written} total={split_count}", flush=True)
                        shard_rows, shard_arrays = [], []
                        shard_index += 1
                    if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                        print(f"[purified generation] split={split} batch={batch_index}/{len(loader)}", flush=True)
            if shard_rows:
                output_path = output_dir / split / f"purified_{split}_shard{shard_index:05d}.npz"
                written = _flush_shard(
                    output_path=output_path,
                    rows=shard_rows,
                    arrays=shard_arrays,
                    writer=writer,
                    checkpoint=checkpoint,
                    diffusion_steps=int(model.diffusion.diffusion_steps),
                    sampling_steps=sampling_steps,
                    sampling_seed=int(args.sampling_seed),
                    representation=str(payload.get("representation", "fixed_length_signed_time_sequence_v1_scaled")),
                    legalization_version=str(model.decoder.legalization_version),
                    output_length_policy=output_length_policy,
                )
                split_count += written
                print(f"[purified generation] split={split} shard={shard_index} rows={written} total={split_count}", flush=True)
            counts[split] = {"pairs": int(split_count), "shards": int(shard_index + (1 if shard_rows else 0))}
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "counts": counts,
        "sampling_steps": int(sampling_steps),
        "sampling_seed": int(args.sampling_seed),
        "output_length_policy": output_length_policy,
        "legalization_version": str(model.decoder.legalization_version),
        "K": 1,
    }
    write_json(run_dir / "purified_generation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
