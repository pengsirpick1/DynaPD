# -*- coding: utf-8 -*-
"""Prepare TAM/RF-prob/fast-keypoint archives for Stage B at CW scale."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.data import load_cw_data
from dmmp.evaluation.attack_models import build_rf_tam_input
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_run_b2e_diverse_search import _default_checkpoint


DEFAULT_SPLIT_FILE = "results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split_file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split_name", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        name = str(args.run_name)
    else:
        name = (
            f"stage_b_fast_keypoint_{args.split_name}_"
            f"shard{int(args.shard_id):03d}of{int(args.num_shards):03d}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_split_indices(path: str | Path, split_name: str) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as arrays:
        if str(split_name) == "all":
            parts = [np.asarray(arrays[f"{name}_indices"], dtype=np.int64) for name in ("train", "val", "test")]
            return np.concatenate(parts, axis=0)
        return np.asarray(arrays[f"{split_name}_indices"], dtype=np.int64)


def _select_shard(indices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if int(args.num_shards) > 1:
        shard_id = int(args.shard_id)
        num_shards = int(args.num_shards)
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"shard_id must be in [0,{num_shards}), got {shard_id}")
        selected = selected[shard_id::num_shards]
    if int(args.sample_offset) > 0:
        selected = selected[int(args.sample_offset) :]
    if int(args.max_samples) > 0:
        selected = selected[: int(args.max_samples)]
    return selected.astype(np.int64)


def _load_cw(args: argparse.Namespace):
    cfg = SimpleNamespace(
        data_root=str(args.data_root),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        max_samples=0,
        max_classes=0,
    )
    raw, labels, trace_ids, _splits, source = load_cw_data(cfg)
    return raw, np.asarray(labels, dtype=np.int64), np.asarray(trace_ids).astype(str), str(source)


def _fast_keypoint_batch(attacker, tam: np.ndarray, *, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []
    logits_out: list[np.ndarray] = []
    values_np = np.asarray(tam, dtype=np.float32)
    for start in range(0, values_np.shape[0], max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), values_np.shape[0])
        values = torch.as_tensor(values_np[start:end], dtype=torch.float32, device=device)
        values.requires_grad_(True)
        logits = attacker.logits(values)
        probs = torch.softmax(logits, dim=1)
        y0 = probs.argmax(dim=1)
        selected = probs[torch.arange(probs.shape[0], device=device), y0]
        masked = probs.clone()
        masked[torch.arange(probs.shape[0], device=device), y0] = -1.0
        other = masked.max(dim=1).values
        objective = (selected - other).sum()
        attacker.model.zero_grad(set_to_none=True)
        objective.backward()
        grad = values.grad.detach().abs()
        mag = values.detach().abs()
        grad = grad / grad.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mag = mag / mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        masks.append((0.70 * grad + 0.30 * mag).detach().cpu().numpy().astype(np.float32))
        probs_out.append(probs.detach().cpu().numpy().astype(np.float32))
        logits_out.append(logits.detach().cpu().numpy().astype(np.float32))
    return (
        np.concatenate(masks, axis=0),
        np.concatenate(probs_out, axis=0),
        np.concatenate(logits_out, axis=0),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    split_indices = _load_split_indices(args.split_file, args.split_name)
    source_indices = _select_shard(split_indices, args)
    if source_indices.size == 0:
        raise ValueError("No samples selected.")
    raw, labels, trace_ids, source = _load_cw(args)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=int(len(np.unique(labels))),
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    store_dtype = np.float16 if str(args.dtype) == "float16" else np.float32
    tam_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    pred_labels: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for start in range(0, source_indices.size, max(1, int(args.batch_size))):
        end = min(start + max(1, int(args.batch_size)), source_indices.size)
        idx = source_indices[start:end]
        if args.progress:
            print(f"[fast-keypoint] {start}:{end} / {source_indices.size}", flush=True)
        raw_batch = np.asarray(raw[idx], dtype=np.float32)
        tam = build_rf_tam_input(
            raw_batch,
            max_len=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
            num_slots=int(args.rf_num_slots),
        ).astype(np.float32)
        mask, prob, _logits = _fast_keypoint_batch(attacker, tam, device=device, batch_size=int(args.batch_size))
        tam_parts.append(tam.astype(store_dtype))
        mask_parts.append(mask.astype(store_dtype))
        prob_parts.append(prob.astype(np.float32))
        pred = np.argmax(prob, axis=1).astype(np.int64)
        pred_labels.append(pred)
        for local, source_index in enumerate(idx.tolist()):
            rows.append(
                {
                    "archive_index": int(start + local),
                    "source_index": int(source_index),
                    "sample_id": str(trace_ids[source_index]),
                    "true_label": int(labels[source_index]),
                    "pred_label": int(pred[local]),
                    "pred_confidence": float(prob[local, pred[local]]),
                    "mask_mean": float(mask[local].mean()),
                    "mask_max": float(mask[local].max()),
                }
            )
    tam_all = np.concatenate(tam_parts, axis=0)
    mask_all = np.concatenate(mask_parts, axis=0)
    prob_all = np.concatenate(prob_parts, axis=0)
    pred_all = np.concatenate(pred_labels, axis=0).astype(np.int64)
    archive_path = output_dir / "fast_keypoint_archive.npz"
    np.savez_compressed(
        archive_path,
        tam=tam_all,
        mask=mask_all,
        pred_prob=prob_all.astype(np.float32),
        pred_labels=pred_all,
        labels=np.asarray(labels[source_indices], dtype=np.int64),
        source_indices=source_indices.astype(np.int64),
        sample_ids=np.asarray(trace_ids[source_indices]).astype(str),
    )
    _write_csv(output_dir / "fast_keypoint_metrics.csv", rows)
    manifest = {
        "archive": str(archive_path),
        "metrics": str(output_dir / "fast_keypoint_metrics.csv"),
        "source": str(source),
        "split_file": str(args.split_file),
        "split_name": str(args.split_name),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "samples": int(source_indices.size),
        "dtype": str(args.dtype),
        "attacker": str(args.attacker),
        "checkpoint": str(Path(checkpoint).resolve()),
        "runtime_sec": float(time.perf_counter() - start_time),
    }
    (output_dir / "fast_keypoint_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
