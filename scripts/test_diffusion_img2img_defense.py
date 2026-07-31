# -*- coding: utf-8 -*-
"""Test diffusion img2img-style defense on clean traffic.

This diagnostic keeps x0 as the original clean trace: q(x_t | x0), then
partially denoises from t to 0. It is meant to test whether residual diffusion
noise can behave like a low-overhead defense perturbation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import stored_npy_from_npz
from dmmp.purifier.config import PurifierConfig
from dmmp.purifier.training import load_purifier_checkpoint
from dmmp.utils import resolve_device, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean x0 -> noisy x_t -> denoised trace diagnostic.")
    parser.add_argument("--purifier-run-dir", required=True)
    parser.add_argument("--checkpoint", default="", help="Default: <run-dir>/checkpoints/best_checkpoint.pt")
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/img2img_defense_test")
    parser.add_argument("--max-sources", type=int, default=256)
    parser.add_argument("--start-timesteps", default="1,2,4,8,16,31")
    parser.add_argument("--sampling-steps", type=int, default=0, help="Default: min(t+1, checkpoint sampling_steps)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=74000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class CleanStore:
    def __init__(self, path: str | Path):
        clean_path = Path(path)
        x_map = stored_npy_from_npz(clean_path, "X")
        y_map = stored_npy_from_npz(clean_path, "y")
        self._payload = None
        if x_map is not None and y_map is not None:
            self.x = x_map
            self.y = np.asarray(y_map, dtype=np.int64)
        else:
            self._payload = np.load(clean_path, allow_pickle=False)
            self.x = self._payload["X"]
            self.y = np.asarray(self._payload["y"], dtype=np.int64)

    def batch(self, indices: list[int], *, seq_length: int, value_scale: float) -> np.ndarray:
        out = np.zeros((len(indices), int(seq_length)), dtype=np.float32)
        for row, index in enumerate(indices):
            values = np.asarray(self.x[int(index)], dtype=np.float32).reshape(-1)
            take = min(out.shape[1], len(values))
            if take:
                out[row, :take] = values[:take]
        return out / float(value_scale)

    def close(self) -> None:
        if self._payload is not None:
            self._payload.close()


def _read_unique_test_rows(path: str | Path, max_sources: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[int] = set()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            source_id = int(row["source_id"])
            if source_id in seen:
                continue
            seen.add(source_id)
            rows.append(row)
            if int(max_sources) > 0 and len(rows) >= int(max_sources):
                break
    if not rows:
        raise ValueError(f"No unique test rows found in {path}")
    return rows


def _tail_lengths(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values)
    positions = np.arange(x.shape[1], dtype=np.int64).reshape(1, -1) + 1
    return np.where(x != 0, positions, 0).max(axis=1).astype(np.int64, copy=False)


def _count_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(values)
    return {
        "nonzero": np.sum(x != 0, axis=1).astype(np.float64),
        "positive": np.sum(x > 0, axis=1).astype(np.float64),
        "negative": np.sum(x < 0, axis=1).astype(np.float64),
        "tail_length": _tail_lengths(x).astype(np.float64),
    }


@torch.no_grad()
def ddim_from_xt(
    model,
    x_t: torch.Tensor,
    condition: torch.Tensor,
    *,
    start_timestep: int,
    sampling_steps: int,
    force_zero_condition: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    start_timestep = max(0, min(int(start_timestep), int(model.diffusion.diffusion_steps) - 1))
    requested = max(1, min(int(sampling_steps), start_timestep + 1))
    timesteps = torch.linspace(start_timestep, 0, requested, device=x_t.device).round().long()
    timesteps = torch.unique_consecutive(timesteps)
    x = x_t
    encoded_condition = None
    if not force_zero_condition:
        encoded_condition = model.encode_condition(condition, force_zero_condition=False)
    for index, timestep_value in enumerate(timesteps.tolist()):
        timestep = torch.full((x.shape[0],), int(timestep_value), dtype=torch.long, device=x.device)
        predicted_noise = model.predict_noise(
            x,
            timestep,
            condition,
            force_zero_condition=force_zero_condition,
            encoded_condition=encoded_condition,
        )
        alpha_t = model.diffusion.alpha_bar[int(timestep_value)]
        x0 = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t).clamp_min(1.0e-8)
        x0 = x0.clamp(-model.diffusion.x0_clip, model.diffusion.x0_clip)
        next_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
        if next_timestep < 0:
            x = x0
            break
        alpha_next = model.diffusion.alpha_bar[next_timestep]
        direction = torch.sqrt(torch.clamp(1.0 - alpha_next, min=0.0)) * predicted_noise
        x = torch.sqrt(alpha_next) * x0 + direction
    return x


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def summarize_mode(
    *,
    mode: str,
    timestep: int,
    clean_units: np.ndarray,
    generated_units: np.ndarray,
) -> dict[str, Any]:
    clean_stats = _count_stats(clean_units)
    gen_stats = _count_stats(generated_units)
    clean_nonzero = clean_units != 0
    gen_nonzero = generated_units != 0
    clean_sign = np.sign(clean_units)
    gen_sign = np.sign(generated_units)
    original_kept = np.sum(clean_nonzero & (gen_sign == clean_sign), axis=1)
    original_removed = np.sum(clean_nonzero & ~gen_nonzero, axis=1)
    original_flipped = np.sum(clean_nonzero & gen_nonzero & (gen_sign != clean_sign), axis=1)
    padding_inserted = np.sum(~clean_nonzero & gen_nonzero, axis=1)
    l1 = np.mean(np.abs(generated_units - clean_units), axis=1)
    return {
        "mode": mode,
        "start_timestep": int(timestep),
        "count": int(clean_units.shape[0]),
        "clean_nonzero_mean": _mean(clean_stats["nonzero"]),
        "generated_nonzero_mean": _mean(gen_stats["nonzero"]),
        "generated_positive_mean": _mean(gen_stats["positive"]),
        "generated_negative_mean": _mean(gen_stats["negative"]),
        "extra_nonzero_mean": _mean(gen_stats["nonzero"] - clean_stats["nonzero"]),
        "extra_positive_mean": _mean(gen_stats["positive"] - clean_stats["positive"]),
        "extra_negative_mean": _mean(gen_stats["negative"] - clean_stats["negative"]),
        "overhead_ratio_mean": _mean((gen_stats["nonzero"] - clean_stats["nonzero"]) / np.maximum(clean_stats["nonzero"], 1.0)),
        "tail_length_mean": _mean(gen_stats["tail_length"]),
        "padding_inserted_mean": _mean(padding_inserted.astype(np.float64)),
        "original_kept_mean": _mean(original_kept.astype(np.float64)),
        "original_removed_mean": _mean(original_removed.astype(np.float64)),
        "original_flipped_mean": _mean(original_flipped.astype(np.float64)),
        "full_sequence_l1_units_mean": _mean(l1),
        "count_add_only_ok_rate": _mean(((gen_stats["positive"] >= clean_stats["positive"]) & (gen_stats["negative"] >= clean_stats["negative"])).astype(np.float64)),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.purifier_run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "checkpoints" / "best_checkpoint.pt"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "img2img_defense_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    model, payload = load_purifier_checkpoint(checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    cfg = PurifierConfig.from_mapping(payload["config"])
    rows = _read_unique_test_rows(cfg.test_manifest, int(args.max_sources))
    clean_indices = [int(row["clean_index"]) for row in rows]
    timesteps = [int(item.strip()) for item in str(args.start_timesteps).split(",") if item.strip()]
    clean_store = CleanStore(cfg.clean_path)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    all_rows: list[dict[str, Any]] = []
    try:
        for timestep in timesteps:
            mode_buffers: dict[str, list[dict[str, Any]]] = {"condition_clean": [], "zero_condition": []}
            for start in range(0, len(clean_indices), int(args.batch_size)):
                batch_indices = clean_indices[start : start + int(args.batch_size)]
                clean_norm_np = clean_store.batch(batch_indices, seq_length=int(cfg.seq_length), value_scale=float(cfg.value_scale))
                clean_norm = torch.as_tensor(clean_norm_np, dtype=torch.float32, device=device)
                t = torch.full((clean_norm.shape[0],), int(timestep), dtype=torch.long, device=device)
                noise = torch.randn(clean_norm.shape, device=device, dtype=clean_norm.dtype, generator=generator)
                x_t, _ = model.diffusion.q_sample(clean_norm, t, noise=noise)
                sampling_steps = int(args.sampling_steps or min(int(cfg.sampling_steps), int(timestep) + 1))
                for mode, force_zero in [("condition_clean", False), ("zero_condition", True)]:
                    generated_norm = ddim_from_xt(
                        model,
                        x_t,
                        clean_norm,
                        start_timestep=int(timestep),
                        sampling_steps=sampling_steps,
                        force_zero_condition=force_zero,
                        generator=generator,
                    )
                    generated_units = model.decoder.legalize_numpy(generated_norm, output_length=None)
                    clean_units = model.decoder.legalize_numpy(clean_norm, output_length=None)
                    mode_buffers[mode].append(
                        summarize_mode(
                            mode=mode,
                            timestep=int(timestep),
                            clean_units=clean_units,
                            generated_units=generated_units,
                        )
                    )
            for mode, parts in mode_buffers.items():
                merged: dict[str, Any] = {"mode": mode, "start_timestep": int(timestep), "count": int(sum(part["count"] for part in parts))}
                for key in parts[0]:
                    if key in {"mode", "start_timestep", "count"}:
                        continue
                    weights = np.asarray([part["count"] for part in parts], dtype=np.float64)
                    values = np.asarray([float(part[key]) for part in parts], dtype=np.float64)
                    merged[key] = float(np.average(values, weights=weights))
                all_rows.append(merged)
                print(json.dumps(merged, ensure_ascii=False), flush=True)
    finally:
        clean_store.close()
    write_csv(output_dir / "img2img_defense_summary.csv", all_rows)
    report = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "rows": int(len(rows)),
        "unique_sources": int(len(rows)),
        "timesteps": timesteps,
        "seed": int(args.seed),
        "sampling_steps_arg": int(args.sampling_steps),
        "note": "x0 is clean. condition_clean feeds clean as c; zero_condition disables condition.",
        "summary_csv": str((output_dir / "img2img_defense_summary.csv").resolve()),
        "rows_summary": all_rows,
    }
    write_json(output_dir / "img2img_defense_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
