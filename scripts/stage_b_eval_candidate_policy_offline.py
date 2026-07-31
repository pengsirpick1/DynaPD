# -*- coding: utf-8 -*-
"""Offline evaluation of a learned candidate scorer on Teacher records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_b.policy_model import CandidateScoringPolicy, PolicyModelConfig
from dmmp.utils import resolve_device
from dmmp.utils.config import DEFAULT_OUTPUT_DIR

from scripts.stage_b_train_candidate_policy import TeacherRecordDataset, collate_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="stage_b_candidate_policy_offline_eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--near_epsilons", default="0.005,0.01,0.02")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    target = Path(args.output_dir) / args.run_name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_model(path: str | Path, device: torch.device) -> CandidateScoringPolicy:
    payload = torch.load(Path(path), map_location=device)
    config = PolicyModelConfig(**payload.get("config", {}))
    model = CandidateScoringPolicy(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    epsilons = _parse_floats(args.near_epsilons)
    dataset = TeacherRecordDataset(args.records_csv)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_records)
    model = _load_model(args.checkpoint, device)
    rows = []
    total_candidates = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            output = model(
                state_tensor=tensors["state_tensor"],
                state_features=tensors["state_features"],
                action_counts=tensors["action_counts"],
                action_features=tensors["action_features"],
                candidate_mask=tensors["candidate_mask"],
            )
            scores = output["scores"]
            gains = tensors["candidate_gains"]
            mask = tensors["candidate_mask"].bool()
            selected = tensors["selected_index"]
            valid = (selected >= 0) & mask.any(dim=1)
            total_candidates += int(mask.sum().item())
            ranked = torch.argsort(scores, dim=1, descending=True)
            best_gain = gains.masked_fill(~mask, -1e9).max(dim=1).values
            top1 = ranked[:, 0]
            top1_gain = gains[torch.arange(gains.shape[0], device=device), top1]
            for idx, path in enumerate(batch["record_path"]):
                if not bool(valid[idx]):
                    continue
                item = {
                    "record_path": str(path),
                    "candidate_count": int(mask[idx].sum().item()),
                    "selected_index": int(selected[idx].item()),
                    "predicted_top1": int(top1[idx].item()),
                    "oracle_best_gain": float(best_gain[idx].item()),
                    "predicted_top1_gain": float(top1_gain[idx].item()),
                    "utility_gap": float(best_gain[idx].item() - top1_gain[idx].item()),
                }
                for k in (1, 4, 8, 16):
                    kk = min(int(k), int(ranked.shape[1]))
                    topk_indices = ranked[idx, :kk]
                    topk_gain = float(gains[idx, topk_indices].max().item())
                    item[f"recall_at_{k}"] = int(bool((ranked[idx, :kk] == selected[idx]).any().item()))
                    item[f"top{k}_best_gain"] = topk_gain
                    item[f"regret_at_{k}"] = float(best_gain[idx].item() - topk_gain)
                    for eps in epsilons:
                        key = str(eps).replace(".", "p")
                        item[f"near_optimal_at_{k}_eps_{key}"] = int(topk_gain >= float(best_gain[idx].item()) - float(eps))
                rows.append(item)
    elapsed = max(time.perf_counter() - start, 1e-9)
    with (output_dir / "offline_policy_eval_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = list(rows[0].keys()) if rows else ["record_path"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "records_csv": str(args.records_csv),
        "checkpoint": str(args.checkpoint),
        "evaluated_action_records": int(len(rows)),
        "total_candidates": int(total_candidates),
        "runtime_sec": float(elapsed),
        "candidates_per_sec": float(total_candidates / elapsed),
        "records_per_sec": float(len(dataset) / elapsed),
    }
    if rows:
        for k in (1, 4, 8, 16):
            summary[f"oracle_action_recall_at_{k}"] = float(np.mean([row[f"recall_at_{k}"] for row in rows]))
            summary[f"mean_regret_at_{k}"] = float(np.mean([row[f"regret_at_{k}"] for row in rows]))
            summary[f"median_regret_at_{k}"] = float(np.median([row[f"regret_at_{k}"] for row in rows]))
            for eps in epsilons:
                key = str(eps).replace(".", "p")
                summary[f"near_optimal_recall_at_{k}_eps_{key}"] = float(np.mean([row[f"near_optimal_at_{k}_eps_{key}"] for row in rows]))
        summary["mean_teacher_student_utility_gap"] = float(np.mean([row["utility_gap"] for row in rows]))
        summary["median_teacher_student_utility_gap"] = float(np.median([row["utility_gap"] for row in rows]))
    (output_dir / "offline_policy_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
