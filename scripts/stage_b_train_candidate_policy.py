# -*- coding: utf-8 -*-
"""Train the learned Stage B candidate-scoring policy from Teacher records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynapd.stage_b.policy_data import ACTION_FEATURE_NAMES, STATE_FEATURE_NAMES, load_record
from dynapd.stage_b.policy_model import CandidateScoringPolicy, PolicyModelConfig
from dynapd.utils import resolve_device, set_seed
from dynapd.utils.config import DEFAULT_OUTPUT_DIR

POLICY_INPUT_FIELDS = [
    "state_tensor",
    "state_features",
    "action_counts",
    "action_features",
    "candidate_mask",
]
POLICY_TRAINING_TARGET_FIELDS = [
    "candidate_gains",
    "selected_index",
    "stop_target",
]
EVAL_OR_TRACEABILITY_ONLY_FIELDS = [
    "true_label_for_eval_only",
    "label",
    "labels",
    "source_index",
    "sample_index",
    "sample_id",
    "pred_label",
    "pred_labels",
    "current_prob",
    "original_prob",
]


class TeacherRecordDataset(Dataset):
    def __init__(self, records_csv: str | Path, *, include_stop_records: bool = True) -> None:
        self.records_csv = Path(records_csv)
        with self.records_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not include_stop_records:
            rows = [row for row in rows if int(float(row.get("stop_target", 0))) == 0]
        self.rows = rows
        if not self.rows:
            raise ValueError(f"No teacher records found in {self.records_csv}.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        record = load_record(row["record_path"])
        if "action_counts" in record:
            action_counts = np.asarray(record["action_counts"], dtype=np.float32)
        elif "action_count_shape" in record:
            shape = tuple(int(item) for item in np.asarray(record["action_count_shape"], dtype=np.int64).tolist())
            action_counts = np.zeros(shape, dtype=np.float32)
            action_idx = np.asarray(record.get("action_sparse_action", []), dtype=np.int64)
            direction = np.asarray(record.get("action_sparse_direction", []), dtype=np.int64)
            bin_index = np.asarray(record.get("action_sparse_bin", []), dtype=np.int64)
            count = np.asarray(record.get("action_sparse_count", []), dtype=np.float32)
            if action_idx.size:
                action_counts[action_idx, direction, bin_index] = count
        else:
            raise KeyError(f"Record {row['record_path']} does not contain dense or sparse action counts.")
        return {
            "state_tensor": np.asarray(record["state_tensor"], dtype=np.float32),
            "state_features": np.asarray(record["state_features"], dtype=np.float32),
            "action_counts": action_counts,
            "action_features": np.asarray(record["action_features"], dtype=np.float32),
            "candidate_gains": np.asarray(record["candidate_gains"], dtype=np.float32),
            "selected_index": int(np.asarray(record["selected_index"]).item()),
            "stop_target": float(np.asarray(record["stop_target"]).item()),
            "record_path": row["record_path"],
        }


def collate_records(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = len(items)
    width = int(items[0]["state_tensor"].shape[-1])
    state_channels = int(items[0]["state_tensor"].shape[0])
    state_feature_dim = int(items[0]["state_features"].shape[-1])
    action_feature_dim = len(ACTION_FEATURE_NAMES)
    max_candidates = max(max(1, int(item["action_counts"].shape[0])) for item in items)
    state_tensor = np.zeros((batch, state_channels, width), dtype=np.float32)
    state_features = np.zeros((batch, state_feature_dim), dtype=np.float32)
    action_counts = np.zeros((batch, max_candidates, 2, width), dtype=np.float32)
    action_features = np.zeros((batch, max_candidates, action_feature_dim), dtype=np.float32)
    gains = np.zeros((batch, max_candidates), dtype=np.float32)
    candidate_mask = np.zeros((batch, max_candidates), dtype=np.bool_)
    selected_index = np.full((batch,), -1, dtype=np.int64)
    stop_target = np.zeros((batch,), dtype=np.float32)
    paths = []
    for idx, item in enumerate(items):
        n = int(item["action_counts"].shape[0])
        state_tensor[idx] = item["state_tensor"]
        state_features[idx] = item["state_features"]
        if n > 0:
            action_counts[idx, :n] = item["action_counts"]
            action_features[idx, :n] = item["action_features"]
            gains[idx, :n] = item["candidate_gains"]
            candidate_mask[idx, :n] = True
        selected_index[idx] = int(item["selected_index"])
        stop_target[idx] = float(item["stop_target"])
        paths.append(item["record_path"])
    return {
        "state_tensor": torch.from_numpy(state_tensor),
        "state_features": torch.from_numpy(state_features),
        "action_counts": torch.from_numpy(action_counts),
        "action_features": torch.from_numpy(action_features),
        "candidate_gains": torch.from_numpy(gains),
        "candidate_mask": torch.from_numpy(candidate_mask),
        "selected_index": torch.from_numpy(selected_index),
        "stop_target": torch.from_numpy(stop_target),
        "record_path": paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_csv", required=True)
    parser.add_argument("--val_records_csv", default="")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="stage_b_candidate_policy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--lambda_choice", type=float, default=1.0)
    parser.add_argument("--lambda_gain", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.25)
    parser.add_argument("--lambda_stop", type=float, default=0.25)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    target = Path(args.output_dir) / args.run_name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = F.huber_loss(pred, target, reduction="none")
    weights = mask.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _choice_loss(scores: torch.Tensor, selected: torch.Tensor, stop_target: torch.Tensor) -> torch.Tensor:
    valid = (selected >= 0) & (stop_target < 0.5)
    if not bool(valid.any()):
        return scores.new_tensor(0.0)
    return F.cross_entropy(scores[valid], selected[valid])


def _rank_loss(scores: torch.Tensor, gains: torch.Tensor, mask: torch.Tensor, margin: float) -> torch.Tensor:
    valid_counts = mask.sum(dim=1)
    valid = valid_counts >= 2
    if not bool(valid.any()):
        return scores.new_tensor(0.0)
    masked_gains = gains.masked_fill(~mask.bool(), -1e9)
    best_idx = masked_gains.argmax(dim=1)
    worst_idx = gains.masked_fill(~mask.bool(), 1e9).argmin(dim=1)
    rows = torch.arange(scores.shape[0], device=scores.device)
    best_scores = scores[rows, best_idx]
    worst_scores = scores[rows, worst_idx]
    values = F.relu(float(margin) - best_scores + worst_scores)
    return values[valid].mean()


def _metrics(scores: torch.Tensor, gains: torch.Tensor, mask: torch.Tensor, selected: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        valid = (selected >= 0) & mask.any(dim=1)
        if not bool(valid.any()):
            return {"choice_acc": 0.0, "recall_at_4": 0.0, "utility_gap": 0.0}
        pred = scores.argmax(dim=1)
        choice_acc = (pred[valid] == selected[valid]).float().mean().item()
        k = min(4, scores.shape[1])
        topk = torch.topk(scores, k=k, dim=1).indices
        recall4 = (topk[valid] == selected[valid, None]).any(dim=1).float().mean().item()
        best_gain = gains.masked_fill(~mask.bool(), -1e9).max(dim=1).values
        pred_gain = gains[torch.arange(gains.shape[0], device=gains.device), pred]
        utility_gap = (best_gain[valid] - pred_gain[valid]).mean().item()
        return {"choice_acc": float(choice_acc), "recall_at_4": float(recall4), "utility_gap": float(utility_gap)}


def _run_epoch(
    model: CandidateScoringPolicy,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {"loss": 0.0, "choice": 0.0, "gain": 0.0, "rank": 0.0, "stop": 0.0, "choice_acc": 0.0, "recall_at_4": 0.0, "utility_gap": 0.0}
    count = 0
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
        stop_logit = output["stop_logit"]
        gains = tensors["candidate_gains"]
        mask = tensors["candidate_mask"].bool()
        choice = _choice_loss(scores, tensors["selected_index"], tensors["stop_target"])
        gain = _masked_huber(scores, gains, mask)
        rank = _rank_loss(scores, gains, mask, float(args.rank_margin))
        stop = F.binary_cross_entropy_with_logits(stop_logit, tensors["stop_target"])
        loss = (
            float(args.lambda_choice) * choice
            + float(args.lambda_gain) * gain
            + float(args.lambda_rank) * rank
            + float(args.lambda_stop) * stop
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        batch_size = int(tensors["state_tensor"].shape[0])
        metric = _metrics(scores, gains, mask, tensors["selected_index"])
        for key, value in {"loss": loss, "choice": choice, "gain": gain, "rank": rank, "stop": stop}.items():
            totals[key] += float(value.detach().cpu().item()) * batch_size
        for key, value in metric.items():
            totals[key] += float(value) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    dataset = TeacherRecordDataset(args.records_csv)
    if str(args.val_records_csv):
        train_set = dataset
        val_set = TeacherRecordDataset(args.val_records_csv)
        train_n = len(train_set)
        val_n = len(val_set)
    else:
        if float(args.val_fraction) <= 0.0:
            val_n = 0
        else:
            val_n = max(1, int(round(len(dataset) * float(args.val_fraction)))) if len(dataset) >= 5 else max(0, len(dataset) // 3)
        train_n = len(dataset) - val_n
        generator = torch.Generator().manual_seed(int(args.seed))
        train_set, val_set = random_split(dataset, [train_n, val_n], generator=generator) if val_n else (dataset, None)
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_records)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_records) if val_set else None
    config = PolicyModelConfig(hidden_dim=int(args.hidden_dim), dropout=float(args.dropout))
    model = CandidateScoringPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer, device=device, args=args)
        val_metrics = _run_epoch(model, val_loader, optimizer=None, device=device, args=args) if val_loader else {}
        row = {f"train_{key}": value for key, value in train_metrics.items()}
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["epoch"] = int(epoch)
        history.append(row)
        metric = float(val_metrics.get("loss", train_metrics["loss"]))
        if metric < best_val:
            best_val = metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "action_feature_names": ACTION_FEATURE_NAMES,
                    "state_feature_names": STATE_FEATURE_NAMES,
                    "policy_input_fields": POLICY_INPUT_FIELDS,
                    "policy_training_target_fields": POLICY_TRAINING_TARGET_FIELDS,
                    "eval_or_traceability_only_fields": EVAL_OR_TRACEABILITY_ONLY_FIELDS,
                    "args": vars(args),
                    "epoch": int(epoch),
                    "best_metric": float(best_val),
                },
                output_dir / "best_policy.pt",
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        keys = list(dict.fromkeys(key for row in history for key in row.keys()))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "records_csv": str(args.records_csv),
                "val_records_csv": str(args.val_records_csv),
                "samples": int(len(dataset)),
                "train_samples": int(train_n),
                "val_samples": int(val_n),
                "checkpoint": str(output_dir / "best_policy.pt"),
                "history": str(output_dir / "training_history.csv"),
                "config": asdict(config),
                "policy_input_fields": POLICY_INPUT_FIELDS,
                "policy_training_target_fields": POLICY_TRAINING_TARGET_FIELDS,
                "eval_or_traceability_only_fields": EVAL_OR_TRACEABILITY_ONLY_FIELDS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
