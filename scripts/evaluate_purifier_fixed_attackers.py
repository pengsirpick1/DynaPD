"""Frozen DF/RF evaluation for clean, defended, and purified traffic.

This script is inference-only. It never trains attackers, never updates
purifier checkpoints, and never feeds attacker outputs back into the purifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import stored_npy_from_npz
from dmmp.evaluation.attack_models import build_df_input, build_rf_tam_input, make_attack_model
from dmmp.purifier.config import PurifierConfig
from dmmp.purifier.training import load_purifier_checkpoint
from dmmp.utils import resolve_device, write_csv, write_json


DEFAULT_DF_CHECKPOINT = REPO_ROOT / "results" / "dmmp2_v5_fixed_oriented_seed0_bwo30" / "attack_eval" / "fixed" / "df" / "fixed_df_checkpoint.pt"
DEFAULT_RF_CHECKPOINT = REPO_ROOT / "results" / "dmmp2_v5_fixed_oriented_seed0_bwo30" / "attack_eval" / "fixed" / "rf" / "fixed_rf_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate purifier recovery with frozen 95-class DF/RF attackers.")
    parser.add_argument("--purifier-run-dir", required=True)
    parser.add_argument("--defense-run-dir", default="", help="Legacy optional: use <defense-run-dir>/split_indices.json if present.")
    parser.add_argument("--split-indices", default="", help="Optional split_indices.json for strict source-id verification.")
    parser.add_argument("--df-checkpoint", default=str(DEFAULT_DF_CHECKPOINT))
    parser.add_argument("--rf-checkpoint", default=str(DEFAULT_RF_CHECKPOINT))
    parser.add_argument("--purified-manifest", default="", help="Optional alternate generated/purified manifest to evaluate.")
    parser.add_argument("--output-dir", default="", help="Default: <purifier-run-dir>/fixed_attacker_eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-pairs", type=int, default=0, help="Debug guard; 0 evaluates the full test-pair manifest.")
    parser.add_argument("--max-trace-length", type=int, default=5000)
    parser.add_argument("--rf-tam-num-slots", type=int, default=1800)
    parser.add_argument("--max-load-time", type=float, default=80.0)
    parser.add_argument("--sampling-seed", type=int, default=72000)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pair_key(row: dict[str, str]) -> tuple[str, str, int, int, int, int]:
    return (
        str(row["split"]),
        str(Path(row["defended_path"]).resolve()),
        int(row["defended_local_index"]),
        int(row["source_id"]),
        int(row["clean_index"]),
        int(row["variant_id"]),
    )


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

    def batch(self, indices: Iterable[int]) -> np.ndarray:
        idx = np.asarray(list(indices), dtype=np.int64)
        return np.asarray(self.x[idx], dtype=np.float32)

    def labels(self, indices: Iterable[int]) -> np.ndarray:
        idx = np.asarray(list(indices), dtype=np.int64)
        return np.asarray(self.y[idx], dtype=np.int64)

    def close(self) -> None:
        if self._payload is not None:
            self._payload.close()


class NpzCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max(1, int(max_items))
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str | Path) -> dict[str, np.ndarray]:
        text = str(Path(path).resolve())
        if text in self.cache:
            item = self.cache.pop(text)
            self.cache[text] = item
            return item
        payload = np.load(text, allow_pickle=False)
        try:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        finally:
            payload.close()
        self.cache[text] = arrays
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return arrays


def _defended_batch(rows: list[dict[str, str]], cache: NpzCache, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), int(seq_length)), dtype=np.float32)
    lengths = np.zeros(len(rows), dtype=np.int64)
    for out_index, row in enumerate(rows):
        arrays = cache.get(row["defended_path"])
        local = int(row["defended_local_index"])
        start = int(arrays["offsets"][local])
        end = int(arrays["offsets"][local + 1])
        values = np.asarray(arrays["flat"][start:end], dtype=np.float32)
        take = min(int(seq_length), len(values))
        if take:
            x[out_index, :take] = values[:take]
        lengths[out_index] = len(values)
    return x, lengths


def _tail_lengths_numpy(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D trace batch, got shape {x.shape}")
    positions = np.arange(x.shape[1], dtype=np.int64).reshape(1, -1) + 1
    return np.where(x != 0, positions, 0).max(axis=1).astype(np.int64, copy=False)


def _purified_batch(rows: list[dict[str, str]], cache: NpzCache) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for row in rows:
        payload = cache.get(row["purified_path"])
        arrays.append(np.asarray(payload["X"][int(row["purified_index"])], dtype=np.float32))
    return np.stack(arrays, axis=0)


def _attack_input(kind: str, raw: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if kind == "df":
        return build_df_input(raw, max_len=int(args.max_trace_length)).astype(np.float32)
    return build_rf_tam_input(
        raw,
        max_len=int(args.max_trace_length),
        max_load_time=float(args.max_load_time),
        num_slots=int(args.rf_tam_num_slots),
    ).astype(np.float32)


class MetricAccumulator:
    def __init__(self, classes: np.ndarray):
        self.classes = np.asarray(classes, dtype=np.int64)
        self.class_to_pos = {int(label): index for index, label in enumerate(self.classes.tolist())}
        n = len(self.classes)
        self.confusion = np.zeros((n, n), dtype=np.int64)
        self.total = 0
        self.true_conf_sum = 0.0
        self.entropy_sum = 0.0
        self.max_conf_sum = 0.0

    def update(self, logits: torch.Tensor, labels: np.ndarray) -> None:
        probs = torch.softmax(logits.detach().float().cpu(), dim=1).numpy()
        pred_pos = np.argmax(probs, axis=1)
        true_pos = np.asarray([self.class_to_pos[int(label)] for label in np.asarray(labels, dtype=np.int64)], dtype=np.int64)
        for truth, pred in zip(true_pos.tolist(), pred_pos.tolist()):
            self.confusion[int(truth), int(pred)] += 1
        self.total += int(len(labels))
        self.true_conf_sum += float(np.sum(probs[np.arange(len(labels)), true_pos]))
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1.0e-12)), axis=1)
        if probs.shape[1] > 1:
            entropy = entropy / np.log(probs.shape[1])
        self.entropy_sum += float(np.sum(entropy))
        self.max_conf_sum += float(np.sum(np.max(probs, axis=1)))

    def finalize(self) -> dict[str, float | int]:
        tp = np.diag(self.confusion).astype(np.float64)
        fp = self.confusion.sum(axis=0).astype(np.float64) - tp
        fn = self.confusion.sum(axis=1).astype(np.float64) - tp
        denom = 2.0 * tp + fp + fn
        f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        return {
            "count": int(self.total),
            "accuracy": float(tp.sum() / max(self.total, 1)),
            "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
            "true_label_confidence": float(self.true_conf_sum / max(self.total, 1)),
            "prediction_entropy": float(self.entropy_sum / max(self.total, 1)),
            "max_confidence": float(self.max_conf_sum / max(self.total, 1)),
        }


def _load_attacker(kind: str, checkpoint: Path, device: torch.device, args: argparse.Namespace):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    classes = np.asarray(payload["classes"], dtype=np.int64)
    model = make_attack_model(
        kind.upper(),
        len(classes),
        max_trace_length=int(args.max_trace_length),
        df_architecture="project",
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, classes, float(payload.get("best_val", 0.0))


def _eval_named_batches(
    name: str,
    batch_iter,
    attackers: dict[str, tuple[torch.nn.Module, np.ndarray]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    accumulators = {kind: MetricAccumulator(classes) for kind, (_, classes) in attackers.items()}
    with torch.no_grad():
        for batch_index, (raw, labels) in enumerate(batch_iter, start=1):
            labels = np.asarray(labels, dtype=np.int64)
            for kind, (model, _) in attackers.items():
                x = _attack_input(kind, raw, args)
                logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
                accumulators[kind].update(logits, labels)
            if batch_index == 1 or batch_index % 50 == 0:
                print(f"[fixed attacker eval] {name}: batch={batch_index}", flush=True)
    return {kind.upper(): acc.finalize() for kind, acc in accumulators.items()}


def _batched(rows: list[dict[str, str]], batch_size: int):
    for start in range(0, len(rows), int(batch_size)):
        yield rows[start : start + int(batch_size)]


def _clean_iter(rows: list[dict[str, str]], clean: CleanStore, batch_size: int):
    for batch in _batched(rows, batch_size):
        indices = [int(row["clean_index"]) for row in batch]
        labels = np.asarray([int(row["class_id"]) for row in batch], dtype=np.int64)
        yield clean.batch(indices), labels


def _defended_iter(rows: list[dict[str, str]], cache: NpzCache, seq_length: int, batch_size: int):
    for batch in _batched(rows, batch_size):
        x, _ = _defended_batch(batch, cache, seq_length)
        labels = np.asarray([int(row["class_id"]) for row in batch], dtype=np.int64)
        yield x, labels


def _purified_iter(rows: list[dict[str, str]], cache: NpzCache, batch_size: int):
    for batch in _batched(rows, batch_size):
        x = _purified_batch(batch, cache)
        labels = np.asarray([int(row["class_id"]) for row in batch], dtype=np.int64)
        yield x, labels


def _shuffled_indices(rows: list[dict[str, str]]) -> np.ndarray:
    source_ids = np.asarray([int(row["source_id"]) for row in rows], dtype=np.int64)
    for shift in range(1, min(len(rows), 32)):
        perm = np.roll(np.arange(len(rows), dtype=np.int64), shift)
        if np.all(source_ids[perm] != source_ids):
            return perm
    perm = np.roll(np.arange(len(rows), dtype=np.int64), 3)
    if not np.all(source_ids[perm] != source_ids):
        raise RuntimeError("Could not construct shuffled condition permutation with j != i")
    return perm


def _optional_expected_test_sources(args: argparse.Namespace, cfg: PurifierConfig) -> set[int] | None:
    candidate: Path | None = None
    if str(args.split_indices).strip():
        candidate = Path(args.split_indices).resolve()
    elif str(args.defense_run_dir).strip():
        candidate = Path(args.defense_run_dir).resolve() / "split_indices.json"
    elif str(cfg.run_dir).strip():
        maybe = Path(cfg.run_dir).resolve() / "split_indices.json"
        if maybe.is_file():
            candidate = maybe
    if candidate is None or not candidate.is_file():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    test_values = payload.get("test", [])
    return set(int(value) for value in test_values)


def _ablation_iter(
    mode: str,
    rows: list[dict[str, str]],
    permuted_rows: list[dict[str, str]],
    defended_cache: NpzCache,
    clean: CleanStore,
    purifier,
    cfg: PurifierConfig,
    device: torch.device,
    args: argparse.Namespace,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.sampling_seed))
    with torch.no_grad():
        for start in range(0, len(rows), int(args.batch_size)):
            batch = rows[start : start + int(args.batch_size)]
            condition_batch = batch if mode != "shuffled_condition" else permuted_rows[start : start + int(args.batch_size)]
            if str(getattr(cfg, "condition_source", "defended")) == "label":
                condition_raw = np.zeros((len(condition_batch), int(cfg.seq_length)), dtype=np.float32)
                condition_labels = torch.as_tensor([int(row["class_id"]) for row in condition_batch], dtype=torch.long, device=device)
            else:
                condition_raw, _ = _defended_batch(condition_batch, defended_cache, int(cfg.seq_length))
                condition_labels = torch.as_tensor([int(row["class_id"]) for row in condition_batch], dtype=torch.long, device=device)
            output_length_policy = str(getattr(cfg, "output_length_policy", "model")).strip().lower()
            if output_length_policy == "defended":
                _, output_lengths = _defended_batch(batch, defended_cache, int(cfg.seq_length))
                output_lengths_tensor = torch.as_tensor(output_lengths, dtype=torch.long)
            elif output_length_policy == "clean":
                clean_raw = clean.batch([int(row["clean_index"]) for row in batch])
                output_lengths_tensor = torch.as_tensor(_tail_lengths_numpy(clean_raw), dtype=torch.long)
            else:
                output_lengths_tensor = None
            defended = torch.as_tensor(condition_raw / float(cfg.value_scale), dtype=torch.float32, device=device)
            sampled = purifier.sample(
                defended,
                labels=condition_labels,
                sampling_steps=int(cfg.sampling_steps),
                force_zero_condition=(mode == "zero_condition"),
                generator=generator,
            )
            x = purifier.decoder.legalize_numpy(sampled, output_length=output_lengths_tensor)
            labels = np.asarray([int(row["class_id"]) for row in batch], dtype=np.int64)
            yield x, labels


def _recovery(clean_acc: float, defended_acc: float, purified_acc: float) -> float | None:
    denom = float(clean_acc) - float(defended_acc)
    if abs(denom) < 1.0e-8:
        return None
    return (float(purified_acc) - float(defended_acc)) / denom


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Frozen DF/RF Purifier Recovery Evaluation",
        "",
        f"- inference only: `{int(report['inference_only'])}`",
        f"- test pairs: `{report['test_pair_count']}`",
        f"- test sources: `{report['test_source_count']}`",
        f"- split indices verified: `{int(report['split_indices_verified'])}`",
        f"- output length policy: `{report['output_length_policy']}`",
        "",
        "| Input | DF Acc | DF Macro-F1 | RF Acc | RF Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["clean_test_pair_expanded", "defended_test", "purified_test"]:
        row = report["main"][name]
        lines.append(
            f"| {name} | {row['DF']['accuracy']:.6f} | {row['DF']['macro_f1']:.6f} | "
            f"{row['RF']['accuracy']:.6f} | {row['RF']['macro_f1']:.6f} |"
        )
    lines.extend(["", "## Recovery", "", "| Attacker | Purified - Defended | Recovery Ratio |", "|---|---:|---:|"])
    for kind, row in report["recovery"].items():
        ratio = row["accuracy_recovery_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        lines.append(f"| {kind} | {row['purified_minus_defended']:.6f} | {ratio_text} |")
    if report.get("ablations"):
        lines.extend(["", "## Condition Ablations", "", "| Input | DF Acc | RF Acc |", "|---|---:|---:|"])
        for name, row in report["ablations"].items():
            lines.append(f"| {name} | {row['DF']['accuracy']:.6f} | {row['RF']['accuracy']:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.purifier_run_dir).resolve()
    checkpoint = run_dir / "checkpoints" / "best_checkpoint.pt"
    device = resolve_device(str(args.device))
    purifier, purifier_payload = load_purifier_checkpoint(checkpoint, device)
    for parameter in purifier.parameters():
        parameter.requires_grad_(False)
    purifier.eval()
    cfg = PurifierConfig.from_mapping(purifier_payload["config"])
    if args.defense_run_dir:
        cfg.run_dir = str(Path(args.defense_run_dir).resolve())
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "fixed_attacker_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_rows = _read_csv(cfg.test_manifest)
    test_rows = [row for row in test_rows if row["split"] == "test"]
    if int(args.max_pairs) > 0:
        test_rows = test_rows[: int(args.max_pairs)]
    source_ids = {int(row["source_id"]) for row in test_rows}
    expected_test_sources = _optional_expected_test_sources(args, cfg)
    if expected_test_sources is not None and int(args.max_pairs) <= 0 and source_ids != expected_test_sources:
        raise RuntimeError("Test manifest source_ids do not match split_indices.json test sources")

    purified_manifest = Path(args.purified_manifest).resolve() if str(args.purified_manifest).strip() else run_dir / "manifests" / "purified_dataset_manifest.csv"
    purified_rows_all = [row for row in _read_csv(purified_manifest) if row["split"] == "test"]
    purified_by_key = {_pair_key(row): row for row in purified_rows_all}
    purified_rows = []
    for row in test_rows:
        key = _pair_key(row)
        if key not in purified_by_key:
            raise RuntimeError(f"Missing purified row for test pair: {key}")
        purified_rows.append(purified_by_key[key])

    attackers = {}
    attacker_meta = {}
    for kind, checkpoint_path in {"df": Path(args.df_checkpoint), "rf": Path(args.rf_checkpoint)}.items():
        model, classes, best_val = _load_attacker(kind, checkpoint_path, device, args)
        attackers[kind] = (model, classes)
        attacker_meta[kind.upper()] = {"checkpoint": str(checkpoint_path.resolve()), "classes": int(len(classes)), "best_val_accuracy": best_val}

    clean = CleanStore(cfg.clean_path)
    defended_cache = NpzCache(max_items=4)
    purified_cache = NpzCache(max_items=8)
    try:
        main_metrics = {
            "clean_test_pair_expanded": _eval_named_batches("clean_test_pair_expanded", _clean_iter(test_rows, clean, int(args.batch_size)), attackers, device, args),
            "defended_test": _eval_named_batches("defended_test", _defended_iter(test_rows, defended_cache, int(cfg.seq_length), int(args.batch_size)), attackers, device, args),
            "purified_test": _eval_named_batches("purified_test", _purified_iter(purified_rows, purified_cache, int(args.batch_size)), attackers, device, args),
        }
        ablations: dict[str, Any] = {}
        if not bool(args.skip_ablations):
            perm = _shuffled_indices(test_rows)
            permuted_rows = [test_rows[int(index)] for index in perm.tolist()]
            for mode in ["correct_condition_resampled", "shuffled_condition", "zero_condition"]:
                print(f"[fixed attacker eval] ablation start: {mode}", flush=True)
                ablations[mode] = _eval_named_batches(
                    mode,
                    _ablation_iter(mode, test_rows, permuted_rows, defended_cache, clean, purifier, cfg, device, args),
                    attackers,
                    device,
                    args,
                )
    finally:
        clean.close()

    recovery = {}
    for kind in ["DF", "RF"]:
        clean_acc = float(main_metrics["clean_test_pair_expanded"][kind]["accuracy"])
        defended_acc = float(main_metrics["defended_test"][kind]["accuracy"])
        purified_acc = float(main_metrics["purified_test"][kind]["accuracy"])
        recovery[kind] = {
            "clean_accuracy": clean_acc,
            "defended_accuracy": defended_acc,
            "purified_accuracy": purified_acc,
            "purified_minus_defended": purified_acc - defended_acc,
            "accuracy_recovery_ratio": _recovery(clean_acc, defended_acc, purified_acc),
            "purified_beats_defended": purified_acc > defended_acc,
        }

    report = {
        "inference_only": True,
        "no_attacker_training": True,
        "no_loss_backward": True,
        "purifier_checkpoint_selection_unchanged": True,
        "purifier_checkpoint": str(checkpoint.resolve()),
        "purified_manifest": str(purified_manifest.resolve()),
        "test_pair_count": int(len(test_rows)),
        "test_source_count": int(len(source_ids)),
        "split_indices_verified": expected_test_sources is not None,
        "output_length_policy": str(getattr(cfg, "output_length_policy", "model")),
        "attacker_meta": attacker_meta,
        "main": main_metrics,
        "recovery": recovery,
        "ablations": ablations,
        "ablation_note": "Ablation samples are generated online with the same sampler seed per mode and use the purifier output_length_policy.",
    }
    write_json(output_dir / "fixed_attacker_recovery_report.json", report)
    rows = []
    for name, metrics in main_metrics.items():
        row = {"input": name}
        for kind in ["DF", "RF"]:
            row[f"{kind.lower()}_accuracy"] = metrics[kind]["accuracy"]
            row[f"{kind.lower()}_macro_f1"] = metrics[kind]["macro_f1"]
            row[f"{kind.lower()}_entropy"] = metrics[kind]["prediction_entropy"]
        rows.append(row)
    write_csv(output_dir / "fixed_attacker_recovery_main.csv", rows)
    if ablations:
        rows = []
        for name, metrics in ablations.items():
            row = {"input": name}
            for kind in ["DF", "RF"]:
                row[f"{kind.lower()}_accuracy"] = metrics[kind]["accuracy"]
                row[f"{kind.lower()}_macro_f1"] = metrics[kind]["macro_f1"]
            rows.append(row)
        write_csv(output_dir / "fixed_attacker_condition_ablations.csv", rows)
    _write_markdown(output_dir / "fixed_attacker_recovery_report.md", report)
    print(json.dumps({"output": str((output_dir / "fixed_attacker_recovery_report.json").resolve()), "recovery": recovery, "ablations": ablations}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
