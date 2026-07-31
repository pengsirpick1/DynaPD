"""Run Stage A deletion-style DynaMask over a TAM subset."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a import DynMaskConfig, load_stage_a_attacker, load_stage_a_tam_dataset, optimize_deletion_masks
from dmmp.stage_a.viz import plot_sample_mask
from dmmp.utils import resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR


DEFAULT_FIXED_DF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt"
DEFAULT_FIXED_RF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt"


def _default_checkpoint(attacker: str) -> str:
    if str(attacker).lower() == "df":
        return DEFAULT_FIXED_DF
    if str(attacker).lower() == "rf":
        return DEFAULT_FIXED_RF
    raise ValueError(f"Unsupported attacker {attacker!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--samples_per_class", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--lambda_l1", type=float, default=1e-3)
    parser.add_argument("--lambda_tv", type=float, default=1e-2)
    parser.add_argument("--lambda_area", type=float, default=0.0)
    parser.add_argument("--target_keep_ratio", type=float, default=0.10)
    parser.add_argument("--baseline_kernel", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--plot_limit", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage_a_{args.attacker}_w{args.width}_n{args.max_samples}_s{args.steps}_{stamp}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_sample_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    run_dir = _run_dir(args)
    mask_dir = run_dir / f"stage_a_masks_{args.attacker}"
    sample_dir = mask_dir / "samples"
    figure_dir = run_dir / "stage_a_figures" / "sample_masks"
    write_json(run_dir / "stage_a_config.json", vars(args) | {"checkpoint": checkpoint, "device_resolved": str(device)})

    dataset = load_stage_a_tam_dataset(
        args.data_root,
        split=str(args.split),
        width=int(args.width),
        seed=int(args.seed),
        max_samples=int(args.max_samples),
        samples_per_class=int(args.samples_per_class),
        max_classes=int(args.max_classes),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        max_trace_length=int(args.max_trace_length),
        max_load_time=float(args.max_load_time),
    )
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=95,
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    dyn_cfg = DynMaskConfig(
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        lambda_l1=float(args.lambda_l1),
        lambda_tv=float(args.lambda_tv),
        lambda_area=float(args.lambda_area),
        target_keep_ratio=float(args.target_keep_ratio),
        baseline_kernel=int(args.baseline_kernel),
        log_every=max(1, int(args.steps) // 4),
    )
    rows: list[dict] = []
    all_payload: dict[str, list[np.ndarray]] = {
        "tam": [],
        "mask": [],
        "tam_base": [],
        "tam_masked": [],
        "tam_keypoint_only": [],
        "pred_prob": [],
        "masked_prob": [],
    }
    labels, source_indices, sample_ids = [], [], []
    total = int(dataset.tam.shape[0])
    for start in range(0, total, max(1, int(args.batch_size))):
        end = min(start + max(1, int(args.batch_size)), total)
        if args.progress:
            print(f"[Stage A] optimizing batch {start}:{end} / {total}", flush=True)
        result = optimize_deletion_masks(
            dataset.tam[start:end],
            attacker,
            dyn_cfg,
            device=device,
            progress=bool(args.progress),
        )
        for local, absolute in enumerate(range(start, end)):
            sample_id = str(dataset.sample_ids[absolute])
            label = int(dataset.labels[absolute])
            source_index = int(dataset.source_indices[absolute])
            row = {
                "sample_id": sample_id,
                "source_index": source_index,
                "true_label": label,
                "pred_label": int(result.pred_label[local]),
                "masked_pred_label": int(result.masked_pred_label[local]),
                "pred_confidence": float(result.pred_prob[local, result.pred_label[local]]),
                "masked_original_confidence": float(result.masked_prob[local, result.pred_label[local]]),
                "masked_top_confidence": float(result.masked_prob[local, result.masked_pred_label[local]]),
                "js_div": float(result.js_div[local]),
                "entropy_gain": float(result.entropy_gain[local]),
                "top1_drop": float(result.top1_drop[local]),
                "mask_mean": float(result.mask[local].mean()),
                "mask_max": float(result.mask[local].max()),
            }
            rows.append(row)
            safe_id = f"{absolute:06d}_{sample_id}".replace(":", "_").replace("\\", "_").replace("/", "_")
            _save_sample_npz(
                sample_dir / f"{safe_id}.npz",
                {
                    "sample_id": np.asarray(sample_id),
                    "source_index": np.asarray(source_index, dtype=np.int64),
                    "true_label": np.asarray(label, dtype=np.int64),
                    "pred_label": np.asarray(row["pred_label"], dtype=np.int64),
                    "masked_pred_label": np.asarray(row["masked_pred_label"], dtype=np.int64),
                    "tam": result.tam[local],
                    "mask": result.mask[local],
                    "tam_base": result.tam_base[local],
                    "tam_masked": result.tam_masked[local],
                    "tam_keypoint_only": result.tam_keypoint_only[local],
                    "pred_logits": result.pred_logits[local],
                    "pred_prob": result.pred_prob[local],
                    "masked_logits": result.masked_logits[local],
                    "masked_prob": result.masked_prob[local],
                    "js_div": np.asarray(row["js_div"], dtype=np.float32),
                    "entropy_gain": np.asarray(row["entropy_gain"], dtype=np.float32),
                    "top1_drop": np.asarray(row["top1_drop"], dtype=np.float32),
                },
            )
            if absolute < int(args.plot_limit):
                plot_sample_mask(
                    result.tam[local],
                    result.mask[local],
                    result.tam_masked[local],
                    result.tam_keypoint_only[local],
                    result.pred_prob[local],
                    result.masked_prob[local],
                    figure_dir / f"{safe_id}.png",
                )
            for key in all_payload:
                all_payload[key].append(getattr(result, key)[local])
            labels.append(np.asarray(label, dtype=np.int64))
            source_indices.append(np.asarray(source_index, dtype=np.int64))
            sample_ids.append(np.asarray(sample_id))

    archive = {
        key: np.stack(value, axis=0).astype(np.float32)
        for key, value in all_payload.items()
    }
    archive["labels"] = np.asarray(labels, dtype=np.int64)
    archive["source_indices"] = np.asarray(source_indices, dtype=np.int64)
    archive["sample_ids"] = np.asarray(sample_ids).astype(str)
    archive["pred_labels"] = np.asarray([row["pred_label"] for row in rows], dtype=np.int64)
    archive["masked_pred_labels"] = np.asarray([row["masked_pred_label"] for row in rows], dtype=np.int64)
    archive["js_div"] = np.asarray([row["js_div"] for row in rows], dtype=np.float32)
    archive["entropy_gain"] = np.asarray([row["entropy_gain"] for row in rows], dtype=np.float32)
    archive["top1_drop"] = np.asarray([row["top1_drop"] for row in rows], dtype=np.float32)
    np.savez_compressed(mask_dir / "all_masks.npz", **archive)
    write_csv(mask_dir / "sample_metrics.csv", rows)
    summary = {
        "run_dir": str(run_dir),
        "mask_archive": str(mask_dir / "all_masks.npz"),
        "samples": len(rows),
        "attacker": str(args.attacker),
        "data_source": dataset.data_source,
        "split": dataset.split,
        "width": int(args.width),
        "mean_js_div": float(np.mean(archive["js_div"])) if rows else 0.0,
        "mean_entropy_gain": float(np.mean(archive["entropy_gain"])) if rows else 0.0,
        "mean_top1_drop": float(np.mean(archive["top1_drop"])) if rows else 0.0,
        "mean_mask": float(np.mean(archive["mask"])) if rows else 0.0,
        "changed_pred_rate": float(np.mean(archive["pred_labels"] != archive["masked_pred_labels"])) if rows else 0.0,
        "figures": str(figure_dir),
    }
    write_json(mask_dir / "summary.json", summary)
    print(f"Stage A DynaMask complete: {run_dir}")
    print(f"mask_archive={mask_dir / 'all_masks.npz'}")


if __name__ == "__main__":
    main()
