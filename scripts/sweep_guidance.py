"""Sweep DMMPv3 inference guidance against rendered DF/RF surrogates without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.utils.config import DefenseConfig, parse_csv_floats, parse_csv_ints
from dmmp.data import load_cw_data
from dmmp.constraints.user_profiles import load_profiles
from dmmp.utils import log, resolve_device, set_seed, write_csv, write_json
from dmmp.diffusion.profile_pipeline import generate_v4_ragged_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep DMMPv3 DDIM hard-guidance settings using existing checkpoints.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--guidance_weights", default="0.10,0.30,0.60,1.00")
    parser.add_argument("--guidance_last_steps", default="4,10,20")
    parser.add_argument("--refine_steps", default="50")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--budget", type=float, default=0.30)
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    values = {name: payload[name] for name in DefenseConfig.__dataclass_fields__ if name in payload and name != "profile_secret"}
    cfg = DefenseConfig(**values)
    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        cfg.data_root = str((REPO_ROOT / data_root).resolve())
    cfg.device = str(args.device)
    cfg.progress = bool(args.progress)
    cfg.stage3_repeats = 1
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    raw, labels, trace_ids, _, _ = load_cw_data(cfg)
    split_payload = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    val_indices = np.asarray(split_payload["val"], dtype=np.int64)
    if int(args.samples) > 0 and len(val_indices) > int(args.samples):
        rng = np.random.default_rng(int(cfg.seed) + 701)
        val_indices = np.sort(rng.choice(val_indices, size=int(args.samples), replace=False)).astype(np.int64)
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    profile = profiles["validation"][0]
    rows = []
    for weight in parse_csv_floats(args.guidance_weights):
        for last_steps in parse_csv_ints(args.guidance_last_steps):
            for refine_steps in parse_csv_ints(args.refine_steps):
                cfg.guidance_weight = float(weight)
                cfg.guidance_last_steps = int(last_steps)
                cfg.refine_steps = int(refine_steps)
                _, _, metrics = generate_v4_ragged_dataset(
                    raw,
                    labels,
                    trace_ids,
                    val_indices,
                    run_dir,
                    cfg,
                    profile=profile,
                    visit_namespace="pareto-inference-margin-sweep",
                    budget=float(args.budget),
                    keep_ratio=float(args.keep_ratio),
                    output_npz=None,
                    device=device,
                )
                row = {
                    "guidance_weight": float(weight),
                    "guidance_last_steps": int(last_steps),
                    "refine_steps": int(refine_steps),
                    **metrics,
                }
                rows.append(row)
                log(
                    f"[DMMPv3 guidance sweep] weight={weight:.2f}, last_steps={last_steps}, refine={refine_steps}, "
                    f"prefix_pressure={float(metrics.get('prefix_policy_label_free_attack_pressure', 1.0)):.4f}, "
                    f"rendered_pressure={float(metrics.get('surrogate_label_free_attack_pressure', 1.0)):.4f}, "
                    f"diagnostic_worst_acc={float(metrics.get('surrogate_defended_accuracy', 1.0)):.4f}",
                    True,
                )
    rows.sort(
        key=lambda row: (
            float(row.get("prefix_policy_label_free_attack_pressure", 1.0)),
            float(row.get("visible_dummy_overhead", 1.0)),
        )
    )
    output_dir = run_dir / "stage3_inference_margin_sweep"
    write_csv(output_dir / "results.csv", rows)
    write_json(output_dir / "results.json", {"rows": rows, "best": rows[0] if rows else {}})
    if rows:
        best = rows[0]
        log(
            f"[DMMPv3 guidance sweep best] weight={best['guidance_weight']:.2f}, "
            f"last_steps={best['guidance_last_steps']}, refine={best['refine_steps']}, "
            f"prefix_pressure={float(best.get('prefix_policy_label_free_attack_pressure', 1.0)):.4f}, "
            f"rendered_pressure={float(best.get('surrogate_label_free_attack_pressure', 1.0)):.4f}, "
            f"diagnostic_worst_acc={float(best.get('surrogate_defended_accuracy', 1.0)):.4f}",
            True,
        )


if __name__ == "__main__":
    main()
