"""Resume only DMMPv3 Stage 3 from an existing Stage 1/2 run."""

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

from dmmp.utils.config import DefenseConfig
from dmmp.data import load_cw_data
from dmmp.constraints.user_profiles import load_profiles
from dmmp.utils import log, resolve_device, set_seed
from dmmp.diffusion.profile_pipeline import run_v4_stage3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume DMMPv3 Stage 3 without retraining Stage 1/2.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow_diagnostic_fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Stage 3 to save the best diagnostic fallback instead of failing the hard quality gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    values = {
        name: payload[name]
        for name in DefenseConfig.__dataclass_fields__
        if name in payload and name != "profile_secret"
    }
    cfg = DefenseConfig(**values)
    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        cfg.data_root = str((REPO_ROOT / data_root).resolve())
    cfg.device = str(args.device)
    cfg.progress = bool(args.progress)
    if bool(args.allow_diagnostic_fallback):
        cfg.stage3_require_quality_gate = False
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    raw, labels, trace_ids, _, _ = load_cw_data(cfg)
    split_payload = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    splits = {name: np.asarray(indices, dtype=np.int64) for name, indices in split_payload.items()}
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    log(f"[DMMPv3 Stage3 resume] run={run_dir}, device={device}; reusing Stage 1/2 checkpoints.", True)
    result = run_v4_stage3(raw, labels, trace_ids, splits, run_dir, cfg, device, profiles)
    selected = result.get("selected", {})
    log(
        f"[DMMPv3 Stage3 resume done] label_free_pressure={float(selected.get('selection_attack_pressure', 1.0)):.4f}, "
        f"gate={float(cfg.stage3_max_label_free_attack_pressure):.4f}, valid={int(selected.get('selection_policy_valid', 0))}",
        True,
    )


if __name__ == "__main__":
    main()
