"""Run small purifier sanity checks before full training."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.purifier import load_purifier_config
from dmmp.purifier.training import train_purifier
from dmmp.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run correct/shuffled/zero/unconditional purifier sanity checks.")
    parser.add_argument("--config", default="configs/purifier/smoke_test.yaml")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--train-sources", type=int, default=512)
    parser.add_argument("--validation-sources", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--condition-channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metric(result: dict) -> float:
    return float(result["checkpoint_selection"]["validation_metric"]["source_l1"])


def main() -> None:
    args = parse_args()
    base = load_purifier_config(args.config)
    root = Path(args.output_root).resolve() if args.output_root else Path(base.output_root) / f"sanity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    modes = {
        "correct": ("correct", "conditioned"),
        "shuffled": ("shuffled", "conditioned"),
        "zero": ("correct", "zero"),
        "unconditional": ("correct", "unconditional"),
    }
    results: dict[str, dict] = {}
    for name, (pairing_mode, condition_mode) in modes.items():
        cfg = replace(
            base,
            output_root=str(root),
            run_name=name,
            max_train_sources=int(args.train_sources),
            max_validation_sources=int(args.validation_sources),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            diffusion_steps=int(args.diffusion_steps),
            sampling_steps=int(args.sampling_steps),
            hidden_channels=int(args.hidden_channels),
            condition_channels=int(args.condition_channels),
            pairing_mode=pairing_mode,
            condition_mode=condition_mode,
            device=str(args.device),
            log_every=20,
        )
        print(f"[purifier sanity] start {name}: pairing={pairing_mode}, condition={condition_mode}", flush=True)
        results[name] = train_purifier(cfg, overwrite=bool(args.overwrite))
        gc.collect()
    correct = _metric(results["correct"])
    comparisons = {
        name: {
            "source_l1": _metric(result),
            "correct_is_better": correct < _metric(result),
            "run_dir": result["run_dir"],
        }
        for name, result in results.items()
        if name != "correct"
    }
    verdict = "PASS" if all(item["correct_is_better"] for item in comparisons.values()) else "UNCERTAIN"
    report = {
        "verdict": verdict,
        "correct_source_l1": correct,
        "comparisons": comparisons,
        "runs": {name: result["run_dir"] for name, result in results.items()},
        "note": "Small sanity runs are a gate for wiring, not final purifier effectiveness.",
    }
    output = root / "sanity_check_results.json"
    write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
