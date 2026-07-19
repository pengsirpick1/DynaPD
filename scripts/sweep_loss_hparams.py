"""Lightweight training-time hyperparameter sweep for DMMPv3 loss weights."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in str(value).replace(";", ",").split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in str(value).replace(";", ",").split(",") if item.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_run_fragment(value: float | int) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight grid/random search over DMMPv3 training-time loss "
            "hyperparameters. Each trial is a short independent run."
        )
    )
    parser.add_argument("--sweep_name", default="")
    parser.add_argument("--search", choices=["grid", "random"], default="grid")
    parser.add_argument("--max_trials", type=int, default=0, help="0 means run every generated trial.")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry_incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a fresh *_retryNNN run name when a previous incomplete trial directory is non-empty.",
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--data_root", default=str(REPO_ROOT / "datasets" / "CW"))
    parser.add_argument("--output_dir", default=str(RESULTS_ROOT))
    parser.add_argument("--max_classes", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=800)
    parser.add_argument("--max_generation_traces", type=int, default=64)
    parser.add_argument("--encoder_train_samples", type=int, default=512)
    parser.add_argument("--surrogate_train_samples", type=int, default=800)
    parser.add_argument("--surrogate_val_samples", type=int, default=200)
    parser.add_argument("--surrogate_epochs", type=int, default=6)
    parser.add_argument("--surrogate_patience", type=int, default=2)
    parser.add_argument(
        "--surrogate_min_val_accuracy",
        type=float,
        default=0.65,
        help=(
            "Relaxed Stage 1 strong-surrogate gate for lightweight sweeps. "
            "Full defense runs should keep train_defense.py's stricter default."
        ),
    )
    parser.add_argument("--candidate_epochs", type=int, default=2)
    parser.add_argument("--encoder_epochs", type=int, default=2)
    parser.add_argument("--diffusion_train_steps", type=int, default=300)
    parser.add_argument("--guidance_train_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pareto_samples", type=int, default=64)
    parser.add_argument("--stage3_fixed_probe_samples", type=int, default=96)
    parser.add_argument("--stage3_fixed_probe_train_samples", type=int, default=512)
    parser.add_argument("--stage3_fixed_probe_val_samples", type=int, default=160)
    parser.add_argument("--stage3_fixed_probe_epochs", type=int, default=1)

    parser.add_argument("--preference_weights", default="0,0.003,0.01")
    parser.add_argument("--defense_soft_objective_scales", default="0.02,0.05")
    parser.add_argument("--defense_soft_utility_weights", default="0.03,0.05")
    parser.add_argument("--prefix_hidden_align_weights", default="0,0.01")
    parser.add_argument("--constraint_weights", default="0.02")
    parser.add_argument("--diversity_weights", default="0.02")
    parser.add_argument("--prior_preference_weights", default="0.15")
    parser.add_argument("--seeds", default="0")
    return parser.parse_args()


def build_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    axes = {
        "preference_weight": parse_csv_floats(args.preference_weights),
        "defense_soft_objective_scale": parse_csv_floats(args.defense_soft_objective_scales),
        "defense_soft_utility_weight": parse_csv_floats(args.defense_soft_utility_weights),
        "prefix_hidden_align_weight": parse_csv_floats(args.prefix_hidden_align_weights),
        "constraint_weight": parse_csv_floats(args.constraint_weights),
        "diversity_weight": parse_csv_floats(args.diversity_weights),
        "prior_preference_weight": parse_csv_floats(args.prior_preference_weights),
        "seed": parse_csv_ints(args.seeds),
    }
    keys = list(axes)
    trials = [dict(zip(keys, values)) for values in itertools.product(*(axes[key] for key in keys))]
    if args.search == "random":
        rng = random.Random(int(args.random_seed))
        rng.shuffle(trials)
    if int(args.max_trials) > 0:
        trials = trials[: int(args.max_trials)]
    return trials


def trial_run_name(sweep_name: str, index: int, trial: dict[str, Any]) -> str:
    return (
        f"{sweep_name}_t{index:03d}"
        f"_pw{safe_run_fragment(float(trial['preference_weight']))}"
        f"_so{safe_run_fragment(float(trial['defense_soft_objective_scale']))}"
        f"_su{safe_run_fragment(float(trial['defense_soft_utility_weight']))}"
        f"_al{safe_run_fragment(float(trial['prefix_hidden_align_weight']))}"
        f"_pp{safe_run_fragment(float(trial['prior_preference_weight']))}"
        f"_s{int(trial['seed'])}"
    )


def completed_metrics_path(run_dir: Path) -> Path:
    return run_dir / "stage3_guided_refinement" / "stage3_metrics.json"


def is_nonempty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.iterdir())


def resolve_trial_target(args: argparse.Namespace, output_dir: Path, base_run_name: str) -> tuple[str, Path, bool]:
    base_run_dir = output_dir / base_run_name
    if bool(args.skip_existing) and completed_metrics_path(base_run_dir).is_file():
        return base_run_name, base_run_dir, True
    if is_nonempty_dir(base_run_dir) and bool(args.retry_incomplete):
        for attempt in range(1, 1000):
            retry_run_name = f"{base_run_name}_retry{attempt:03d}"
            retry_run_dir = output_dir / retry_run_name
            if bool(args.skip_existing) and completed_metrics_path(retry_run_dir).is_file():
                return retry_run_name, retry_run_dir, True
            if not is_nonempty_dir(retry_run_dir):
                return retry_run_name, retry_run_dir, False
        raise RuntimeError(f"Could not find a free retry directory for {base_run_name}")
    return base_run_name, base_run_dir, False


def build_command(args: argparse.Namespace, run_name: str, trial: dict[str, Any]) -> list[str]:
    return [
        str(args.python),
        str(PROJECT_ROOT / "scripts" / "train_defense.py"),
        "--run_name",
        run_name,
        "--data_root",
        str(args.data_root),
        "--output_dir",
        str(args.output_dir),
        "--device",
        str(args.device),
        "--seed",
        str(int(trial["seed"])),
        "--max_classes",
        str(int(args.max_classes)),
        "--max_samples",
        str(int(args.max_samples)),
        "--max_generation_traces",
        str(int(args.max_generation_traces)),
        "--encoder_train_samples",
        str(int(args.encoder_train_samples)),
        "--surrogate_train_samples",
        str(int(args.surrogate_train_samples)),
        "--surrogate_val_samples",
        str(int(args.surrogate_val_samples)),
        "--surrogate_epochs",
        str(int(args.surrogate_epochs)),
        "--surrogate_patience",
        str(int(args.surrogate_patience)),
        "--surrogate_min_val_accuracy",
        str(float(args.surrogate_min_val_accuracy)),
        "--candidate_epochs",
        str(int(args.candidate_epochs)),
        "--encoder_epochs",
        str(int(args.encoder_epochs)),
        "--diffusion_train_steps",
        str(int(args.diffusion_train_steps)),
        "--guidance_train_steps",
        str(int(args.guidance_train_steps)),
        "--batch_size",
        str(int(args.batch_size)),
        "--pareto_samples",
        str(int(args.pareto_samples)),
        "--stage3_fixed_probe_samples",
        str(int(args.stage3_fixed_probe_samples)),
        "--stage3_fixed_probe_train_samples",
        str(int(args.stage3_fixed_probe_train_samples)),
        "--stage3_fixed_probe_val_samples",
        str(int(args.stage3_fixed_probe_val_samples)),
        "--stage3_fixed_probe_epochs",
        str(int(args.stage3_fixed_probe_epochs)),
        "--preference_weight",
        str(float(trial["preference_weight"])),
        "--defense_soft_objective_scale",
        str(float(trial["defense_soft_objective_scale"])),
        "--defense_soft_utility_weight",
        str(float(trial["defense_soft_utility_weight"])),
        "--prefix_hidden_align_weight",
        str(float(trial["prefix_hidden_align_weight"])),
        "--constraint_weight",
        str(float(trial["constraint_weight"])),
        "--diversity_weight",
        str(float(trial["diversity_weight"])),
        "--prior_preference_weight",
        str(float(trial["prior_preference_weight"])),
        "--no-stage3_require_quality_gate",
        "--progress" if bool(args.progress) else "--no-progress",
    ]


def summarize_trial(run_dir: Path, index: int, run_name: str, trial: dict[str, Any], return_code: int) -> dict[str, Any]:
    stage2 = load_json(run_dir / "stage2_user_diffusion" / "stage2_metrics.json")
    stage3 = load_json(run_dir / "stage3_guided_refinement" / "stage3_metrics.json")
    selected = stage3.get("selected", {}) if isinstance(stage3, dict) else {}
    row = {
        "trial": int(index),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "return_code": int(return_code),
        **trial,
        "stage2_loss": get_nested(stage2, "diffusion", "loss"),
        "stage2_denoise": get_nested(stage2, "diffusion", "denoise"),
        "stage2_defense": get_nested(stage2, "diffusion", "defense"),
        "stage2_preference": get_nested(stage2, "diffusion", "preference_gated"),
        "stage2_alignment": get_nested(stage2, "diffusion", "alignment"),
        "selection_policy_valid": selected.get("selection_policy_valid"),
        "selection_attack_pressure": selected.get("selection_attack_pressure"),
        "selection_attack_accuracy": selected.get("selection_attack_accuracy"),
        "selection_rendered_rf_accuracy": selected.get("selection_rendered_rf_accuracy"),
        "fixed_probe_reliable_worst_accuracy": selected.get("fixed_probe_reliable_worst_accuracy"),
        "visible_dummy_overhead": selected.get("visible_dummy_overhead"),
        "raw_bandwidth_overhead": selected.get("raw_bandwidth_overhead"),
        "dummy_incoming_share": selected.get("dummy_incoming_share"),
        "template_entropy": selected.get("template_entropy"),
    }
    return row


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    def value(name: str, default: float) -> float:
        item = row.get(name)
        try:
            if item is None:
                return default
            return float(item)
        except (TypeError, ValueError):
            return default

    return (
        value("selection_attack_pressure", 1e9),
        value("selection_rendered_rf_accuracy", 1e9),
        value("selection_attack_accuracy", 1e9),
        value("visible_dummy_overhead", 1e9),
    )


def main() -> None:
    args = parse_args()
    sweep_name = args.sweep_name or f"loss_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    sweep_dir = output_dir / "hparam_sweeps" / sweep_name
    trials = build_trials(args)
    write_json(sweep_dir / "sweep_config.json", {"args": vars(args), "trials": trials})
    rows: list[dict[str, Any]] = []
    print(f"[DMMPv3 loss sweep] {len(trials)} trial(s); output={sweep_dir}", flush=True)
    for index, trial in enumerate(trials, start=1):
        base_run_name = trial_run_name(sweep_name, index, trial)
        run_name, run_dir, reuse_completed = resolve_trial_target(args, output_dir, base_run_name)
        command = build_command(args, run_name, trial)
        print(f"[DMMPv3 loss sweep] trial {index}/{len(trials)}: {run_name}", flush=True)
        if run_name != base_run_name:
            print(
                f"[DMMPv3 loss sweep] previous incomplete directory exists; retrying as {run_name}",
                flush=True,
            )
        print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in command), flush=True)
        if reuse_completed:
            print(f"[DMMPv3 loss sweep] reuse completed trial: {run_name}", flush=True)
            return_code = 0
        elif args.dry_run:
            return_code = 0
        else:
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
            return_code = int(completed.returncode)
        row = summarize_trial(run_dir, index, run_name, trial, return_code)
        if run_name != base_run_name:
            row["base_run_name"] = base_run_name
        rows.append(row)
        write_csv(sweep_dir / "results_unsorted.csv", rows)
        write_json(sweep_dir / "results_unsorted.json", {"rows": rows})
        if return_code != 0:
            print(f"[DMMPv3 loss sweep] trial failed with code {return_code}: {run_name}", flush=True)
    ranked = sorted(rows, key=sort_key)
    write_csv(sweep_dir / "results_ranked.csv", ranked)
    write_json(sweep_dir / "results_ranked.json", {"rows": ranked, "best": ranked[0] if ranked else {}})
    if ranked:
        best = ranked[0]
        print(
            "[DMMPv3 loss sweep best] "
            f"run={best['run_name']}, pressure={best.get('selection_attack_pressure')}, "
            f"rendered_rf={best.get('selection_rendered_rf_accuracy')}, "
            f"diagnostic_acc={best.get('selection_attack_accuracy')}, "
            f"visible_bw={best.get('visible_dummy_overhead')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
