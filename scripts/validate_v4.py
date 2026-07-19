"""Validate the ten Phase-1 correctness requirements for a DMMPv3 V4 smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.utils.config import DefenseConfig
from dmmp.data import load_cw_data
from dmmp.projection.padding import load_ragged_npz
from dmmp.encoders.prefix import nonzero_trace
from dmmp.constraints.user_profiles import load_profiles, select_visit
from dmmp.utils import resolve_device, write_json
from dmmp.diffusion.profile_pipeline import _candidate_context, _encoder_input, _load_v4_models


def _config(run_dir: Path) -> DefenseConfig:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    values = {key: payload[key] for key in DefenseConfig.__dataclass_fields__ if key in payload}
    return DefenseConfig(**values)


def validate(run_dir: Path) -> dict:
    cfg = _config(run_dir)
    device = resolve_device(str(cfg.device))
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    exact = np.load(run_dir / "stage1_executable_condition" / "exact_probe_subset.npz")
    predicted = np.asarray(exact["predicted_utility"], dtype=np.float32)
    allowed = np.asarray(exact["allowed_mask"], dtype=np.float32)
    candidate = np.asarray(exact["candidate_mask"], dtype=np.float32)
    checks: dict[str, dict] = {}
    outside_max = float(max(np.max(np.abs(predicted * (1.0 - allowed))), np.max(np.abs(candidate * (1.0 - allowed)))))
    checks["executable_only_in_allowed_region"] = {"passed": outside_max <= 1e-7, "outside_max": outside_max}

    raw, labels, trace_ids, _, _ = load_cw_data(cfg)
    (
        scorer,
        selected_views,
        scorer_mean,
        scorer_scale,
        bundle,
        encoder,
        encoder_payload,
        _,
        _,
        _,
    ) = _load_v4_models(run_dir, cfg, device)
    sample_raw = raw[:2]
    sample_y = labels[:2]
    context = _candidate_context(
        sample_raw,
        cfg,
        scorer,
        selected_views,
        scorer_mean,
        scorer_scale,
        bundle,
        sample_y,
        device,
        include_exact_utility=False,
    )
    ecfg = encoder_payload["config"]
    raw_features = _encoder_input(context, np.asarray(ecfg["view_vector"], dtype=np.float32))
    features = np.clip(
        (raw_features - np.asarray(ecfg["feature_mean"], dtype=np.float32)) / np.asarray(ecfg["feature_scale"], dtype=np.float32),
        -8.0,
        8.0,
    )
    with torch.no_grad():
        first = encoder(torch.as_tensor(features, dtype=torch.float32, device=device))
        second = encoder(torch.as_tensor(features, dtype=torch.float32, device=device))
    global_norm = float(first["c_global"].norm().cpu())
    reproducible_encoder = bool(torch.allclose(first["c_global"], second["c_global"]) and torch.allclose(first["c_leakage"], second["c_leakage"]))
    global_loss = float(encoder_payload["metrics"].get("global", np.inf))
    checks["encoder_condition_consistent"] = {
        "passed": reproducible_encoder and global_norm > 1e-6 and np.isfinite(global_loss),
        "c_global_norm": global_norm,
        "global_supervision_loss": global_loss,
    }

    all_profile_masks = [tuple(profile.profile_mask_20d) for rows in profiles.values() for profile in rows]
    active_profile_cells = [float(np.asarray(mask, dtype=np.float32).sum()) for mask in all_profile_masks]
    checks["user_pair_profiles_fixed_and_diverse"] = {
        "passed": all(abs(value - 1.0) <= 1e-6 for value in active_profile_cells) and len(set(all_profile_masks)) > 1,
        "profiles": len(all_profile_masks),
        "unique_pair_masks": len(set(all_profile_masks)),
        "active_cells_per_profile_min": min(active_profile_cells) if active_profile_cells else 0.0,
        "active_cells_per_profile_max": max(active_profile_cells) if active_profile_cells else 0.0,
    }

    reference_path = run_dir / "stage2_user_diffusion" / "reference_target_user_test.npz"
    traces, origins, metadata = load_ragged_npz(reference_path)
    unique_combinations = int(len(np.unique(np.asarray(metadata["combination_index"], dtype=np.int64))))
    unique_weights = int(len(np.unique(np.round(np.asarray(metadata["primitive_weights"], dtype=np.float32), 5), axis=0)))
    checks["same_user_visits_fixed"] = {
        "passed": unique_combinations == 1 and unique_weights == 1,
        "unique_combinations": unique_combinations,
        "unique_weights": unique_weights,
    }

    train_ids = {profile.profile_id for profile in profiles["train"]}
    test_ids = {profile.profile_id for profile in profiles["test"]}
    checks["target_profiles_unseen"] = {"passed": train_ids.isdisjoint(test_ids), "train": sorted(train_ids), "target": sorted(test_ids)}

    diffusion_payload = torch.load(run_dir / "stage2_user_diffusion" / "diffusion_guided_checkpoint.pt", map_location="cpu", weights_only=False)
    policy_statistics = np.load(reference_path.with_name(reference_path.stem + "_policy_statistics.npz"))
    policy_std = float(np.std(np.asarray(policy_statistics["policy_logits"], dtype=np.float32)))
    checks["diffusion_is_final_policy_generator"] = {
        "passed": str(cfg.policy_generator) == "diffusion" and "diffusion_state" in diffusion_payload and policy_std > 1e-6,
        "policy_generator": str(cfg.policy_generator),
        "policy_logit_std": policy_std,
    }

    retained_real = sum(int(origin.sum()) for origin in origins)
    expected_real = sum(int(nonzero_trace(raw[int(index)]).size) for index in np.asarray(metadata["clean_index"], dtype=np.int64))
    raw_retention = float(retained_real / max(expected_real, 1))
    reference_metrics = json.loads(reference_path.with_name(reference_path.stem + "_metrics.json").read_text(encoding="utf-8"))
    checks["raw_renderer_retains_all_real_packets"] = {
        "passed": abs(float(reference_metrics["raw_real_packet_retention"]) - 1.0) <= 1e-8 and abs(raw_retention - 1.0) <= 1e-8,
        "reported_retention": float(reference_metrics["raw_real_packet_retention"]),
    }

    pareto = json.loads((run_dir / "stage3_guided_refinement" / "stage3_metrics.json").read_text(encoding="utf-8"))["pareto_rows"]
    rows = sorted(pareto, key=lambda row: float(row["keep_ratio"]), reverse=True)
    bandwidth_by_keep = {float(row["keep_ratio"]): float(row["raw_bandwidth_overhead"]) for row in rows}
    keep_one = bandwidth_by_keep.get(1.0, 0.0)
    reduced = [value for keep, value in bandwidth_by_keep.items() if keep < 1.0]
    checks["stage3_reduces_dummy"] = {
        "passed": bool(reduced) and all(value < keep_one for value in reduced),
        "raw_bandwidth_by_keep": bandwidth_by_keep,
    }

    same = json.loads((run_dir / "attack_eval" / "same_user" / "df" / "same_user_df_metrics.json").read_text(encoding="utf-8"))
    cross = json.loads((run_dir / "attack_eval" / "cross_user" / "df" / "cross_user_df_metrics.json").read_text(encoding="utf-8"))
    same_sources = [row["profile_id"] for row in same["source_profiles"]]
    cross_sources = [row["profile_id"] for row in cross["source_profiles"]]
    target_id = same["target_profile"]["profile_id"]
    checks["adaptive_protocol_sources_correct"] = {
        "passed": same_sources == [target_id] and all(source != target_id for source in cross_sources),
        "same_user_sources": same_sources,
        "cross_user_sources": cross_sources,
        "target": target_id,
    }

    profile = profiles["train"][0]
    visit_a = select_visit(profile, "validation-visit", "validation-trace", str(cfg.visit_selector))
    visit_b = select_visit(profile, "validation-visit", "validation-trace", str(cfg.visit_selector))
    checks["keyed_visit_reproducible"] = {
        "passed": visit_a.combination == visit_b.combination
        and np.array_equal(visit_a.primitive_weights, visit_b.primitive_weights)
        and visit_a.diffusion_seed == visit_b.diffusion_seed
        and visit_a.renderer_seed == visit_b.renderer_seed,
        "combination": list(visit_a.combination),
    }

    for value in checks.values():
        value["passed"] = bool(value["passed"])
    passed = all(value["passed"] for value in checks.values())
    return {"passed": passed, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    result = validate(run_dir)
    write_json(run_dir / "phase1_acceptance.json", result)
    lines = ["# DMMPv3 V4 Phase 1 correctness acceptance", "", f"- overall: {'PASS' if result['passed'] else 'FAIL'}", ""]
    for name, row in result["checks"].items():
        lines.append(f"- [{'x' if row['passed'] else ' '}] {name}")
    (run_dir / "phase1_acceptance.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

