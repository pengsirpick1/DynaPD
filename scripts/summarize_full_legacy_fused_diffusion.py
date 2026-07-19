from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _attack_summary(run_dir: Path, protocol: str) -> dict[str, Any]:
    return _read_json(run_dir / "attack_eval" / protocol / "attack_summary.json")


def _stage3_selected(run_dir: Path) -> dict[str, Any]:
    selected = _read_json(run_dir / "stage3_guided_refinement" / "selected_policy.json")
    metrics = _read_json(run_dir / "stage3_guided_refinement" / "stage3_metrics.json")
    return {"selected_policy": selected, "stage3_metrics": metrics}


def _row_value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def build_summary(run_dir: Path, protocols: list[str]) -> dict[str, Any]:
    run_cfg = _read_json(run_dir / "run_config.json")
    split_payload = _read_json(run_dir / "split_indices.json")
    split_sizes = {
        key: len(value) if isinstance(value, list) else 0
        for key, value in split_payload.items()
    }
    stage2 = _read_json(run_dir / "stage2_user_diffusion" / "stage2_metrics.json")
    profile_stats = _read_json(run_dir / "stage2_user_diffusion" / "profile_statistics.json")
    stage3 = _stage3_selected(run_dir)
    attacks = {protocol: _attack_summary(run_dir, protocol) for protocol in protocols}
    return {
        "run_dir": str(run_dir.resolve()),
        "method": "full_dmmpv3_legacy_direct_v1_modes_fused_multi_view_renderer",
        "config": {
            "version": run_cfg.get("version"),
            "budget": run_cfg.get("budgets"),
            "render_coordinate": run_cfg.get("render_coordinate"),
            "multi_view_mode": run_cfg.get("multi_view_mode"),
            "multi_view_df_share": run_cfg.get("multi_view_df_share"),
            "multi_view_awf_share": run_cfg.get("multi_view_awf_share"),
            "multi_view_rf_share": run_cfg.get("multi_view_rf_share"),
            "v1_mode_pool": run_cfg.get("v1_mode_pool"),
            "v1_mode_prior_weight": run_cfg.get("v1_mode_prior_weight"),
            "profile_combination_mode": run_cfg.get("profile_combination_mode"),
            "active_pair_count": run_cfg.get("active_pair_count"),
            "active_triple_count": run_cfg.get("active_triple_count"),
            "deployment_repeats": run_cfg.get("deployment_repeats"),
        },
        "split_sizes": split_sizes,
        "stage2": stage2,
        "profile_statistics": profile_stats,
        "stage3": stage3,
        "attacks": attacks,
    }


def write_markdown(path: Path, summary: dict[str, Any], protocols: list[str]) -> None:
    cfg = summary["config"]
    selected = summary["stage3"].get("selected_policy", {})
    lines = [
        "# Full DMMPv3 Legacy-Fused Diffusion Experiment",
        "",
        f"- run_dir: {summary['run_dir']}",
        f"- split sizes: {summary['split_sizes']}",
        f"- renderer: {cfg.get('render_coordinate')} / {cfg.get('multi_view_mode')}",
        f"- multi-view DF/AWF/RF: {cfg.get('multi_view_df_share')}/{cfg.get('multi_view_awf_share')}/{cfg.get('multi_view_rf_share')}",
        f"- V1 mode pool: {cfg.get('v1_mode_pool')} with weight {cfg.get('v1_mode_prior_weight')}",
        f"- profile mode: {cfg.get('profile_combination_mode')} pair/triple={cfg.get('active_pair_count')}/{cfg.get('active_triple_count')}",
        f"- deployment repeats: {cfg.get('deployment_repeats')}",
        "",
        "## Stage 3 Selection",
        "",
        f"- selected budget / keep: {selected.get('budget', cfg.get('budget'))} / {selected.get('keep_ratio', 1.0)}",
        f"- selected surrogate worst acc: {selected.get('surrogate_defended_accuracy', 'NA')}",
        f"- selected visible overhead: {selected.get('visible_dummy_overhead', 'NA')}",
        f"- selected raw retention: {selected.get('raw_real_packet_retention', 'NA')}",
        "",
        "## Full Attack Evaluation",
        "",
        "| protocol | attacker | source users | clean acc | defended acc | drop pp | bandwidth | retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in protocols:
        attack = summary["attacks"].get(protocol, {})
        for row in attack.get("rows", []) or []:
            clean = _row_value(row, "clean_acc")
            defended = _row_value(row, "fresh_defended_acc")
            lines.append(
                f"| {protocol} | {row.get('attacker', '')} | {int(row.get('source_user_count', 0))} | "
                f"{clean:.6f} | {defended:.6f} | {100.0 * (clean - defended):.2f} | "
                f"{_row_value(row, 'visible_bandwidth'):.6f} | {_row_value(row, 'raw_retention'):.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a full legacy-fused DMMPv3 diffusion experiment.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--protocols", default="fixed,same_user,full_catalogue")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    protocols = [item.strip() for item in str(args.protocols).split(",") if item.strip()]
    summary = build_summary(run_dir, protocols)
    json_path = run_dir / "full_legacy_fused_experiment_summary.json"
    md_path = run_dir / "full_legacy_fused_experiment_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, summary, protocols)
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
