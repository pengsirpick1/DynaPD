"""User-specific adaptive attack protocols for DMMPv3 V4."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..evaluation.attack_models import build_df_input, build_rf_tam_input
from ..evaluation.attacks import _eval_torch, _load_checkpoint_state, train_df_model
from ..utils.config import AttackConfig, DefenseConfig
from ..data import choose_stratified_subset, load_cw_data
from ..projection.padding import crop_ragged_for_attacker, load_ragged_npz
from ..constraints.user_profiles import UserDefenseProfile, load_profiles, profile_overlap
from ..utils import log, resolve_device, set_seed, write_csv, write_json
from ..diffusion.profile_pipeline import generate_v4_ragged_dataset


def _defense_config_from_run(run_dir: Path, attack_cfg: AttackConfig) -> DefenseConfig:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    values = {key: payload[key] for key in DefenseConfig.__dataclass_fields__ if key in payload}
    if attack_cfg.data_root:
        values["data_root"] = str(attack_cfg.data_root)
    return DefenseConfig(**values)


def _load_splits(run_dir: Path) -> dict[str, np.ndarray]:
    payload = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def _subsample(indices: np.ndarray, labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if int(maximum) <= 0 or len(indices) <= int(maximum):
        return np.asarray(indices, dtype=np.int64)
    local = choose_stratified_subset(labels[indices], int(maximum), int(seed))
    return np.asarray(indices, dtype=np.int64)[local]


def _find_profile(profiles: dict[str, list[UserDefenseProfile]], profile_id: str, default_split: str, default_index: int = 0) -> UserDefenseProfile:
    if profile_id:
        for rows in profiles.values():
            for profile in rows:
                if profile.profile_id == profile_id:
                    return profile
        raise ValueError(f"Unknown V4 profile: {profile_id}")
    return profiles[default_split][int(default_index)]


def _full_catalogue_sources(profiles: Sequence[UserDefenseProfile]) -> list[UserDefenseProfile]:
    chosen: list[UserDefenseProfile] = []
    covered: set[tuple[str, ...]] = set()
    for profile in profiles:
        gain = set(profile.active_combinations) - covered
        if gain:
            chosen.append(profile)
            covered.update(profile.active_combinations)
        if len(covered) >= 20:
            break
    return chosen


def _source_profiles(attack_cfg: AttackConfig, profiles: dict[str, list[UserDefenseProfile]], protocol: str, target: UserDefenseProfile) -> list[UserDefenseProfile]:
    if protocol in {"same_user", "profile_known"}:
        return [target]
    if protocol == "full_catalogue":
        return _full_catalogue_sources(profiles["train"])
    if attack_cfg.source_profile_values:
        return [_find_profile(profiles, value, "train") for value in attack_cfg.source_profile_values]
    count = max(1, int(attack_cfg.source_user_count)) if protocol == "multi_source" else 1
    return profiles["train"][:count]


def _balanced_indices(
    base_indices: np.ndarray,
    labels: np.ndarray,
    source_count: int,
    attack_cfg: AttackConfig,
    seed: int,
) -> list[np.ndarray]:
    if int(attack_cfg.fixed_per_user_adaptive_samples) > 0:
        per_user = int(attack_cfg.fixed_per_user_adaptive_samples)
    else:
        total = int(attack_cfg.fixed_total_adaptive_samples) if int(attack_cfg.fixed_total_adaptive_samples) > 0 else len(base_indices)
        total = min(int(total), len(base_indices))
        per_user = int(np.ceil(total / max(int(source_count), 1)))
    return [_subsample(base_indices, labels, per_user, int(seed) + index) for index in range(int(source_count))]


def _cache_key(indices: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:10]


def _file_sha1(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_signature(path: Path) -> list[dict[str, str]]:
    if not path.is_dir():
        return []
    rows = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            rows.append({"path": item.relative_to(path).as_posix(), "sha1": _file_sha1(item) or ""})
    return rows


def _defense_artifact_signature(run_dir: Path) -> dict:
    tracked_files = [
        "run_config.json",
        "split_indices.json",
        "stage1_executable_condition/strong_surrogate_ensemble.pt",
        "stage1_executable_condition/candidate_scorer_checkpoint.pt",
        "stage1_executable_condition/candidate_metrics.json",
        "stage2_user_diffusion/encoder_checkpoint.pt",
        "stage2_user_diffusion/condition_encoder_checkpoint.pt",
        "stage2_user_diffusion/policy_diffusion_checkpoint.pt",
        "stage3_guided_refinement/selected_policy.json",
    ]
    payload = {
        "files": {name: _file_sha1(run_dir / name) for name in tracked_files},
        "profiles": _tree_signature(run_dir / "stage2_user_diffusion" / "user_profiles"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {"digest": hashlib.sha1(encoded).hexdigest()[:16], **payload}


def _attack_cache_signature(attack_cfg: AttackConfig, protocol: str, kind: str, setting: str, defense_signature: dict) -> dict:
    ignored = {"progress", "log_every", "force_retrain"}
    return {
        "protocol": str(protocol),
        "kind": str(kind),
        "setting": str(setting),
        "config": {key: value for key, value in vars(attack_cfg).items() if key not in ignored},
        "defense_artifact_signature": defense_signature,
    }


def _transfer_output_name(protocol: str, source_run_dir: Path, source_label: str) -> str:
    label = str(source_label).strip() or source_run_dir.name
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    return f"{protocol}_transfer_from_{safe_label}"


def _get_profile_dataset(
    raw: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    indices: np.ndarray,
    run_dir: Path,
    cfg: DefenseConfig,
    profile: UserDefenseProfile,
    role: str,
    budget: float,
    keep_ratio: float,
    defense_signature: dict,
    device: torch.device,
):
    cache_dir = run_dir / "defended_datasets" / "profiles" / profile.profile_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature_digest = str(defense_signature.get("digest", "unknown"))
    path = cache_dir / f"{role}_b{budget:.2f}_k{keep_ratio:.2f}_n{len(indices)}_{_cache_key(indices)}_d{signature_digest}.npz"
    metrics_path = path.with_name(path.stem + "_metrics.json")
    needs_generate = not path.is_file() or not metrics_path.is_file()
    if not needs_generate:
        cached_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        needs_generate = cached_metrics.get("defense_artifact_signature_digest") != signature_digest
    if needs_generate:
        log(
            f"[attack dataset] generating defended traces: role={role}, profile={profile.profile_id}, "
            f"traces={len(indices)}, budget={float(budget):.3f}, keep={float(keep_ratio):.2f}",
            cfg.progress,
        )
        generate_v4_ragged_dataset(
            raw,
            labels,
            trace_ids,
            indices,
            run_dir,
            cfg,
            profile=profile,
            visit_namespace=f"{role}:{profile.profile_id}",
            budget=float(budget),
            keep_ratio=float(keep_ratio),
            output_npz=path,
            device=device,
        )
    else:
        log(
            f"[attack dataset] using cached defended traces: role={role}, profile={profile.profile_id}, path={path}",
            cfg.progress,
        )
    traces, origins, metadata = load_ragged_npz(path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("defense_artifact_signature_digest") != signature_digest:
        metrics["defense_artifact_signature_digest"] = signature_digest
        metrics["defense_artifact_signature"] = defense_signature
        write_json(metrics_path, metrics)
    return traces, origins, np.asarray(metadata["y"], dtype=np.int64), metrics, path


def _ragged_rf_tam(traces: Sequence[np.ndarray], slots: int, max_load_time: float) -> np.ndarray:
    tam = np.zeros((len(traces), 2, int(slots)), dtype=np.float32)
    scale = float(int(slots) - 1) / float(max_load_time)
    for row_index, trace in enumerate(traces):
        nonzero = np.asarray(trace, dtype=np.float32)
        outgoing = nonzero[nonzero > 0]
        incoming = -nonzero[nonzero < 0]
        if outgoing.size:
            bins = np.floor(outgoing * scale).astype(np.int64)
            bins[outgoing >= float(max_load_time)] = int(slots) - 1
            np.add.at(tam[row_index, 0], np.clip(bins, 0, int(slots) - 1), 1.0)
        if incoming.size:
            bins = np.floor(incoming * scale).astype(np.int64)
            bins[incoming >= float(max_load_time)] = int(slots) - 1
            np.add.at(tam[row_index, 1], np.clip(bins, 0, int(slots) - 1), 1.0)
    return tam


def _clean_input(kind: str, raw: np.ndarray, cfg: DefenseConfig, attack_cfg: AttackConfig) -> np.ndarray:
    if kind == "df":
        return build_df_input(raw, int(cfg.max_trace_length))
    return build_rf_tam_input(
        raw,
        max_len=int(cfg.max_trace_length),
        max_load_time=float(attack_cfg.max_load_time),
        num_slots=int(attack_cfg.rf_tam_num_slots),
    )


def _clean_input_indexed(
    kind: str,
    raw: np.ndarray,
    indices: np.ndarray,
    cfg: DefenseConfig,
    attack_cfg: AttackConfig,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    """Build attack inputs without materializing a full float64 memmap slice."""

    selected = np.asarray(indices, dtype=np.int64)
    if kind == "df":
        result = np.empty((len(selected), 1, int(cfg.max_trace_length)), dtype=np.float32)
    else:
        result = np.empty((len(selected), 2, int(attack_cfg.rf_tam_num_slots)), dtype=np.float32)
    for start in range(0, len(selected), int(chunk_size)):
        end = min(start + int(chunk_size), len(selected))
        batch_raw = np.asarray(raw[selected[start:end]])
        result[start:end] = _clean_input(kind, batch_raw, cfg, attack_cfg)
        if start == 0 or end == len(selected) or end % 5000 < int(chunk_size):
            log(f"{kind.upper()} clean input preparation: {end}/{len(selected)}", attack_cfg.progress)
    return result


def _defended_input(kind: str, traces: Sequence[np.ndarray], origins: Sequence[np.ndarray], cfg: DefenseConfig, attack_cfg: AttackConfig):
    if kind == "df":
        padded, stats = crop_ragged_for_attacker(traces, origins, int(cfg.max_trace_length))
        return build_df_input(padded, int(cfg.max_trace_length)), {"df_input_retention": stats["attacker_input_real_packet_retention"], "clip_rate": stats["clip_rate"]}
    return _ragged_rf_tam(traces, int(attack_cfg.rf_tam_num_slots), float(attack_cfg.max_load_time)), {"rf_input_retention": 1.0, "clip_rate": 0.0}


def _selected_budget_and_keep(run_dir: Path, cfg: DefenseConfig) -> tuple[float, float]:
    path = run_dir / "stage3_guided_refinement" / "selected_policy.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload.get("budget", cfg.budget_values[-1])), float(payload.get("keep_ratio", 1.0))
    return float(cfg.budget_values[-1]), 1.0


def run_v4_attack_evaluation(attack_cfg: AttackConfig) -> Path:
    started_at = time.perf_counter()
    set_seed(int(attack_cfg.seed))
    run_dir = Path(attack_cfg.run_dir)
    cfg = _defense_config_from_run(run_dir, attack_cfg)
    version_tag = str(cfg.version).lower()
    version_label = "DMMPv3" if version_tag == "v3" else str(cfg.version).upper()
    version_title = "DMMPv3" if version_tag == "v3" else f"DMMPv3 {version_label}"
    device = resolve_device(str(attack_cfg.device))
    log(
        f"========== Fixed/Mixed Attack Evaluation START | run={run_dir}, protocol={attack_cfg.adaptive_protocol}, "
        f"attackers={attack_cfg.attackers}, device={device} ==========",
        attack_cfg.progress,
    )
    log(f"[attack eval] loading CW data and split indices...", attack_cfg.progress)
    raw, labels, trace_ids, _, data_source = load_cw_data(cfg)
    splits = _load_splits(run_dir)
    train_idx = _subsample(splits["train"], labels, int(attack_cfg.max_train_traces), int(attack_cfg.seed) + 11)
    val_idx = _subsample(splits["val"], labels, int(attack_cfg.max_val_traces), int(attack_cfg.seed) + 12)
    test_idx = _subsample(splits["test"], labels, int(attack_cfg.max_test_traces), int(attack_cfg.seed) + 13)
    log(
        f"[attack eval] data ready: train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, "
        f"classes={len(np.unique(labels))}",
        attack_cfg.progress,
    )
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    target = _find_profile(profiles, str(attack_cfg.target_profile_id), "test")
    protocol = str(attack_cfg.adaptive_protocol).lower()
    budget, keep_ratio = _selected_budget_and_keep(run_dir, cfg)
    defense_signature = _defense_artifact_signature(run_dir)
    adaptive_source_run_dir = Path(str(attack_cfg.adaptive_source_run_dir)).resolve() if str(attack_cfg.adaptive_source_run_dir).strip() else run_dir
    uses_transfer_source = protocol != "fixed" and adaptive_source_run_dir.resolve() != run_dir.resolve()
    if uses_transfer_source:
        source_cfg = _defense_config_from_run(adaptive_source_run_dir, attack_cfg)
        source_profiles_all = load_profiles(adaptive_source_run_dir / "stage2_user_diffusion" / "user_profiles")
        source_target = _find_profile(source_profiles_all, target.profile_id, "test")
        source_budget, source_keep_ratio = _selected_budget_and_keep(adaptive_source_run_dir, source_cfg)
        source_defense_signature = _defense_artifact_signature(adaptive_source_run_dir)
        source_profiles = [] if protocol == "fixed" else _source_profiles(attack_cfg, source_profiles_all, protocol, source_target)
        output_leaf = _transfer_output_name(protocol, adaptive_source_run_dir, str(attack_cfg.adaptive_source_label))
    else:
        source_cfg = cfg
        source_profiles_all = profiles
        source_target = target
        source_budget, source_keep_ratio = budget, keep_ratio
        source_defense_signature = defense_signature
        source_profiles = [] if protocol == "fixed" else _source_profiles(attack_cfg, profiles, protocol, target)
        output_leaf = protocol
    sources = source_profiles
    cache_defense_signature = {
        "target_run": str(run_dir),
        "target_defense_artifact_signature": defense_signature,
        "adaptive_source_run": str(adaptive_source_run_dir),
        "adaptive_source_defense_artifact_signature": source_defense_signature,
    }
    output_dir = Path(attack_cfg.output_dir) if attack_cfg.output_dir else run_dir / "attack_eval" / output_leaf
    output_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"[attack eval] selected defense policy: budget={budget:.4f}, keep_ratio={keep_ratio:.4f}, "
        f"target_profile={target.profile_id}, output={output_dir}",
        attack_cfg.progress,
    )
    if uses_transfer_source:
        log(
            f"[attack eval] transfer mixed source: adaptive_train/val run={adaptive_source_run_dir}, "
            f"source_profile={source_target.profile_id}, source_budget={source_budget:.4f}, "
            f"source_keep={source_keep_ratio:.4f}; fresh test run={run_dir}",
            attack_cfg.progress,
        )
    target_test_traces, target_test_origins, target_test_y, target_test_metrics, target_test_path = _get_profile_dataset(
        raw,
        labels,
        trace_ids,
        test_idx,
        run_dir,
        cfg,
        target,
        "fresh_deployment_test",
        budget,
        keep_ratio,
        defense_signature,
        device,
    )
    source_train_sets, source_val_sets = [], []
    if sources:
        train_parts = _balanced_indices(train_idx, labels, len(sources), attack_cfg, int(attack_cfg.seed) + 100)
        val_parts = _balanced_indices(val_idx, labels, len(sources), attack_cfg, int(attack_cfg.seed) + 200)
        for source, source_train_idx, source_val_idx in zip(sources, train_parts, val_parts):
            source_train_sets.append(
                _get_profile_dataset(
                    raw,
                    labels,
                    trace_ids,
                    source_train_idx,
                    adaptive_source_run_dir,
                    source_cfg,
                    source,
                    "adaptive_train",
                    source_budget,
                    source_keep_ratio,
                    source_defense_signature,
                    device,
                )
            )
            source_val_sets.append(
                _get_profile_dataset(
                    raw,
                    labels,
                    trace_ids,
                    source_val_idx,
                    adaptive_source_run_dir,
                    source_cfg,
                    source,
                    "adaptive_validation",
                    source_budget,
                    source_keep_ratio,
                    source_defense_signature,
                    device,
                )
            )
    kinds = sorted({"df" if item.lower().endswith("df") else "rf" for item in attack_cfg.attacker_values})
    rows = []
    for kind in kinds:
        kind_timer = time.perf_counter()
        setting = f"{protocol}_{kind}"
        setting_dir = output_dir / kind
        setting_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = setting_dir / f"{setting}_metrics.json"
        checkpoint_path = setting_dir / f"{setting}_checkpoint.pt"
        cache_signature = _attack_cache_signature(attack_cfg, protocol, kind, setting, cache_defense_signature)
        if metrics_path.is_file() and checkpoint_path.is_file() and not bool(attack_cfg.force_retrain):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("cache_signature") != cache_signature:
                log(
                    f"[{version_label} attack cache miss] protocol={protocol}, attacker={kind.upper()} "
                    "because cached config differs; retraining.",
                    True,
                )
            else:
                cached_row = metrics["summary_row"]
                cached_row["quality_gate_passed"] = int(
                    float(cached_row["clean_acc"]) >= float(attack_cfg.attack_min_clean_accuracy)
                    and float(cached_row["fresh_defended_acc"]) <= float(attack_cfg.attack_max_defended_accuracy)
                )
                rows.append(cached_row)
                log(
                    f"[{version_label} attack cached] protocol={protocol}, attacker={kind.upper()}, "
                    f"clean_acc={float(cached_row['clean_acc']):.4f}, "
                    f"defended_acc={float(cached_row['fresh_defended_acc']):.4f}, "
                    f"drop_pp={100.0 * (float(cached_row['clean_acc']) - float(cached_row['fresh_defended_acc'])):.2f}, "
                    f"hard_gate_pass={int(cached_row['quality_gate_passed'])}",
                    True,
                )
                continue
        log(
            f"[attack eval] {kind.upper()} preparing inputs: clean_train={len(train_idx)}, clean_val={len(val_idx)}, "
            f"clean_test={len(test_idx)}, defended_test={len(target_test_y)}",
            attack_cfg.progress,
        )
        clean_train_x = _clean_input_indexed(kind, raw, train_idx, cfg, attack_cfg)
        clean_val_x = _clean_input_indexed(kind, raw, val_idx, cfg, attack_cfg)
        clean_test_x = _clean_input_indexed(kind, raw, test_idx, cfg, attack_cfg)
        defended_test_x, target_adapter_stats = _defended_input(kind, target_test_traces, target_test_origins, cfg, attack_cfg)
        if protocol == "fixed":
            train_x, train_y = clean_train_x, labels[train_idx]
            val_x, val_y = clean_val_x, labels[val_idx]
            initial_state = None
        else:
            defended_train_x, defended_train_y = [], []
            defended_val_x, defended_val_y = [], []
            for traces, origins, y_values, _, _ in source_train_sets:
                values, _ = _defended_input(kind, traces, origins, cfg, attack_cfg)
                defended_train_x.append(values)
                defended_train_y.append(y_values)
            for traces, origins, y_values, _, _ in source_val_sets:
                values, _ = _defended_input(kind, traces, origins, cfg, attack_cfg)
                defended_val_x.append(values)
                defended_val_y.append(y_values)
            train_x = np.concatenate([clean_train_x, *defended_train_x], axis=0)
            train_y = np.concatenate([labels[train_idx], *defended_train_y], axis=0)
            val_x = np.concatenate([clean_val_x, *defended_val_x], axis=0)
            val_y = np.concatenate([labels[val_idx], *defended_val_y], axis=0)
            fixed_checkpoint_root = adaptive_source_run_dir if uses_transfer_source else run_dir
            fixed_path = fixed_checkpoint_root / "attack_eval" / "fixed" / kind / f"fixed_{kind}_checkpoint.pt"
            initial_state = _load_checkpoint_state(fixed_path, device) if str(attack_cfg.adaptive_init) == "checkpoint" else None
        log(
            f"[attack eval] {kind.upper()} training starts: train={len(train_y)}, val={len(val_y)}, "
            f"epochs<={int(attack_cfg.clean_df_epochs if protocol == 'fixed' else attack_cfg.adaptive_epochs)}, "
            f"batch_size={int(attack_cfg.df_batch_size)}",
            attack_cfg.progress,
        )
        model, classes, best_val = train_df_model(
            train_x,
            train_y,
            val_x,
            val_y,
            attacker_kind=kind.upper(),
            defense_cfg=cfg,
            attack_cfg=attack_cfg,
            initial_state=initial_state,
            epochs=int(attack_cfg.clean_df_epochs if protocol == "fixed" else attack_cfg.adaptive_epochs),
            patience=int(attack_cfg.clean_df_patience if protocol == "fixed" else attack_cfg.adaptive_patience),
            lr=float(attack_cfg.clean_df_lr if protocol == "fixed" else attack_cfg.adaptive_lr),
            batch_size=int(attack_cfg.df_batch_size),
            device=device,
            seed=int(attack_cfg.seed),
            progress=bool(attack_cfg.progress),
        )
        log(f"[attack eval] {kind.upper()} evaluating clean test and fresh defended test...", attack_cfg.progress)
        clean_metrics = _eval_torch(model, clean_test_x, labels[test_idx], classes, device, int(attack_cfg.df_batch_size))
        defended_metrics = _eval_torch(model, defended_test_x, target_test_y, classes, device, int(attack_cfg.df_batch_size))
        torch.save({"model_state": model.state_dict(), "classes": classes, "best_val": best_val}, checkpoint_path)
        overlap_rows = [profile_overlap(source, target) for source in sources]
        overlap = float(np.mean([row["combined_jaccard"] for row in overlap_rows])) if overlap_rows else 0.0
        summary_row = {
            "protocol": protocol,
            "attacker": kind.upper(),
            "source_user_count": int(len(sources)),
            "source_profiles": ",".join(source.profile_id for source in sources),
            "target_profile": target.profile_id,
            "profile_overlap": overlap,
            "clean_acc": float(clean_metrics["defended_accuracy"]),
            "fresh_defended_acc": float(defended_metrics["defended_accuracy"]),
            "visible_bandwidth": float(target_test_metrics["visible_dummy_overhead"]),
            "raw_retention": float(target_test_metrics["raw_real_packet_retention"]),
        }
        summary_row["quality_gate_passed"] = int(
            summary_row["clean_acc"] >= float(attack_cfg.attack_min_clean_accuracy)
            and summary_row["fresh_defended_acc"] <= float(attack_cfg.attack_max_defended_accuracy)
        )
        log(
            f"[{version_label} attack result] protocol={protocol}, attacker={kind.upper()}, "
            f"clean_val={float(best_val):.4f}, clean_acc={summary_row['clean_acc']:.4f}, "
            f"defended_acc={summary_row['fresh_defended_acc']:.4f}, "
            f"drop_pp={100.0 * (summary_row['clean_acc'] - summary_row['fresh_defended_acc']):.2f}, "
            f"overhead={summary_row['visible_bandwidth']:.4f}, retention={summary_row['raw_retention']:.4f}, "
            f"hard_gate_pass={int(summary_row['quality_gate_passed'])}, elapsed={time.perf_counter() - kind_timer:.1f}s",
            True,
        )
        metrics = {
            "setting": setting,
            "protocol": protocol,
            "attacker": kind.upper(),
            "best_val_accuracy": float(best_val),
            "clean_test": clean_metrics,
            "fresh_target_user_defended_test": defended_metrics,
            "source_profiles": [source.to_dict(False) for source in sources],
            "target_profile": target.to_dict(False),
            "profile_overlap": overlap_rows,
            "target_dataset": str(target_test_path),
            "target_dataset_metrics": target_test_metrics,
            "target_adapter_metrics": target_adapter_stats,
            "cache_signature": cache_signature,
            "summary_row": summary_row,
        }
        write_json(metrics_path, metrics)
        rows.append(summary_row)
    write_csv(output_dir / "attack_summary.csv", rows)
    write_json(
        output_dir / "attack_summary.json",
        {
            "protocol": protocol,
            "transfer_mixed": bool(uses_transfer_source),
            "data_source": data_source,
            "target_run_dir": str(run_dir),
            "adaptive_source_run_dir": str(adaptive_source_run_dir),
            "budget": budget,
            "keep_ratio": keep_ratio,
            "adaptive_source_budget": source_budget,
            "adaptive_source_keep_ratio": source_keep_ratio,
            "defense_artifact_signature_digest": defense_signature.get("digest", ""),
            "adaptive_source_defense_artifact_signature_digest": source_defense_signature.get("digest", ""),
            "source_profiles": [source.profile_id for source in sources],
            "target_profile": target.profile_id,
            "quality_gate": {
                "min_clean_accuracy": float(attack_cfg.attack_min_clean_accuracy),
                "max_defended_accuracy": float(attack_cfg.attack_max_defended_accuracy),
                "all_passed": bool(rows) and all(int(row.get("quality_gate_passed", 0)) for row in rows),
            },
            "rows": rows,
        },
    )
    lines = [
        f"# {version_title} {protocol} attack evaluation",
        "",
        f"- transfer mixed: {int(uses_transfer_source)}",
        f"- adaptive source run: {adaptive_source_run_dir}",
        f"- fresh target run: {run_dir}",
        f"- source profiles: {', '.join(source.profile_id for source in sources) if sources else 'clean only'}",
        f"- target profile: {target.profile_id}",
        f"- budget / keep ratio: {budget:.4f} / {keep_ratio:.4f}",
        f"- adaptive source budget / keep ratio: {source_budget:.4f} / {source_keep_ratio:.4f}",
        f"- raw real-packet retention: {target_test_metrics['raw_real_packet_retention']:.6f}",
        f"- visible dummy overhead: {target_test_metrics['visible_dummy_overhead']:.6f}",
        f"- hard gate: clean >= {float(attack_cfg.attack_min_clean_accuracy):.4f}, defended <= {float(attack_cfg.attack_max_defended_accuracy):.4f}",
        "",
        "| protocol | attacker | source users | overlap | clean acc | defended acc | gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['attacker']} | {row['source_user_count']} | {row['profile_overlap']:.4f} | {row['clean_acc']:.6f} | {row['fresh_defended_acc']:.6f} | {int(row.get('quality_gate_passed', 0))} |"
        )
    (output_dir / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output_dir / "attack_eval_config.json", vars(attack_cfg))
    if rows:
        for row in rows:
            log(
                f"[defense effect] {row['attacker']}: clean_acc={float(row['clean_acc']):.4f}, "
                f"defended_acc={float(row['fresh_defended_acc']):.4f}, "
                f"drop_pp={100.0 * (float(row['clean_acc']) - float(row['fresh_defended_acc'])):.2f}, "
                f"visible_overhead={float(row['visible_bandwidth']):.4f}, retention={float(row['raw_retention']):.4f}",
                True,
            )
    log(
        f"========== Fixed/Mixed Attack Evaluation DONE in {time.perf_counter() - started_at:.1f}s | saved to: {output_dir} ==========",
        attack_cfg.progress,
    )
    failed_rows = [row for row in rows if not int(row.get("quality_gate_passed", 0))]
    if failed_rows and bool(attack_cfg.attack_require_quality_gate):
        failed = ", ".join(
            f"{row['attacker']}(clean={float(row['clean_acc']):.4f}, defended={float(row['fresh_defended_acc']):.4f})"
            for row in failed_rows
        )
        raise RuntimeError(
            f"Attack hard quality gate failed: require clean_acc >= {float(attack_cfg.attack_min_clean_accuracy):.4f} "
            f"and defended_acc <= {float(attack_cfg.attack_max_defended_accuracy):.4f}; failed: {failed}. "
            f"Metrics were saved to {output_dir}."
        )
    return output_dir

