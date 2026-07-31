"""Training and validation routines for the conditional purifier."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dmmp.utils import resolve_device, set_seed, write_json

from .config import PurifierConfig
from .dataset import (
    PairManifestDataset,
    SourceBalancedPairDataset,
    assert_disjoint_splits,
    sha256_file,
    split_source_sets,
)
from .pipeline import ConditionalTrafficPurifier, build_purifier, gradient_report


def _json_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_dir(cfg: PurifierConfig, overwrite: bool) -> Path:
    name = cfg.run_name.strip() or f"{cfg.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_root) / name
    sentinels = ["training_history.json", "checkpoint_selection.json", "checkpoints"]
    if run_dir.exists() and any((run_dir / item).exists() for item in sentinels):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite purifier run directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _check_pair_audit(cfg: PurifierConfig) -> dict[str, Any]:
    if not bool(cfg.requires_pair_audit_pass):
        return {"required": False, "verdict": "SKIPPED", "reason": "requires_pair_audit_pass=false"}
    candidates: list[Path] = []
    if str(cfg.pair_audit_report).strip():
        candidates.append(Path(cfg.pair_audit_report))
    for manifest in [cfg.train_manifest, cfg.validation_manifest, cfg.test_manifest]:
        path = Path(manifest).parent / "purifier_split_audit_report.json"
        if path not in candidates:
            candidates.append(path)
    if str(cfg.run_dir).strip():
        legacy = Path(cfg.run_dir) / "manifests" / "purifier_split_audit_report.json"
        if legacy not in candidates:
            candidates.append(legacy)
    report_path = next((path for path in candidates if path.is_file()), None)
    if report_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing purifier split audit report; searched: {searched}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if str(report.get("verdict")) != "PASS":
        raise RuntimeError(f"Purifier split audit is not PASS: {report_path}")
    return {
        "required": True,
        "verdict": str(report.get("verdict")),
        "report": str(report_path.resolve()),
        "unified_manifest": str(report.get("unified_manifest", "")),
        "counts": report.get("counts", {}),
    }


def _manifest_hashes(cfg: PurifierConfig) -> dict[str, str]:
    return {
        "train": sha256_file(cfg.train_manifest, int(cfg.manifest_hash_bytes)),
        "validation": sha256_file(cfg.validation_manifest, int(cfg.manifest_hash_bytes)),
        "test": sha256_file(cfg.test_manifest, int(cfg.manifest_hash_bytes)),
    }


def _split_assertions(cfg: PurifierConfig) -> dict[str, Any]:
    sources, fingerprints = split_source_sets(
        {
            "train": cfg.train_manifest,
            "validation": cfg.validation_manifest,
            "test": cfg.test_manifest,
        }
    )
    assert_disjoint_splits(sources, fingerprints)
    return {
        "train_sources": int(len(sources["train"])),
        "validation_sources": int(len(sources["validation"])),
        "test_sources": int(len(sources["test"])),
        "train_validation_disjoint": sources["train"].isdisjoint(sources["validation"]),
        "train_test_disjoint": sources["train"].isdisjoint(sources["test"]),
        "validation_test_disjoint": sources["validation"].isdisjoint(sources["test"]),
        "train_validation_fingerprint_disjoint": fingerprints["train"].isdisjoint(fingerprints["validation"]),
        "train_test_fingerprint_disjoint": fingerprints["train"].isdisjoint(fingerprints["test"]),
        "validation_test_fingerprint_disjoint": fingerprints["validation"].isdisjoint(fingerprints["test"]),
    }


def _make_train_dataset(cfg: PurifierConfig) -> tuple[PairManifestDataset, SourceBalancedPairDataset]:
    base = PairManifestDataset(
        cfg.train_manifest,
        cfg.clean_path,
        expected_split="train",
        seq_length=int(cfg.seq_length),
        value_scale=float(cfg.value_scale),
        max_sources=int(cfg.max_train_sources),
        preload_shards=bool(cfg.preload_shards),
        max_open_shards=int(cfg.max_open_shards),
        pairing_mode=str(cfg.pairing_mode),
        condition_source=str(cfg.condition_source),
    )
    return base, SourceBalancedPairDataset(base, seed=int(cfg.seed))


def _make_validation_dataset(cfg: PurifierConfig) -> PairManifestDataset:
    return PairManifestDataset(
        cfg.validation_manifest,
        cfg.clean_path,
        expected_split="validation",
        seq_length=int(cfg.seq_length),
        value_scale=float(cfg.value_scale),
        max_sources=int(cfg.max_validation_sources),
        preload_shards=bool(cfg.preload_shards),
        max_open_shards=int(cfg.max_open_shards),
        pairing_mode="correct",
        condition_source=str(cfg.condition_source),
    )


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    clean = batch["clean"].to(device=device, dtype=torch.float32, non_blocking=True)
    defended = batch["defended"].to(device=device, dtype=torch.float32, non_blocking=True)
    return clean, defended


def _labels_to_device(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return batch["label"].to(device=device, dtype=torch.long, non_blocking=True)


def validate_purifier(
    model: ConditionalTrafficPurifier,
    dataset: PairManifestDataset,
    cfg: PurifierConfig,
    device: torch.device,
    *,
    batch_size: int | None = None,
    validation_seed: int | None = None,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(batch_size or cfg.batch_size), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(validation_seed if validation_seed is not None else int(cfg.seed) + 9001))
    pair_diff_sum = 0.0
    pair_l1_sum = 0.0
    pair_mse_sum = 0.0
    pair_count = 0
    source_sums: dict[int, list[float]] = {}
    variant_sums: dict[int, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            clean, defended = _batch_to_device(batch, device)
            labels = _labels_to_device(batch, device)
            bsz = int(clean.shape[0])
            timesteps = torch.randint(0, model.diffusion.diffusion_steps, (bsz,), device=device, dtype=torch.long, generator=generator)
            noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
            x_t, target_noise = model.diffusion.q_sample(clean, timesteps, noise=noise)
            predicted_noise = model.predict_noise(x_t, timesteps, defended, labels=labels)
            predicted_x0 = model.diffusion.predict_x0_from_noise(x_t, timesteps, predicted_noise)
            diff_per = torch.mean((predicted_noise - target_noise) ** 2, dim=1).detach().cpu().numpy()
            l1_per = torch.mean(torch.abs(predicted_x0 - clean), dim=1).detach().cpu().numpy()
            mse_per = torch.mean((predicted_x0 - clean) ** 2, dim=1).detach().cpu().numpy()
            source_ids = batch["source_id"].numpy()
            variant_ids = batch["variant_id"].numpy()
            for index, source_id in enumerate(source_ids.tolist()):
                source_sums.setdefault(int(source_id), [0.0, 0.0])
                source_sums[int(source_id)][0] += float(l1_per[index])
                source_sums[int(source_id)][1] += 1.0
            for index, variant_id in enumerate(variant_ids.tolist()):
                variant_sums.setdefault(int(variant_id), [0.0, 0.0])
                variant_sums[int(variant_id)][0] += float(l1_per[index])
                variant_sums[int(variant_id)][1] += 1.0
            pair_diff_sum += float(np.sum(diff_per))
            pair_l1_sum += float(np.sum(l1_per))
            pair_mse_sum += float(np.sum(mse_per))
            pair_count += bsz
    source_values = [total / max(count, 1.0) for total, count in source_sums.values()]
    per_variant = {
        str(variant): {"l1": total / max(count, 1.0), "pairs": int(count)}
        for variant, (total, count) in sorted(variant_sums.items())
    }
    return {
        "pair_count": int(pair_count),
        "unique_source_count": int(len(source_sums)),
        "pair_diffusion_loss": pair_diff_sum / max(pair_count, 1),
        "pair_l1": pair_l1_sum / max(pair_count, 1),
        "pair_mse": pair_mse_sum / max(pair_count, 1),
        "source_l1": float(np.mean(source_values)) if source_values else math.inf,
        "per_variant": per_variant,
    }


def _save_checkpoint(
    path: Path,
    model: ConditionalTrafficPurifier,
    cfg: PurifierConfig,
    epoch: int,
    metrics: dict[str, Any],
    config_hash: str,
    manifest_hashes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.model_config(),
            "config": cfg.to_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "config_hash": config_hash,
            "manifest_hashes": manifest_hashes,
            "representation": "fixed_length_signed_time_sequence_v1_scaled",
            "legalization_version": model.decoder.legalization_version,
            "classifier_free": True,
        },
        path,
    )


def train_purifier(cfg: PurifierConfig, *, overwrite: bool = False) -> dict[str, Any]:
    pair_audit = _check_pair_audit(cfg)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    run_dir = _run_dir(cfg, overwrite=overwrite)
    cfg = replace(cfg, run_name=run_dir.name)
    write_json(run_dir / "config.json", cfg.to_dict())
    write_json(run_dir / "pair_audit_verification.json", pair_audit)
    config_hash = _json_hash(cfg.to_dict())
    manifest_hashes = _manifest_hashes(cfg)
    write_json(run_dir / "manifest_hashes.json", manifest_hashes)
    split_assertions = _split_assertions(cfg)
    write_json(run_dir / "split_leakage_assertions.json", split_assertions)

    train_base, train_dataset = _make_train_dataset(cfg)
    validation_dataset = _make_validation_dataset(cfg)
    split_summary = {
        "train": train_base.summary().__dict__,
        "validation": validation_dataset.summary().__dict__,
        "source_balanced_effective_train_pairs_per_epoch": int(len(train_dataset)),
        "pairing_mode": str(cfg.pairing_mode),
        "condition_mode": str(cfg.condition_mode),
        "condition_source": str(cfg.condition_source),
    }
    write_json(run_dir / "split_summary.json", split_summary)

    model = build_purifier(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg.epochs)))
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_epoch = 0
    latest_gradient_report: dict[str, Any] = {}
    checkpoints = run_dir / "checkpoints"

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        train_dataset.set_epoch(epoch)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(cfg.seed) + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        train_loss_sum = 0.0
        train_diff_sum = 0.0
        train_rec_sum = 0.0
        train_count = 0
        force_zero = str(cfg.condition_mode) == "zero"
        for step, batch in enumerate(loader, start=1):
            clean, defended = _batch_to_device(batch, device)
            labels = _labels_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, parts = model.training_losses(
                    clean,
                    defended,
                    labels=labels,
                    lambda_rec=float(cfg.lambda_rec),
                    reconstruction_mode=str(cfg.reconstruction_loss),
                    force_zero_condition=force_zero,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            latest_gradient_report = gradient_report(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            batch_n = int(clean.shape[0])
            train_loss_sum += float(parts["loss"].detach().cpu()) * batch_n
            train_diff_sum += float(parts["diffusion_loss"].detach().cpu()) * batch_n
            train_rec_sum += float(parts["reconstruction_loss"].detach().cpu()) * batch_n
            train_count += batch_n
            if int(cfg.log_every) > 0 and (step == 1 or step % int(cfg.log_every) == 0 or step == len(loader)):
                print(
                    f"[purifier train] epoch={epoch}/{cfg.epochs} step={step}/{len(loader)} "
                    f"loss={train_loss_sum / max(train_count, 1):.6f} diff={train_diff_sum / max(train_count, 1):.6f} "
                    f"rec={train_rec_sum / max(train_count, 1):.6f}",
                    flush=True,
                )
        scheduler.step()
        train_metrics = {
            "loss": train_loss_sum / max(train_count, 1),
            "diffusion_loss": train_diff_sum / max(train_count, 1),
            "reconstruction_loss": train_rec_sum / max(train_count, 1),
            "samples": int(train_count),
            "source_balanced_epoch": train_dataset.epoch_stats(),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        validate_now = epoch == int(cfg.epochs) or epoch % max(1, int(cfg.validate_every)) == 0
        validation_metrics: dict[str, Any] = {}
        if validate_now:
            validation_metrics = validate_purifier(
                model,
                validation_dataset,
                cfg,
                device,
                validation_seed=int(cfg.seed) + 9001,
            )
            primary = float(validation_metrics["source_l1"])
            secondary = float(validation_metrics["pair_diffusion_loss"])
            improved = best is None or primary < float(best["primary_metric"]) or (
                math.isclose(primary, float(best["primary_metric"])) and secondary < float(best["secondary_metric"])
            )
            if improved:
                best_epoch = epoch
                best = {
                    "selected_epoch": int(epoch),
                    "primary_metric": primary,
                    "secondary_metric": secondary,
                    "validation_metric": validation_metrics,
                }
                _save_checkpoint(checkpoints / "best_checkpoint.pt", model, cfg, epoch, validation_metrics, config_hash, manifest_hashes)
            print(
                f"[purifier validation] epoch={epoch} source_l1={validation_metrics['source_l1']:.6f} "
                f"pair_l1={validation_metrics['pair_l1']:.6f} diff={validation_metrics['pair_diffusion_loss']:.6f}",
                flush=True,
            )
        _save_checkpoint(checkpoints / "last_checkpoint.pt", model, cfg, epoch, validation_metrics or train_metrics, config_hash, manifest_hashes)
        history.append({"epoch": int(epoch), "train": train_metrics, "validation": validation_metrics})
        write_json(run_dir / "training_history.json", history)
        write_json(run_dir / "gradient_flow_report.json", latest_gradient_report)

    if best is None:
        raise RuntimeError("No validation checkpoint was selected")
    selection = {
        **best,
        "lambda_rec": float(cfg.lambda_rec),
        "diffusion_steps": int(cfg.diffusion_steps),
        "sampling_steps": int(cfg.sampling_steps),
        "random_seed": int(cfg.seed),
        "manifest_hash": manifest_hashes,
        "config_hash": config_hash,
        "selection_scope": "validation_only",
        "primary_rule": "min validation source-level L1 reconstruction",
        "secondary_rule": "min validation pair-level diffusion loss",
        "test_metric_used": False,
        "test_loader_constructed": False,
    }
    write_json(run_dir / "checkpoint_selection.json", selection)
    write_json(run_dir / "gradient_flow_report.json", latest_gradient_report)
    result = {
        "run_dir": str(run_dir.resolve()),
        "best_checkpoint": str((checkpoints / "best_checkpoint.pt").resolve()),
        "last_checkpoint": str((checkpoints / "last_checkpoint.pt").resolve()),
        "selected_epoch": int(best_epoch),
        "checkpoint_selection": selection,
        "gradient_flow_report": latest_gradient_report,
        "split_summary": split_summary,
        "pair_audit": pair_audit,
    }
    write_json(run_dir / "training_summary.json", result)
    return result


def load_purifier_checkpoint(path: str | Path, device: torch.device) -> tuple[ConditionalTrafficPurifier, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg_map = payload.get("model_config", {})

    class _Cfg:
        pass

    cfg = _Cfg()
    for key, value in cfg_map.items():
        setattr(cfg, key, value)
    # Compatibility with build_purifier naming.
    setattr(cfg, "dropout", 0.0)
    model = build_purifier(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
