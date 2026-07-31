"""Configuration helpers for the conditional traffic purifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from dmmp.target_policy.config import _load_yaml_like


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CLEAN_PATH = REPO_ROOT / "datasets" / "CW" / "CW.npz"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _resolve_optional_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> str:
    if not str(value).strip():
        return ""
    return _resolve_path(value, base=base)


@dataclass
class PurifierConfig:
    experiment_name: str = "conditional_purifier_v1"
    # Optional provenance for legacy DMMPv3-produced manifests. The purifier
    # must be trainable from manifests alone after defended traces are generated.
    run_dir: str = ""
    clean_path: str = str(DEFAULT_CLEAN_PATH)
    pair_manifest: str = ""
    pair_audit_report: str = ""
    requires_pair_audit_pass: bool = True
    label_free_purifier: bool = True
    train_manifest: str = ""
    validation_manifest: str = ""
    test_manifest: str = ""
    output_root: str = str(PROJECT_ROOT / "results" / "purifier_runs")
    run_name: str = ""
    seed: int = 0
    device: str = "auto"
    seq_length: int = 10000
    value_scale: float = 80.0
    value_clip: float = 80.0
    zero_threshold: float = 0.03
    diffusion_steps: int = 32
    sampling_steps: int = 8
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    hidden_channels: int = 32
    condition_channels: int = 32
    num_classes: int = 95
    time_dim: int = 128
    num_denoiser_blocks: int = 6
    dropout: float = 0.0
    lambda_rec: float = 0.05
    reconstruction_loss: str = "smooth_l1"
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip: float = 1.0
    use_amp: bool = True
    max_train_sources: int = 0
    max_validation_sources: int = 0
    log_every: int = 100
    validate_every: int = 1
    num_workers: int = 0
    preload_shards: bool = True
    max_open_shards: int = 2
    pairing_mode: str = "correct"
    condition_mode: str = "conditioned"
    condition_source: str = "defended"
    output_length_policy: str = "model"
    selection_primary: str = "validation_source_l1"
    selection_secondary: str = "validation_diffusion_loss"
    manifest_hash_bytes: int = 1024 * 1024

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PurifierConfig":
        flat: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
                    flat[sub_key] = sub_value
            else:
                flat[key] = value

        if "pair_manifest" in flat and "train_manifest" not in flat:
            manifest = Path(str(flat["pair_manifest"]))
            if manifest.name == "purifier_clean_defended_pairs.csv":
                root = manifest.parent
                flat.setdefault("train_manifest", str(root / "purifier_train_pairs.csv"))
                flat.setdefault("validation_manifest", str(root / "purifier_validation_pairs.csv"))
                flat.setdefault("test_manifest", str(root / "purifier_test_pairs.csv"))
                flat.setdefault("pair_audit_report", str(root / "purifier_split_audit_report.json"))
        if "loss.reconstruction_weight" in flat:
            flat["lambda_rec"] = flat["loss.reconstruction_weight"]
        if "reconstruction_weight" in flat:
            flat["lambda_rec"] = flat["reconstruction_weight"]
        if "sampling.reverse_steps" in flat:
            flat["sampling_steps"] = flat["sampling.reverse_steps"]
        if "reverse_steps" in flat:
            flat["sampling_steps"] = flat["reverse_steps"]
        if "model.hidden_channels" in flat:
            flat["hidden_channels"] = flat["model.hidden_channels"]
        if "model.condition_channels" in flat:
            flat["condition_channels"] = flat["model.condition_channels"]
        if "training.epochs" in flat:
            flat["epochs"] = flat["training.epochs"]
        if "training.batch_size" in flat:
            flat["batch_size"] = flat["training.batch_size"]
        if "training.learning_rate" in flat:
            flat["learning_rate"] = flat["training.learning_rate"]
        if "output.length_policy" in flat:
            flat["output_length_policy"] = flat["output.length_policy"]

        known = {field.name for field in fields(cls)}
        values: dict[str, Any] = {}
        for key, value in flat.items():
            if key not in known:
                continue
            if key in {"preload_shards", "use_amp", "requires_pair_audit_pass", "label_free_purifier"}:
                values[key] = _bool(value)
            else:
                values[key] = value
        cfg = cls(**values)
        cfg.run_dir = _resolve_optional_path(cfg.run_dir)
        cfg.clean_path = _resolve_path(cfg.clean_path)
        cfg.pair_manifest = _resolve_optional_path(cfg.pair_manifest)

        manifest_dir: Path | None = None
        if cfg.pair_manifest:
            manifest_dir = Path(cfg.pair_manifest).parent
        elif cfg.run_dir:
            manifest_dir = Path(cfg.run_dir) / "manifests"

        if not cfg.train_manifest and manifest_dir is not None:
            cfg.train_manifest = str(manifest_dir / "purifier_train_pairs.csv")
        if not cfg.validation_manifest and manifest_dir is not None:
            cfg.validation_manifest = str(manifest_dir / "purifier_validation_pairs.csv")
        if not cfg.test_manifest and manifest_dir is not None:
            cfg.test_manifest = str(manifest_dir / "purifier_test_pairs.csv")
        if not (cfg.train_manifest and cfg.validation_manifest and cfg.test_manifest):
            raise ValueError("PurifierConfig requires train_manifest, validation_manifest, and test_manifest, or a pair_manifest/run_dir to derive them")
        cfg.train_manifest = _resolve_path(cfg.train_manifest)
        cfg.validation_manifest = _resolve_path(cfg.validation_manifest)
        cfg.test_manifest = _resolve_path(cfg.test_manifest)
        if not cfg.pair_audit_report:
            if cfg.pair_manifest:
                cfg.pair_audit_report = str(Path(cfg.pair_manifest).parent / "purifier_split_audit_report.json")
            elif cfg.run_dir:
                cfg.pair_audit_report = str(Path(cfg.run_dir) / "manifests" / "purifier_split_audit_report.json")
        cfg.pair_audit_report = _resolve_optional_path(cfg.pair_audit_report)
        cfg.output_root = _resolve_path(cfg.output_root)
        cfg.output_length_policy = str(cfg.output_length_policy).strip().lower()
        if cfg.output_length_policy not in {"model", "defended", "clean"}:
            raise ValueError("output_length_policy must be one of: model, defended, clean")
        cfg.condition_source = str(cfg.condition_source).strip().lower()
        if cfg.condition_source not in {"defended", "clean", "label"}:
            raise ValueError("condition_source must be one of: defended, clean, label")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_purifier_config(path: str | Path | None = None) -> PurifierConfig:
    if path is None or not str(path).strip():
        return PurifierConfig.from_mapping({})
    return PurifierConfig.from_mapping(_load_yaml_like(Path(path)))
