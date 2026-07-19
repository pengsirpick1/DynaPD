"""Self-contained fixed/mixed attack evaluation for DMMPv3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.config import AttackConfig, DefenseConfig
from ..evaluation.attack_models import build_df_input, build_rf_tam_input, make_attack_model
from ..data import load_cw_data
from ..diffusion.pipeline import generate_defended_dataset_from_pool
from ..encoders.prefix import nonzero_trace
from ..utils import log, resolve_device, set_seed, write_csv, write_json


def _defense_config_from_run(run_dir: Path, attack_cfg: AttackConfig) -> DefenseConfig:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    values = {key: payload[key] for key in DefenseConfig.__dataclass_fields__ if key in payload}
    cfg = DefenseConfig(**values)
    if attack_cfg.data_root:
        cfg.data_root = attack_cfg.data_root
    cfg.progress = attack_cfg.progress
    cfg.device = attack_cfg.device
    return cfg


def _load_splits(run_dir: Path) -> dict[str, np.ndarray]:
    payload = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def _subsample(indices: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if int(max_count) <= 0 or len(indices) <= int(max_count):
        return indices
    rng = np.random.default_rng(int(seed))
    return np.asarray(sorted(rng.choice(indices, size=int(max_count), replace=False)), dtype=np.int64)


def direction_input(raw: np.ndarray, max_len: int) -> np.ndarray:
    return build_df_input(raw, max_len=int(max_len)).astype(np.float32)


def tam_features(raw: np.ndarray, slots: int = 1800, max_len: int = 5000, max_load_time: float = 80.0) -> np.ndarray:
    return build_rf_tam_input(
        raw,
        max_len=int(max_len),
        max_load_time=float(max_load_time),
        num_slots=int(slots),
    ).astype(np.float32)


class TinyDF(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.net(x.float()).flatten(1))


class CentroidClassifier:
    def __init__(self):
        self.classes: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.mean = x.mean(axis=0, keepdims=True)
        self.scale = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
        z = (x - self.mean) / self.scale
        self.classes = np.unique(y)
        self.centroids = np.stack([z[y == label].mean(axis=0) for label in self.classes], axis=0).astype(np.float32)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.classes is not None and self.centroids is not None and self.mean is not None and self.scale is not None
        z = (np.asarray(x, dtype=np.float32) - self.mean) / self.scale
        logits = -np.sum((z[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(logits, -60.0, 0.0))
        return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _metrics_from_proba(probs: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    pred = classes[np.argmax(probs, axis=1)]
    class_to_pos = {int(label): pos for pos, label in enumerate(classes.tolist())}
    true_pos = np.asarray([class_to_pos.get(int(label), -1) for label in y], dtype=np.int64)
    valid = true_pos >= 0
    true_conf = np.zeros(y.shape[0], dtype=np.float32)
    true_conf[valid] = probs[np.arange(y.shape[0])[valid], true_pos[valid]]
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    if probs.shape[1] > 1:
        entropy = entropy / np.log(probs.shape[1])
    return {
        "defended_accuracy": float(np.mean(pred == y)) if y.size else 0.0,
        "true_label_confidence": float(np.mean(true_conf)) if true_conf.size else 0.0,
        "prediction_entropy": float(np.mean(entropy)) if entropy.size else 0.0,
        "max_confidence": float(np.mean(np.max(probs, axis=1))) if probs.size else 0.0,
    }


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), generator=generator)


def _eval_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, classes: np.ndarray, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for xb, _ in _make_loader(x, y, batch_size, seed=0, shuffle=False):
            logits = model(xb.to(device))
            rows.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    probs = np.concatenate(rows, axis=0) if rows else np.zeros((0, len(classes)), dtype=np.float32)
    return _metrics_from_proba(probs, y, classes)


def train_df_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    attacker_kind: str,
    defense_cfg: DefenseConfig,
    attack_cfg: AttackConfig,
    initial_state: dict[str, torch.Tensor] | None = None,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
    progress: bool,
) -> tuple[nn.Module, np.ndarray, float]:
    classes = np.unique(train_y)
    class_to_pos = {int(label): pos for pos, label in enumerate(classes.tolist())}
    y_train_pos = np.asarray([class_to_pos[int(label)] for label in train_y], dtype=np.int64)
    y_val_pos = np.asarray([class_to_pos[int(label)] for label in val_y], dtype=np.int64)
    model = make_attack_model(
        str(attacker_kind).upper(),
        len(classes),
        max_trace_length=int(defense_cfg.max_trace_length),
        df_architecture=str(attack_cfg.df_architecture),
    ).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    opt = torch.optim.Adamax(model.parameters(), lr=float(lr), weight_decay=1e-5)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_acc = -1.0
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_batches = int((len(train_x) + max(int(batch_size), 1) - 1) // max(int(batch_size), 1))
        heartbeat = max(1, min(max(int(getattr(attack_cfg, "log_every", 100)), 1), max(total_batches // 4, 1)))
        for batch_index, (xb, yb) in enumerate(_make_loader(train_x, y_train_pos, batch_size, seed + epoch, shuffle=True), start=1):
            logits = model(xb.to(device))
            loss = nn.functional.cross_entropy(logits, yb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if batch_index == 1 or batch_index == total_batches or batch_index % heartbeat == 0:
                log(
                    f"{str(attacker_kind).upper()} train: epoch {epoch}/{epochs}, "
                    f"batch {batch_index}/{total_batches}, loss={float(loss.detach().cpu()):.6f}",
                    progress,
                )
        val_metrics = _eval_torch(model, val_x, y_val_pos, np.arange(len(classes)), device, batch_size)
        val_acc = float(val_metrics["defended_accuracy"])
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        log(
            f"{str(attacker_kind).upper()} train: epoch {epoch}/{epochs}, "
            f"val_acc={val_acc:.6f}, best={best_acc:.6f}, stale={stale}/{patience}",
            progress,
        )
        if stale >= int(patience):
            break
    model.load_state_dict(best_state)
    return model, classes.astype(np.int64), float(best_acc)


def train_rf_model(train_x: np.ndarray, train_y: np.ndarray):
    try:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=0, class_weight="balanced_subsample")
        model.fit(train_x, train_y)
        return model, np.asarray(model.classes_, dtype=np.int64), "sklearn_random_forest"
    except Exception:
        model = CentroidClassifier().fit(train_x, train_y)
        return model, np.asarray(model.classes, dtype=np.int64), "centroid_fallback"


def _eval_rf(model, x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    probs = model.predict_proba(x)
    return _metrics_from_proba(probs, y, classes)


def _attack_input(kind: str, raw: np.ndarray, cfg: DefenseConfig, attack_cfg: AttackConfig) -> np.ndarray:
    if kind == "df":
        return direction_input(raw, int(cfg.max_trace_length))
    return tam_features(
        raw,
        int(attack_cfg.rf_tam_num_slots),
        max_len=int(cfg.max_trace_length),
        max_load_time=float(attack_cfg.max_load_time),
    )


def _get_or_generate(
    name: str,
    raw: np.ndarray,
    y: np.ndarray,
    trace_ids: np.ndarray,
    indices: np.ndarray,
    run_dir: Path,
    cfg: DefenseConfig,
    out_dir: Path,
    *,
    defense_seed: int,
    policy_variant: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    path = out_dir / f"{name}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as arrays:
            return np.asarray(arrays["X"], dtype=np.float32), np.asarray(arrays["y"], dtype=np.int64)
    defended, labels, _ = generate_defended_dataset_from_pool(
        raw,
        y,
        trace_ids,
        indices,
        run_dir,
        cfg,
        defense_seed=int(defense_seed),
        output_npz=path,
        policy_variant=str(policy_variant),
        device=device,
    )
    return defended, labels


def _summary_row_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    clean = metrics.get("clean_test", {}) or {}
    defended = metrics.get("fresh_defended_test", {}) or {}
    return {
        "setting": metrics.get("setting", ""),
        "attacker": metrics.get("attacker", ""),
        "clean_acc": clean.get("defended_accuracy", 0.0),
        "fresh_defended_acc": defended.get("defended_accuracy", 0.0),
        "fresh_defended_entropy": defended.get("prediction_entropy", 0.0),
        "fresh_defended_max_conf": defended.get("max_confidence", 0.0),
    }


def _load_checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        return None
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def run_attack_evaluation(attack_cfg: AttackConfig) -> Path:
    set_seed(int(attack_cfg.seed))
    run_dir = Path(attack_cfg.run_dir)
    cfg = _defense_config_from_run(run_dir, attack_cfg)
    device = resolve_device(str(attack_cfg.device))
    raw, y, trace_ids, _, data_source = load_cw_data(cfg)
    splits = _load_splits(run_dir)
    train_idx = _subsample(splits["train"], int(attack_cfg.max_train_traces), int(attack_cfg.seed) + 11)
    val_idx = _subsample(splits["val"], int(attack_cfg.max_val_traces), int(attack_cfg.seed) + 12)
    test_idx = _subsample(splits["test"], int(attack_cfg.max_test_traces), int(attack_cfg.seed) + 13)
    out_dir = Path(attack_cfg.output_dir) if attack_cfg.output_dir else run_dir / "dmmpv3_attack_eval"
    data_dir = out_dir / "pool_resampled_defended_datasets"
    data_dir.mkdir(parents=True, exist_ok=True)

    requested = [item.lower() for item in attack_cfg.attacker_values]
    need_fixed = any(item.startswith("fixed") for item in requested)
    need_mixed = any(item.startswith("mixed") for item in requested)
    fixed_def_x = fixed_def_y = None
    mixed_train_x = mixed_train_y = mixed_val_x = mixed_val_y = mixed_test_x = mixed_test_y = None
    if need_fixed:
        fixed_def_x, fixed_def_y = _get_or_generate(
            "fixed_fresh_test",
            raw,
            y,
            trace_ids,
            test_idx,
            run_dir,
            cfg,
            data_dir,
            defense_seed=int(attack_cfg.fixed_eval_defense_seed),
            policy_variant=str(attack_cfg.policy_variant),
            device=device,
        )
    if need_mixed:
        mixed_train_x, mixed_train_y = _get_or_generate(
            "mixed_pool_train",
            raw,
            y,
            trace_ids,
            train_idx,
            run_dir,
            cfg,
            data_dir,
            defense_seed=int(attack_cfg.mixed_train_defense_seed),
            policy_variant=str(attack_cfg.policy_variant),
            device=device,
        )
        mixed_val_x, mixed_val_y = _get_or_generate(
            "mixed_pool_val",
            raw,
            y,
            trace_ids,
            val_idx,
            run_dir,
            cfg,
            data_dir,
            defense_seed=int(attack_cfg.mixed_val_defense_seed),
            policy_variant=str(attack_cfg.policy_variant),
            device=device,
        )
        mixed_test_x, mixed_test_y = _get_or_generate(
            "mixed_fresh_deploy_test",
            raw,
            y,
            trace_ids,
            test_idx,
            run_dir,
            cfg,
            data_dir,
            defense_seed=int(attack_cfg.mixed_test_defense_seed),
            policy_variant=str(attack_cfg.policy_variant),
            device=device,
        )

    clean_train, clean_val, clean_test = raw[train_idx], raw[val_idx], raw[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    rows: list[dict[str, Any]] = []
    fixed_states: dict[str, dict[str, torch.Tensor]] = {}

    for setting in attack_cfg.attacker_values:
        setting = setting.lower()
        if setting not in {"fixed_df", "fixed_rf", "mixed_df", "mixed_rf"}:
            continue
        kind = "df" if setting.endswith("df") else "rf"
        setting_dir = out_dir / setting
        setting_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = setting_dir / f"{setting}_metrics.json"
        checkpoint_path = setting_dir / f"{setting}_checkpoint.pt"
        if metrics_path.is_file() and checkpoint_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if setting.startswith("fixed"):
                state = _load_checkpoint_state(checkpoint_path, device)
                if state is not None:
                    fixed_states[kind] = state
            rows.append(_summary_row_from_metrics(metrics))
            log(f"Skip {setting}: existing metrics/checkpoint found.", attack_cfg.progress)
            continue

        if setting.startswith("fixed"):
            assert fixed_def_x is not None and fixed_def_y is not None
            train_x = _attack_input(kind, clean_train, cfg, attack_cfg)
            val_x = _attack_input(kind, clean_val, cfg, attack_cfg)
            train_labels = y_train
            val_labels = y_val
            defended_test_x = _attack_input(kind, fixed_def_x, cfg, attack_cfg)
            defended_labels = fixed_def_y
        else:
            assert mixed_train_x is not None and mixed_train_y is not None
            assert mixed_val_x is not None and mixed_val_y is not None
            assert mixed_test_x is not None and mixed_test_y is not None
            clean_train_x = _attack_input(kind, clean_train, cfg, attack_cfg)
            defended_train_x = _attack_input(kind, mixed_train_x, cfg, attack_cfg)
            clean_val_x = _attack_input(kind, clean_val, cfg, attack_cfg)
            defended_val_x = _attack_input(kind, mixed_val_x, cfg, attack_cfg)
            train_x = np.concatenate([clean_train_x, defended_train_x], axis=0)
            train_labels = np.concatenate([y_train, mixed_train_y], axis=0)
            val_x = np.concatenate([clean_val_x, defended_val_x], axis=0)
            val_labels = np.concatenate([y_val, mixed_val_y], axis=0)
            defended_test_x = _attack_input(kind, mixed_test_x, cfg, attack_cfg)
            defended_labels = mixed_test_y

        clean_test_x = _attack_input(kind, clean_test, cfg, attack_cfg)

        is_fixed = setting.startswith("fixed")
        initial_state = None
        if not is_fixed and str(attack_cfg.adaptive_init) == "checkpoint" and kind in fixed_states:
            initial_state = {key: value.detach().cpu().clone() for key, value in fixed_states[kind].items()}
        elif not is_fixed and str(attack_cfg.adaptive_init) == "checkpoint":
            fixed_path = out_dir / f"fixed_{kind}" / f"fixed_{kind}_checkpoint.pt"
            initial_state = _load_checkpoint_state(fixed_path, device)
        model, classes, best_val = train_df_model(
            train_x,
            train_labels,
            val_x,
            val_labels,
            attacker_kind=kind.upper(),
            defense_cfg=cfg,
            attack_cfg=attack_cfg,
            initial_state=initial_state,
            epochs=int(attack_cfg.clean_df_epochs if is_fixed else attack_cfg.adaptive_epochs),
            patience=int(attack_cfg.clean_df_patience if is_fixed else attack_cfg.adaptive_patience),
            lr=float(attack_cfg.clean_df_lr if is_fixed else attack_cfg.adaptive_lr),
            batch_size=int(attack_cfg.df_batch_size),
            device=device,
            seed=int(attack_cfg.seed),
            progress=bool(attack_cfg.progress),
        )
        if is_fixed:
            fixed_states[kind] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        clean_metrics = _eval_torch(model, clean_test_x, y_test, classes, device, int(attack_cfg.df_batch_size))
        defended_metrics = _eval_torch(model, defended_test_x, defended_labels, classes, device, int(attack_cfg.df_batch_size))
        torch.save({"model_state": model.state_dict(), "classes": classes, "best_val": best_val}, setting_dir / f"{setting}_checkpoint.pt")
        model_name = f"{attack_cfg.df_architecture}_df" if kind == "df" else "project_rf_tam"

        metrics = {
            "attacker": kind.upper(),
            "setting": setting,
            "model": model_name,
            "best_val_accuracy": float(best_val),
            "clean_test": clean_metrics,
            "fresh_defended_test": defended_metrics,
            "protocol": "current_method_only_pool_resampled_mixed",
        }
        write_json(setting_dir / f"{setting}_metrics.json", metrics)
        rows.append(
            {
                "setting": setting,
                "attacker": kind.upper(),
                "clean_acc": clean_metrics["defended_accuracy"],
                "fresh_defended_acc": defended_metrics["defended_accuracy"],
                "fresh_defended_entropy": defended_metrics["prediction_entropy"],
                "fresh_defended_max_conf": defended_metrics["max_confidence"],
            }
        )

    summary = {
            "protocol": "dmmpv3_current_method_only_pool_resampled_mixed",
        "policy_variant": str(attack_cfg.policy_variant),
        "data_source": data_source,
        "rows": rows,
        "defense_pool_seed_protocol": {
            "fixed_eval_defense_seed": int(attack_cfg.fixed_eval_defense_seed),
            "mixed_train_defense_seed": int(attack_cfg.mixed_train_defense_seed),
            "mixed_val_defense_seed": int(attack_cfg.mixed_val_defense_seed),
            "mixed_test_defense_seed": int(attack_cfg.mixed_test_defense_seed),
        },
    }
    write_json(out_dir / "current_method_attack_summary.json", summary)
    write_csv(out_dir / "current_method_attack_summary.csv", rows)
    lines = [
        "# DMMPv3 current method attack evaluation",
        "",
        "| setting | attacker | clean acc | fresh defended acc | entropy | max confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting']} | {row['attacker']} | {row['clean_acc']:.6f} | {row['fresh_defended_acc']:.6f} | {row['fresh_defended_entropy']:.6f} | {row['fresh_defended_max_conf']:.6f} |"
        )
    (out_dir / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / "attack_eval_config.json", vars(attack_cfg))
    log(f"[done] DMMPv3 attack eval saved to: {out_dir}", attack_cfg.progress)
    return out_dir

