# -*- coding: utf-8 -*-
"""Run a DTPN-style traffic purification experiment on the local DMMPv3 data.

This is a protocol-aligned reproduction for the current project format:

- x0 is the defended/adversarial trace, not the clean trace.
- c is the frozen DF penultimate-layer embedding of x0, not a class label.
- training uses diffusion noise prediction plus a frozen-DF CE auxiliary loss.
- purification starts from defended traffic noised to a small t* and reverses to t=0.

The original DTPN paper uses burst sequences of length 512. This script keeps
the current signed timestamp representation so the existing frozen DF/RF
checkpoints can evaluate the generated traffic directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import stored_npy_from_npz
from dmmp.evaluation.attack_models import build_df_input, build_rf_tam_input, make_attack_model
from dmmp.purifier.condition_encoder import ConditionFeatures
from dmmp.purifier.dataset import PairManifestDataset, SourceBalancedPairDataset
from dmmp.purifier.decoder import TrafficOutputDecoder
from dmmp.purifier.denoiser import ConditionalDenoiser
from dmmp.utils import resolve_device, set_seed, write_csv, write_json


DEFAULT_DEFENSE_RUN = PROJECT_ROOT / "results" / "dmmpv3_rf_tam_shape_v2_fullcw_b010_seed0_20260722_113000"
DEFAULT_CLEAN_PATH = REPO_ROOT / "datasets" / "CW" / "CW.npz"
DEFAULT_DF_CHECKPOINT = REPO_ROOT / "results" / "dmmp2_v5_fixed_oriented_seed0_bwo30" / "attack_eval" / "fixed" / "df" / "fixed_df_checkpoint.pt"
DEFAULT_RF_CHECKPOINT = REPO_ROOT / "results" / "dmmp2_v5_fixed_oriented_seed0_bwo30" / "attack_eval" / "fixed" / "rf" / "fixed_rf_checkpoint.pt"


@dataclass
class DTPNStyleConfig:
    experiment_name: str = "dtpn_style_timestamp_protocol"
    run_name: str = ""
    defense_run_dir: str = str(DEFAULT_DEFENSE_RUN)
    clean_path: str = str(DEFAULT_CLEAN_PATH)
    output_root: str = str(PROJECT_ROOT / "results" / "purifier_runs")
    df_checkpoint: str = str(DEFAULT_DF_CHECKPOINT)
    rf_checkpoint: str = str(DEFAULT_RF_CHECKPOINT)
    seed: int = 0
    device: str = "auto"
    seq_length: int = 5000
    max_trace_length: int = 5000
    value_scale: float = 80.0
    value_clip: float = 80.0
    zero_threshold: float = 0.03
    diffusion_steps: int = 1000
    beta_start: float = 0.0002
    beta_end: float = 0.025
    t_star: float = 0.005
    tau: float = 20.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 5
    hidden_channels: int = 32
    condition_channels: int = 128
    time_dim: int = 128
    num_denoiser_blocks: int = 4
    dropout: float = 0.0
    sign_temperature_seconds: float = 0.05
    max_train_sources: int = 20000
    max_validation_sources: int = 2000
    max_test_pairs: int = 0
    preload_shards: bool = True
    max_open_shards: int = 2
    log_every: int = 50
    validate_every: int = 1
    use_amp: bool = True
    gradient_clip: float = 1.0
    shard_size: int = 1024
    rf_tam_num_slots: int = 1800
    max_load_time: float = 80.0
    tam_num_slots: int = 1000
    tam_slot_ms: float = 80.0
    plot_row_index: int = 0

    @property
    def manifest_dir(self) -> Path:
        return Path(self.defense_run_dir) / "manifests"

    @property
    def train_manifest(self) -> Path:
        return self.manifest_dir / "purifier_train_pairs.csv"

    @property
    def validation_manifest(self) -> Path:
        return self.manifest_dir / "purifier_validation_pairs.csv"

    @property
    def test_manifest(self) -> Path:
        return self.manifest_dir / "purifier_test_pairs.csv"


class DTPNStylePurifier(nn.Module):
    """Small score model with DTPN-style DF embedding conditioning."""

    def __init__(self, cfg: DTPNStyleConfig, *, df_embedding_dim: int = 512):
        super().__init__()
        self.seq_length = int(cfg.seq_length)
        self.condition_channels = int(cfg.condition_channels)
        self.global_dim = int(cfg.condition_channels) * 2
        self.embedding_mlp = nn.Sequential(
            nn.Linear(int(df_embedding_dim), self.global_dim),
            nn.SiLU(),
            nn.Linear(self.global_dim, self.global_dim),
        )
        self.local_projection = nn.Linear(self.global_dim, int(cfg.condition_channels))
        self.denoiser = ConditionalDenoiser(
            hidden_channels=int(cfg.hidden_channels),
            condition_channels=int(cfg.condition_channels),
            global_dim=self.global_dim,
            time_dim=int(cfg.time_dim),
            num_blocks=int(cfg.num_denoiser_blocks),
            dropout=float(cfg.dropout),
        )
        self.diffusion = ContinuousVPDiffusion(
            diffusion_steps=int(cfg.diffusion_steps),
            beta_start=float(cfg.beta_start),
            beta_end=float(cfg.beta_end),
            x0_clip=float(cfg.value_clip) / max(float(cfg.value_scale), 1.0e-8),
        )
        self.decoder = TrafficOutputDecoder(
            value_scale=float(cfg.value_scale),
            value_clip=float(cfg.value_clip),
            zero_threshold=float(cfg.zero_threshold),
        )

    def encode_embedding(self, embedding: torch.Tensor, *, length: int, dtype: torch.dtype) -> ConditionFeatures:
        global_condition = self.embedding_mlp(embedding.float()).to(dtype=dtype)
        local = self.local_projection(global_condition).to(dtype=dtype).unsqueeze(-1).expand(-1, -1, int(length))
        half = max(1, int(length) // 2)
        quarter = max(1, int(length) // 4)
        return ConditionFeatures(
            c_local=local,
            c_multi=(local[..., :half].contiguous(), local[..., :quarter].contiguous()),
            c_global=global_condition,
        )

    def predict_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        condition = self.encode_embedding(embedding, length=int(x_t.shape[-1]), dtype=x_t.dtype)
        return self.denoiser(x_t, timesteps, condition)


class ContinuousVPDiffusion(nn.Module):
    """Continuous VP-SDE marginal used by DTPN/Song-style score training."""

    def __init__(self, *, diffusion_steps: int, beta_start: float, beta_end: float, x0_clip: float):
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.x0_clip = float(x0_clip)

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().clamp(0.0, 1.0)
        log_mean = -0.25 * t.square() * (self.beta_end - self.beta_start) - 0.5 * t * self.beta_start
        return torch.exp(log_mean)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        mean = self.mean_coeff(t)
        return torch.sqrt(torch.clamp(1.0 - mean.square(), min=1.0e-12))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        mean = self.mean_coeff(t).reshape(-1, *([1] * (x0.ndim - 1)))
        std = self.std(t).reshape(-1, *([1] * (x0.ndim - 1)))
        return mean * x0 + std * noise, noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        mean = self.mean_coeff(t).reshape(-1, *([1] * (x_t.ndim - 1)))
        std = self.std(t).reshape(-1, *([1] * (x_t.ndim - 1)))
        return (x_t - std * predicted_noise) / mean.clamp_min(1.0e-8)


class CleanStore:
    def __init__(self, path: str | Path):
        clean_path = Path(path)
        x_map = stored_npy_from_npz(clean_path, "X")
        y_map = stored_npy_from_npz(clean_path, "y")
        self._payload = None
        if x_map is not None and y_map is not None:
            self.x = x_map
            self.y = np.asarray(y_map, dtype=np.int64)
        else:
            self._payload = np.load(clean_path, allow_pickle=False)
            self.x = self._payload["X"]
            self.y = np.asarray(self._payload["y"], dtype=np.int64)

    def batch(self, indices: Iterable[int], seq_length: int) -> np.ndarray:
        idx = list(indices)
        out = np.zeros((len(idx), int(seq_length)), dtype=np.float32)
        for row, index in enumerate(idx):
            values = np.asarray(self.x[int(index)], dtype=np.float32).reshape(-1)
            take = min(out.shape[1], values.size)
            if take:
                out[row, :take] = values[:take]
        return out

    def close(self) -> None:
        if self._payload is not None:
            self._payload.close()


class NpzCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max(1, int(max_items))
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str | Path) -> dict[str, np.ndarray]:
        text = str(Path(path).resolve())
        if text in self.cache:
            item = self.cache.pop(text)
            self.cache[text] = item
            return item
        payload = np.load(text, allow_pickle=False)
        try:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        finally:
            payload.close()
        self.cache[text] = arrays
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return arrays


class MetricAccumulator:
    def __init__(self, classes: np.ndarray):
        self.classes = np.asarray(classes, dtype=np.int64)
        self.class_to_pos = {int(label): index for index, label in enumerate(self.classes.tolist())}
        n = len(self.classes)
        self.confusion = np.zeros((n, n), dtype=np.int64)
        self.total = 0
        self.true_conf_sum = 0.0
        self.entropy_sum = 0.0
        self.max_conf_sum = 0.0

    def update(self, logits: torch.Tensor, labels: np.ndarray) -> None:
        probs = torch.softmax(logits.detach().float().cpu(), dim=1).numpy()
        pred_pos = np.argmax(probs, axis=1)
        true_pos = np.asarray([self.class_to_pos[int(label)] for label in np.asarray(labels, dtype=np.int64)], dtype=np.int64)
        for truth, pred in zip(true_pos.tolist(), pred_pos.tolist()):
            self.confusion[int(truth), int(pred)] += 1
        self.total += int(len(labels))
        self.true_conf_sum += float(np.sum(probs[np.arange(len(labels)), true_pos]))
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1.0e-12)), axis=1)
        if probs.shape[1] > 1:
            entropy = entropy / np.log(probs.shape[1])
        self.entropy_sum += float(np.sum(entropy))
        self.max_conf_sum += float(np.sum(np.max(probs, axis=1)))

    def finalize(self) -> dict[str, float | int]:
        tp = np.diag(self.confusion).astype(np.float64)
        fp = self.confusion.sum(axis=0).astype(np.float64) - tp
        fn = self.confusion.sum(axis=1).astype(np.float64) - tp
        denom = 2.0 * tp + fp + fn
        f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        return {
            "count": int(self.total),
            "accuracy": float(tp.sum() / max(self.total, 1)),
            "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
            "true_label_confidence": float(self.true_conf_sum / max(self.total, 1)),
            "prediction_entropy": float(self.entropy_sum / max(self.total, 1)),
            "max_confidence": float(self.max_conf_sum / max(self.total, 1)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a DTPN-style purifier on DMMPv3 manifests.")
    parser.add_argument("--defense-run-dir", default=str(DEFAULT_DEFENSE_RUN))
    parser.add_argument("--clean-path", default=str(DEFAULT_CLEAN_PATH))
    parser.add_argument("--df-checkpoint", default=str(DEFAULT_DF_CHECKPOINT))
    parser.add_argument("--rf-checkpoint", default=str(DEFAULT_RF_CHECKPOINT))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "purifier_runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--t-star", type=float, default=0.005)
    parser.add_argument("--beta-start", type=float, default=0.0002)
    parser.add_argument("--beta-end", type=float, default=0.025)
    parser.add_argument("--seq-length", type=int, default=5000)
    parser.add_argument("--max-trace-length", type=int, default=5000)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--condition-channels", type=int, default=128)
    parser.add_argument("--num-denoiser-blocks", type=int, default=4)
    parser.add_argument("--max-train-sources", type=int, default=20000)
    parser.add_argument("--max-validation-sources", type=int, default=2000)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--checkpoint", default="", help="Use with --skip-training to generate/evaluate an existing checkpoint.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> DTPNStyleConfig:
    values = {
        "defense_run_dir": str(Path(args.defense_run_dir).resolve()),
        "clean_path": str(Path(args.clean_path).resolve()),
        "df_checkpoint": str(Path(args.df_checkpoint).resolve()),
        "rf_checkpoint": str(Path(args.rf_checkpoint).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "run_name": str(args.run_name),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "tau": float(args.tau),
        "diffusion_steps": int(args.diffusion_steps),
        "t_star": float(args.t_star),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "seq_length": int(args.seq_length),
        "max_trace_length": int(args.max_trace_length),
        "hidden_channels": int(args.hidden_channels),
        "condition_channels": int(args.condition_channels),
        "num_denoiser_blocks": int(args.num_denoiser_blocks),
        "max_train_sources": int(args.max_train_sources),
        "max_validation_sources": int(args.max_validation_sources),
        "max_test_pairs": int(args.max_test_pairs),
        "log_every": int(args.log_every),
        "validate_every": int(args.validate_every),
        "device": str(args.device),
        "seed": int(args.seed),
    }
    return DTPNStyleConfig(**values)


def run_dir_for(cfg: DTPNStyleConfig, overwrite: bool) -> Path:
    name = cfg.run_name.strip() or f"dtpn_style_b010_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_root) / name
    if run_dir.exists() and any((run_dir / item).exists() for item in ["config.json", "checkpoints", "fixed_attacker_eval"]):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def defended_batch(rows: list[dict[str, str]], cache: NpzCache, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), int(seq_length)), dtype=np.float32)
    lengths = np.zeros(len(rows), dtype=np.int64)
    for out_index, row in enumerate(rows):
        arrays = cache.get(row["defended_path"])
        local = int(row["defended_local_index"])
        start = int(arrays["offsets"][local])
        end = int(arrays["offsets"][local + 1])
        values = np.asarray(arrays["flat"][start:end], dtype=np.float32)
        take = min(int(seq_length), values.size)
        if take:
            x[out_index, :take] = values[:take]
        lengths[out_index] = min(int(seq_length), values.size)
    return x, lengths


def load_attack_model(kind: str, checkpoint: str | Path, device: torch.device, cfg: DTPNStyleConfig):
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    classes = np.asarray(payload["classes"], dtype=np.int64)
    model = make_attack_model(
        kind.upper(),
        len(classes),
        max_trace_length=int(cfg.max_trace_length),
        df_architecture="project",
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, classes, float(payload.get("best_val", 0.0))


def df_logits_and_embedding(df_model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not all(hasattr(df_model, name) for name in ["layer1", "layer2", "layer3", "layer4", "layer5", "fc"]):
        raise TypeError("DTPN-style embedding extraction expects ProjectDF with layer1..layer5/fc.")
    out = df_model.layer1(x)
    out = df_model.layer2(out)
    out = df_model.layer3(out)
    out = df_model.layer4(out)
    out = out.reshape(out.size(0), -1)
    embedding = df_model.layer5(out)
    return df_model.fc(embedding), embedding


def hard_df_input_from_norm(x_norm: torch.Tensor, cfg: DTPNStyleConfig) -> torch.Tensor:
    x = x_norm[:, : int(cfg.max_trace_length)]
    if x.shape[1] < int(cfg.max_trace_length):
        x = F.pad(x, (0, int(cfg.max_trace_length) - x.shape[1]))
    return torch.sign(x).unsqueeze(1)


def soft_df_input_from_norm(x_norm: torch.Tensor, cfg: DTPNStyleConfig) -> torch.Tensor:
    raw_units = x_norm[:, : int(cfg.max_trace_length)] * float(cfg.value_scale)
    if raw_units.shape[1] < int(cfg.max_trace_length):
        raw_units = F.pad(raw_units, (0, int(cfg.max_trace_length) - raw_units.shape[1]))
    direction = torch.tanh(raw_units / max(float(cfg.sign_temperature_seconds), 1.0e-6))
    return direction.unsqueeze(1)


def labels_to_positions(labels: torch.Tensor, classes: np.ndarray, device: torch.device) -> torch.Tensor:
    if np.array_equal(classes, np.arange(len(classes), dtype=np.int64)):
        return labels.to(device=device, dtype=torch.long)
    mapping = {int(label): pos for pos, label in enumerate(classes.tolist())}
    return torch.as_tensor([mapping[int(label)] for label in labels.detach().cpu().tolist()], dtype=torch.long, device=device)


@torch.no_grad()
def embedding_from_defended(df_model: nn.Module, x0_norm: torch.Tensor, cfg: DTPNStyleConfig) -> torch.Tensor:
    _, embedding = df_logits_and_embedding(df_model, hard_df_input_from_norm(x0_norm, cfg))
    return embedding.detach()


def tstar_index(cfg: DTPNStyleConfig) -> int:
    return max(0, min(int(cfg.diffusion_steps) - 1, int(round(float(cfg.t_star) * int(cfg.diffusion_steps)))))


def time_embedding_value(t: torch.Tensor, cfg: DTPNStyleConfig) -> torch.Tensor:
    return t.float().clamp(0.0, 1.0) * float(max(int(cfg.diffusion_steps) - 1, 1))


@torch.no_grad()
def purify_from_defended(
    model: DTPNStylePurifier,
    df_model: nn.Module,
    defended_norm: torch.Tensor,
    cfg: DTPNStyleConfig,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    embedding = embedding_from_defended(df_model, defended_norm, cfg)
    start_t = float(cfg.t_star)
    timesteps = torch.full((defended_norm.shape[0],), start_t, dtype=torch.float32, device=defended_norm.device)
    noise = torch.randn(defended_norm.shape, device=defended_norm.device, dtype=defended_norm.dtype, generator=generator)
    x, _ = model.diffusion.q_sample(defended_norm, timesteps, noise=noise)
    reverse_steps = max(1, tstar_index(cfg))
    times = torch.linspace(start_t, 0.0, reverse_steps + 1, device=defended_norm.device, dtype=torch.float32)
    for index, t_value in enumerate(times[:-1]):
        t = torch.full((x.shape[0],), float(t_value.item()), dtype=torch.float32, device=x.device)
        predicted_noise = model.predict_noise(x, time_embedding_value(t, cfg), embedding)
        x0 = model.diffusion.predict_x0_from_noise(x, t, predicted_noise)
        x0 = x0.clamp(-model.diffusion.x0_clip, model.diffusion.x0_clip)
        t_next = torch.full((x.shape[0],), float(times[index + 1].item()), dtype=torch.float32, device=x.device)
        mean_next = model.diffusion.mean_coeff(t_next).reshape(-1, *([1] * (x.ndim - 1)))
        std_next = model.diffusion.std(t_next).reshape(-1, *([1] * (x.ndim - 1)))
        x = mean_next * x0 + std_next * predicted_noise
    return x.clamp(-model.diffusion.x0_clip, model.diffusion.x0_clip)


def make_train_datasets(cfg: DTPNStyleConfig):
    base = PairManifestDataset(
        cfg.train_manifest,
        cfg.clean_path,
        expected_split="train",
        seq_length=int(cfg.seq_length),
        value_scale=float(cfg.value_scale),
        max_sources=int(cfg.max_train_sources),
        preload_shards=bool(cfg.preload_shards),
        max_open_shards=int(cfg.max_open_shards),
        condition_source="defended",
        include_clean=False,
    )
    return base, SourceBalancedPairDataset(base, seed=int(cfg.seed))


def make_validation_dataset(cfg: DTPNStyleConfig):
    return PairManifestDataset(
        cfg.validation_manifest,
        cfg.clean_path,
        expected_split="validation",
        seq_length=int(cfg.seq_length),
        value_scale=float(cfg.value_scale),
        max_sources=int(cfg.max_validation_sources),
        preload_shards=bool(cfg.preload_shards),
        max_open_shards=int(cfg.max_open_shards),
        condition_source="defended",
        include_clean=False,
    )


def train(cfg: DTPNStyleConfig, run_dir: Path, device: torch.device) -> dict[str, Any]:
    df_model, df_classes, df_best_val = load_attack_model("df", cfg.df_checkpoint, device, cfg)
    model = DTPNStylePurifier(cfg).to(device)
    train_base, train_dataset = make_train_datasets(cfg)
    validation_dataset = make_validation_dataset(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed) + 1000)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    class_meta = {"classes": df_classes.tolist(), "best_val_accuracy": df_best_val}
    write_json(run_dir / "df_condition_model_meta.json", class_meta)

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
        sums = {"loss": 0.0, "diffusion_loss": 0.0, "wf_loss": 0.0, "wf_accuracy": 0.0}
        count = 0
        for step, batch in enumerate(loader, start=1):
            x0 = batch["defended"].to(device=device, dtype=torch.float32, non_blocking=True)
            labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
            targets = labels_to_positions(labels, df_classes, device)
            with torch.no_grad():
                embedding = embedding_from_defended(df_model, x0, cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                bsz = int(x0.shape[0])
                t_cont = torch.rand((bsz,), device=device, dtype=torch.float32, generator=generator).clamp_min(1.0e-5)
                noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=generator)
                x_t, target_noise = model.diffusion.q_sample(x0, t_cont, noise=noise)
                predicted_noise = model.predict_noise(x_t, time_embedding_value(t_cont, cfg), embedding)
                predicted_x0 = model.diffusion.predict_x0_from_noise(x_t, t_cont, predicted_noise).clamp(
                    -model.diffusion.x0_clip,
                    model.diffusion.x0_clip,
                )
                diffusion_loss = F.mse_loss(predicted_noise, target_noise)
                wf_logits, _ = df_logits_and_embedding(df_model, soft_df_input_from_norm(predicted_x0, cfg))
                ce_per = F.cross_entropy(wf_logits.float(), targets, reduction="none")
                t_weight = 1.0 - t_cont.float()
                wf_loss = torch.mean(t_weight * ce_per)
                loss = diffusion_loss + float(cfg.tau) * wf_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gradient_clip))
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds = torch.argmax(wf_logits.detach(), dim=1)
                acc = torch.mean((preds == targets).float())
            batch_n = int(x0.shape[0])
            sums["loss"] += float(loss.detach().cpu()) * batch_n
            sums["diffusion_loss"] += float(diffusion_loss.detach().cpu()) * batch_n
            sums["wf_loss"] += float(wf_loss.detach().cpu()) * batch_n
            sums["wf_accuracy"] += float(acc.detach().cpu()) * batch_n
            count += batch_n
            if int(cfg.log_every) > 0 and (step == 1 or step % int(cfg.log_every) == 0 or step == len(loader)):
                print(
                    f"[dtpn-style train] epoch={epoch}/{cfg.epochs} step={step}/{len(loader)} "
                    f"loss={sums['loss']/max(count,1):.6f} diff={sums['diffusion_loss']/max(count,1):.6f} "
                    f"wf={sums['wf_loss']/max(count,1):.6f} wf_acc={sums['wf_accuracy']/max(count,1):.4f}",
                    flush=True,
                )
        train_metrics = {key: value / max(count, 1) for key, value in sums.items()}
        train_metrics["samples"] = int(count)
        train_metrics["source_balanced_epoch"] = train_dataset.epoch_stats()

        validation_metrics: dict[str, Any] = {}
        if epoch == int(cfg.epochs) or epoch % max(1, int(cfg.validate_every)) == 0:
            validation_metrics = validate(model, df_model, df_classes, validation_dataset, cfg, device)
            print(
                f"[dtpn-style validation] epoch={epoch} purified_df_acc={validation_metrics['purified_df_accuracy']:.6f} "
                f"diff={validation_metrics['diffusion_loss']:.6f} sign_change={validation_metrics['sign_change_rate']:.6f}",
                flush=True,
            )
            improved = best is None or validation_metrics["purified_df_accuracy"] > best["validation"]["purified_df_accuracy"]
            if not improved and best is not None and math.isclose(
                validation_metrics["purified_df_accuracy"],
                best["validation"]["purified_df_accuracy"],
            ):
                improved = validation_metrics["diffusion_loss"] < best["validation"]["diffusion_loss"]
            if improved:
                best = {"epoch": int(epoch), "validation": validation_metrics}
                save_checkpoint(checkpoints / "best_checkpoint.pt", model, cfg, epoch, validation_metrics)
        save_checkpoint(checkpoints / "last_checkpoint.pt", model, cfg, epoch, validation_metrics or train_metrics)
        history.append({"epoch": int(epoch), "train": train_metrics, "validation": validation_metrics})
        write_json(run_dir / "training_history.json", history)

    if best is None:
        raise RuntimeError("No DTPN-style checkpoint was selected")
    selection = {
        "selected_epoch": int(best["epoch"]),
        "selection_primary": "max validation purified DF accuracy",
        "selection_secondary": "min validation diffusion loss",
        "validation": best["validation"],
        "test_metric_used": False,
        "paper_aligned_parameters": {
            "x0": "defended/adversarial trace",
            "condition": "frozen DF penultimate-layer embedding",
            "N": int(cfg.diffusion_steps),
            "t_star": float(cfg.t_star),
            "beta_start": float(cfg.beta_start),
            "beta_end": float(cfg.beta_end),
            "tau": float(cfg.tau),
            "learning_rate": float(cfg.learning_rate),
        },
    }
    write_json(run_dir / "checkpoint_selection.json", selection)
    return {
        "checkpoint": str((checkpoints / "best_checkpoint.pt").resolve()),
        "selected_epoch": int(best["epoch"]),
        "checkpoint_selection": selection,
    }


def save_checkpoint(path: Path, model: DTPNStylePurifier, cfg: DTPNStyleConfig, epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "epoch": int(epoch),
            "metrics": metrics,
            "representation": "fixed_length_signed_timestamp_v1_project_adapted_dtpn_style",
            "x0_semantics": "defended/adversarial traffic",
            "condition_semantics": "frozen DF penultimate embedding of x0",
            "sampling_semantics": "start from defended trace noised to t_star, then reverse to t=0",
        },
        path,
    )


def load_dtpn_checkpoint(path: str | Path, device: torch.device) -> tuple[DTPNStylePurifier, DTPNStyleConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = DTPNStyleConfig(**payload["config"])
    model = DTPNStylePurifier(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, payload


@torch.no_grad()
def validate(
    model: DTPNStylePurifier,
    df_model: nn.Module,
    df_classes: np.ndarray,
    dataset: PairManifestDataset,
    cfg: DTPNStyleConfig,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed) + 5000)
    model.eval()
    diff_sum = 0.0
    count = 0
    correct = 0
    defended_correct = 0
    sign_changes = 0
    sign_total = 0
    for batch in loader:
        x0 = batch["defended"].to(device=device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
        targets = labels_to_positions(labels, df_classes, device)
        embedding = embedding_from_defended(df_model, x0, cfg)
        t_cont = torch.rand((x0.shape[0],), device=device, dtype=torch.float32, generator=generator).clamp_min(1.0e-5)
        noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=generator)
        x_t, target_noise = model.diffusion.q_sample(x0, t_cont, noise=noise)
        predicted_noise = model.predict_noise(x_t, time_embedding_value(t_cont, cfg), embedding)
        diff_sum += float(F.mse_loss(predicted_noise, target_noise, reduction="sum").detach().cpu())
        purified_norm = purify_from_defended(model, df_model, x0, cfg, generator=generator)
        logits, _ = df_logits_and_embedding(df_model, hard_df_input_from_norm(purified_norm, cfg))
        defended_logits, _ = df_logits_and_embedding(df_model, hard_df_input_from_norm(x0, cfg))
        correct += int((torch.argmax(logits, dim=1) == targets).sum().detach().cpu())
        defended_correct += int((torch.argmax(defended_logits, dim=1) == targets).sum().detach().cpu())
        src_sign = torch.sign(x0)
        dst_sign = torch.sign(purified_norm)
        active = src_sign != 0
        sign_changes += int(((src_sign != dst_sign) & active).sum().detach().cpu())
        sign_total += int(active.sum().detach().cpu())
        count += int(x0.shape[0])
    return {
        "count": int(count),
        "defended_df_accuracy": float(defended_correct / max(count, 1)),
        "purified_df_accuracy": float(correct / max(count, 1)),
        "diffusion_loss": float(diff_sum / max(count * int(cfg.seq_length), 1)),
        "sign_change_rate": float(sign_changes / max(sign_total, 1)),
        "t_star_index": int(tstar_index(cfg)),
    }


def generate_purified_test(
    model: DTPNStylePurifier,
    df_model: nn.Module,
    cfg: DTPNStyleConfig,
    run_dir: Path,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, Any]:
    test_rows = [row for row in read_csv_rows(cfg.test_manifest) if row["split"] == "test"]
    if int(cfg.max_test_pairs) > 0:
        test_rows = test_rows[: int(cfg.max_test_pairs)]
    output_dir = run_dir / "purified_datasets" / "test"
    manifest_path = run_dir / "manifests" / "dtpn_style_purified_test_manifest.csv"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache = NpzCache(max_items=4)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed) + 7000)
    columns = [
        "source_id",
        "clean_index",
        "defended_index",
        "defended_local_index",
        "variant_id",
        "split",
        "class_id",
        "defended_path",
        "defended_length",
        "purified_path",
        "purified_index",
        "purifier_checkpoint",
        "diffusion_steps",
        "t_star",
        "t_star_index",
        "sampling_seed",
        "representation",
        "legalization_version",
        "output_length_policy",
        "output_length",
    ]
    written = 0
    shard_index = 0
    shard_arrays: list[np.ndarray] = []
    shard_rows: list[dict[str, Any]] = []

    def flush(writer: csv.DictWriter) -> None:
        nonlocal written, shard_index, shard_arrays, shard_rows
        if not shard_rows:
            return
        x = np.concatenate(shard_arrays, axis=0).astype(np.float32, copy=False)
        path = output_dir / f"dtpn_style_test_shard{shard_index:05d}.npz"
        np.savez_compressed(
            path,
            X=x,
            y=np.asarray([row["class_id"] for row in shard_rows], dtype=np.int64),
            source_id=np.asarray([row["source_id"] for row in shard_rows], dtype=np.int64),
            clean_index=np.asarray([row["clean_index"] for row in shard_rows], dtype=np.int64),
            defended_index=np.asarray([row["defended_index"] for row in shard_rows], dtype=np.int64),
            defended_local_index=np.asarray([row["defended_local_index"] for row in shard_rows], dtype=np.int64),
            variant_id=np.asarray([row["variant_id"] for row in shard_rows], dtype=np.int64),
            defended_length=np.asarray([row["defended_length"] for row in shard_rows], dtype=np.int64),
            output_length=np.asarray([row["output_length"] for row in shard_rows], dtype=np.int64),
        )
        for local, row in enumerate(shard_rows):
            payload = dict(row)
            payload.update(
                {
                    "purified_path": str(path.resolve()),
                    "purified_index": int(local),
                    "purifier_checkpoint": str(checkpoint.resolve()),
                    "diffusion_steps": int(cfg.diffusion_steps),
                    "t_star": float(cfg.t_star),
                    "t_star_index": int(tstar_index(cfg)),
                    "sampling_seed": int(cfg.seed) + 7000,
                    "representation": "fixed_length_signed_timestamp_v1_project_adapted_dtpn_style",
                    "legalization_version": model.decoder.legalization_version,
                    "output_length_policy": "defended",
                }
            )
            writer.writerow(payload)
        written += len(shard_rows)
        print(f"[dtpn-style generate] shard={shard_index} rows={len(shard_rows)} total={written}", flush=True)
        shard_index += 1
        shard_arrays = []
        shard_rows = []

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for start in range(0, len(test_rows), int(cfg.batch_size)):
            batch_rows = test_rows[start : start + int(cfg.batch_size)]
            raw_units, lengths = defended_batch(batch_rows, cache, int(cfg.seq_length))
            defended_norm = torch.as_tensor(raw_units / float(cfg.value_scale), dtype=torch.float32, device=device)
            purified_norm = purify_from_defended(model, df_model, defended_norm, cfg, generator=generator)
            output_lengths = torch.as_tensor(lengths, dtype=torch.long)
            purified_units = model.decoder.legalize_numpy(purified_norm, output_length=output_lengths)
            shard_arrays.append(purified_units)
            for row, output_len in zip(batch_rows, lengths.tolist()):
                shard_rows.append(
                    {
                        "source_id": int(row["source_id"]),
                        "clean_index": int(row["clean_index"]),
                        "defended_index": int(row.get("defended_global_index") or row.get("defended_index") or 0),
                        "defended_local_index": int(row["defended_local_index"]),
                        "variant_id": int(row["variant_id"]),
                        "split": "test",
                        "class_id": int(row["class_id"]),
                        "defended_path": row["defended_path"],
                        "defended_length": int(output_len),
                        "output_length": int(output_len),
                    }
                )
            if len(shard_rows) >= int(cfg.shard_size):
                flush(writer)
            if start == 0 or (start // int(cfg.batch_size) + 1) % 50 == 0:
                print(f"[dtpn-style generate] batch={start // int(cfg.batch_size) + 1}/{math.ceil(len(test_rows)/int(cfg.batch_size))}", flush=True)
        flush(writer)
    summary = {
        "manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "test_pairs": int(len(test_rows)),
        "shards": int(shard_index),
        "output_length_policy": "defended",
    }
    write_json(run_dir / "purified_generation_summary.json", summary)
    return summary


def attack_input(kind: str, raw: np.ndarray, cfg: DTPNStyleConfig) -> np.ndarray:
    if kind == "DF":
        return build_df_input(raw, max_len=int(cfg.max_trace_length)).astype(np.float32)
    return build_rf_tam_input(
        raw,
        max_len=int(cfg.max_trace_length),
        max_load_time=float(cfg.max_load_time),
        num_slots=int(cfg.rf_tam_num_slots),
    ).astype(np.float32)


def eval_named_batches(
    name: str,
    batch_iter,
    attackers: dict[str, tuple[nn.Module, np.ndarray]],
    device: torch.device,
    cfg: DTPNStyleConfig,
) -> dict[str, dict[str, Any]]:
    accumulators = {kind: MetricAccumulator(classes) for kind, (_, classes) in attackers.items()}
    with torch.no_grad():
        for batch_index, (raw, labels) in enumerate(batch_iter, start=1):
            labels_np = np.asarray(labels, dtype=np.int64)
            for kind, (model, _) in attackers.items():
                x = attack_input(kind, raw, cfg)
                logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
                accumulators[kind].update(logits, labels_np)
            if batch_index == 1 or batch_index % 50 == 0:
                print(f"[dtpn-style fixed eval] {name}: batch={batch_index}", flush=True)
    return {kind: acc.finalize() for kind, acc in accumulators.items()}


def clean_iter(rows: list[dict[str, str]], clean: CleanStore, batch_size: int, cfg: DTPNStyleConfig):
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        yield clean.batch([int(row["clean_index"]) for row in batch], int(cfg.seq_length)), np.asarray(
            [int(row["class_id"]) for row in batch],
            dtype=np.int64,
        )


def defended_iter(rows: list[dict[str, str]], cache: NpzCache, batch_size: int, cfg: DTPNStyleConfig):
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        raw, _ = defended_batch(batch, cache, int(cfg.seq_length))
        yield raw, np.asarray([int(row["class_id"]) for row in batch], dtype=np.int64)


def purified_iter(rows: list[dict[str, str]], cache: NpzCache, batch_size: int):
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        arrays = []
        for row in batch:
            payload = cache.get(row["purified_path"])
            arrays.append(np.asarray(payload["X"][int(row["purified_index"])], dtype=np.float32))
        yield np.stack(arrays, axis=0), np.asarray([int(row["class_id"]) for row in batch], dtype=np.int64)


def recovery(clean_acc: float, defended_acc: float, purified_acc: float) -> float | None:
    denom = float(clean_acc) - float(defended_acc)
    if abs(denom) < 1.0e-8:
        return None
    return float((purified_acc - defended_acc) / denom)


def evaluate_fixed_attackers(cfg: DTPNStyleConfig, run_dir: Path, device: torch.device, purified_manifest: Path) -> dict[str, Any]:
    test_rows = [row for row in read_csv_rows(cfg.test_manifest) if row["split"] == "test"]
    if int(cfg.max_test_pairs) > 0:
        test_rows = test_rows[: int(cfg.max_test_pairs)]
    purified_rows = [row for row in read_csv_rows(purified_manifest) if row["split"] == "test"]
    if len(purified_rows) != len(test_rows):
        raise RuntimeError(f"Purified rows mismatch: {len(purified_rows)} != {len(test_rows)}")
    attackers = {}
    attacker_meta = {}
    for kind, checkpoint in {"DF": cfg.df_checkpoint, "RF": cfg.rf_checkpoint}.items():
        model, classes, best_val = load_attack_model(kind, checkpoint, device, cfg)
        attackers[kind] = (model, classes)
        attacker_meta[kind] = {"checkpoint": str(Path(checkpoint).resolve()), "classes": int(len(classes)), "best_val_accuracy": best_val}
    clean = CleanStore(cfg.clean_path)
    defended_cache = NpzCache(max_items=4)
    purified_cache = NpzCache(max_items=8)
    try:
        main = {
            "clean_test_pair_expanded": eval_named_batches("clean_test_pair_expanded", clean_iter(test_rows, clean, int(cfg.batch_size), cfg), attackers, device, cfg),
            "defended_test": eval_named_batches("defended_test", defended_iter(test_rows, defended_cache, int(cfg.batch_size), cfg), attackers, device, cfg),
            "dtpn_style_purified_test": eval_named_batches("dtpn_style_purified_test", purified_iter(purified_rows, purified_cache, int(cfg.batch_size)), attackers, device, cfg),
        }
    finally:
        clean.close()
    recovery_rows = {}
    for kind in ["DF", "RF"]:
        clean_acc = float(main["clean_test_pair_expanded"][kind]["accuracy"])
        defended_acc = float(main["defended_test"][kind]["accuracy"])
        purified_acc = float(main["dtpn_style_purified_test"][kind]["accuracy"])
        recovery_rows[kind] = {
            "clean_accuracy": clean_acc,
            "defended_accuracy": defended_acc,
            "purified_accuracy": purified_acc,
            "purified_minus_defended": purified_acc - defended_acc,
            "accuracy_recovery_ratio": recovery(clean_acc, defended_acc, purified_acc),
            "purified_beats_defended": purified_acc > defended_acc,
        }
    report = {
        "protocol": "DTPN-style fixed attacker evaluation",
        "inference_only": True,
        "test_pair_count": int(len(test_rows)),
        "test_source_count": int(len({int(row["source_id"]) for row in test_rows})),
        "attacker_meta": attacker_meta,
        "main": main,
        "recovery": recovery_rows,
    }
    out_dir = run_dir / "fixed_attacker_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "fixed_attacker_recovery_report.json", report)
    rows = []
    for name, metrics in main.items():
        row = {"input": name}
        for kind in ["DF", "RF"]:
            row[f"{kind.lower()}_accuracy"] = metrics[kind]["accuracy"]
            row[f"{kind.lower()}_macro_f1"] = metrics[kind]["macro_f1"]
        rows.append(row)
    write_csv(out_dir / "fixed_attacker_recovery_main.csv", rows)
    write_recovery_md(out_dir / "fixed_attacker_recovery_report.md", report)
    return report


def write_recovery_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DTPN-style Frozen DF/RF Evaluation",
        "",
        f"- test pairs: `{report['test_pair_count']}`",
        f"- test sources: `{report['test_source_count']}`",
        "",
        "| Input | DF Acc | DF Macro-F1 | RF Acc | RF Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["clean_test_pair_expanded", "defended_test", "dtpn_style_purified_test"]:
        row = report["main"][name]
        lines.append(
            f"| {name} | {row['DF']['accuracy']:.6f} | {row['DF']['macro_f1']:.6f} | "
            f"{row['RF']['accuracy']:.6f} | {row['RF']['macro_f1']:.6f} |"
        )
    lines.extend(["", "| Attacker | Purified - Defended | Recovery Ratio |", "|---|---:|---:|"])
    for kind, row in report["recovery"].items():
        ratio = row["accuracy_recovery_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        lines.append(f"| {kind} | {row['purified_minus_defended']:.6f} | {ratio_text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def trace_stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    nonzero = x[x != 0]
    return {
        "total": int(nonzero.size),
        "outgoing": int(np.sum(nonzero > 0)),
        "incoming": int(np.sum(nonzero < 0)),
        "max_time": float(np.max(np.abs(nonzero))) if nonzero.size else 0.0,
    }


def tam_counts(values: np.ndarray, *, num_slots: int, slot_ms: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    x = x[x != 0]
    out_counts = np.zeros(int(num_slots), dtype=np.float32)
    in_counts = np.zeros(int(num_slots), dtype=np.float32)
    if x.size == 0:
        return out_counts, in_counts
    slot_seconds = float(slot_ms) / 1000.0
    slots = np.floor(np.abs(x) / max(slot_seconds, 1.0e-12)).astype(np.int64)
    slots = np.clip(slots, 0, int(num_slots) - 1)
    np.add.at(out_counts, slots[x > 0], 1.0)
    np.add.at(in_counts, slots[x < 0], 1.0)
    return out_counts, -in_counts


def setup_chinese_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    return None


def plot_single_trace(cfg: DTPNStyleConfig, run_dir: Path, purified_manifest: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    font_prop = setup_chinese_font()
    rows = [row for row in read_csv_rows(purified_manifest) if row["split"] == "test"]
    row = rows[int(cfg.plot_row_index)]
    clean = CleanStore(cfg.clean_path)
    cache = NpzCache(max_items=4)
    try:
        clean_values = clean.batch([int(row["clean_index"])], int(cfg.seq_length))[0]
        defended_values, _ = defended_batch([row], cache, int(cfg.seq_length))
        defended_values = defended_values[0]
        purified_values = next(purified_iter([row], cache, 1))[0][0]
    finally:
        clean.close()
    traces = [
        ("干净流量 Clean", clean_values),
        ("防御后流量 Defended", defended_values),
        ("DTPN-style 净化后流量", purified_values),
    ]
    fig, axes = plt.subplots(len(traces), 1, figsize=(11, 7.8), sharex=True)
    axes = np.atleast_1d(axes)
    x_axis = np.arange(int(cfg.tam_num_slots))
    ranges: list[tuple[float, float]] = []
    summaries = {}
    blue = "#0000ff"
    red = "#ff0000"
    for ax, (title, values) in zip(axes, traces):
        outgoing, incoming = tam_counts(values, num_slots=int(cfg.tam_num_slots), slot_ms=float(cfg.tam_slot_ms))
        stats = trace_stats(values)
        summaries[title] = stats
        kwargs = {"fontproperties": font_prop} if font_prop is not None else {}
        ax.bar(x_axis, outgoing, width=1.0, color=blue, edgecolor=blue, linewidth=0.0)
        ax.bar(x_axis, incoming, width=1.0, color=red, edgecolor=red, linewidth=0.0)
        ax.axhline(0, color="#d9a8c4", linewidth=1.0, alpha=0.9)
        ax.grid(True, axis="both", alpha=0.28, linewidth=0.7)
        ax.set_ylabel("包数量", fontsize=10, **kwargs)
        ax.set_title(f"{title}  (上行={stats['outgoing']}, 下行={stats['incoming']}, 总数={stats['total']})", fontsize=12, **kwargs)
        ranges.append((float(np.min(incoming)), float(np.max(outgoing))))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            if font_prop is not None:
                label.set_fontproperties(font_prop)
    y_min = min(item[0] for item in ranges)
    y_max = max(item[1] for item in ranges)
    margin = max(2.0, 0.08 * max(abs(y_min), abs(y_max), 1.0))
    for ax in axes:
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlim(0, int(cfg.tam_num_slots))
    axes[-1].set_xlabel("时间槽", fontsize=10, fontproperties=font_prop)
    legend_kwargs = {"prop": font_prop} if font_prop is not None else {}
    axes[0].legend(
        handles=[Patch(color=blue, label="上行包"), Patch(color=red, label="下行包")],
        loc="upper right",
        frameon=True,
        **legend_kwargs,
    )
    title_kwargs = {"fontproperties": font_prop} if font_prop is not None else {}
    fig.suptitle(
        f"TAM 单 trace 对比：source_id={row['source_id']}，variant={row['variant_id']}，class={row['class_id']}",
        fontsize=13,
        **title_kwargs,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir = run_dir / "visualizations" / "tam"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"dtpn_style_tam_row{int(cfg.plot_row_index):05d}_zh_shared.png"
    summary_path = out_dir / f"dtpn_style_tam_row{int(cfg.plot_row_index):05d}_zh_shared_summary.json"
    fig.savefig(image_path, dpi=220)
    plt.close(fig)
    summary = {
        "image": str(image_path.resolve()),
        "row_index": int(cfg.plot_row_index),
        "source_id": int(row["source_id"]),
        "variant_id": int(row["variant_id"]),
        "class_id": int(row["class_id"]),
        "clean_index": int(row["clean_index"]),
        "defended_path": row["defended_path"],
        "purified_path": row["purified_path"],
        "num_slots": int(cfg.tam_num_slots),
        "slot_ms": float(cfg.tam_slot_ms),
        "summaries": summaries,
    }
    write_json(summary_path, summary)
    return summary


def write_experiment_md(run_dir: Path, cfg: DTPNStyleConfig, train_summary: dict[str, Any], generation: dict[str, Any], evaluation: dict[str, Any], plot: dict[str, Any]) -> None:
    lines = [
        "# DTPN-style Purifier Experiment",
        "",
        "## Protocol",
        "",
        "- `x0`: defended/adversarial trace from the audited clean-defended manifest.",
        "- `c`: frozen DF penultimate-layer embedding of `x0`.",
        "- `x_t`: noised `x0` under the diffusion forward process.",
        "- Sampling: start from defended trace noised to `t*`, then reverse to `t=0`.",
        "- Evaluation: send clean / defended / purified traffic to frozen DF and RF.",
        "",
        "## Paths",
        "",
        f"- defense run: `{cfg.defense_run_dir}`",
        f"- clean data: `{cfg.clean_path}`",
        f"- train manifest: `{cfg.train_manifest}`",
        f"- validation manifest: `{cfg.validation_manifest}`",
        f"- test manifest: `{cfg.test_manifest}`",
        f"- DF checkpoint: `{cfg.df_checkpoint}`",
        f"- RF checkpoint: `{cfg.rf_checkpoint}`",
        f"- DTPN-style checkpoint: `{train_summary.get('checkpoint', '')}`",
        f"- purified manifest: `{generation.get('manifest', '')}`",
        f"- single-trace plot: `{plot.get('image', '')}`",
        "",
        "## Hyperparameters",
        "",
        f"- representation length: `{cfg.seq_length}` signed timestamp positions",
        f"- diffusion steps `N`: `{cfg.diffusion_steps}`",
        f"- `t*`: `{cfg.t_star}`; discrete index `{tstar_index(cfg)}`",
        f"- `beta_start/beta_end`: `{cfg.beta_start}` / `{cfg.beta_end}`",
        f"- WF guidance `tau`: `{cfg.tau}`",
        f"- learning rate: `{cfg.learning_rate}`",
        f"- epochs: `{cfg.epochs}`",
        f"- batch size: `{cfg.batch_size}`",
        f"- train max sources: `{cfg.max_train_sources}`",
        f"- validation max sources: `{cfg.max_validation_sources}`",
        "",
        "## Frozen Attacker Results",
        "",
        "| Input | DF Acc | DF Macro-F1 | RF Acc | RF Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["clean_test_pair_expanded", "defended_test", "dtpn_style_purified_test"]:
        row = evaluation["main"][name]
        lines.append(
            f"| {name} | {row['DF']['accuracy']:.6f} | {row['DF']['macro_f1']:.6f} | "
            f"{row['RF']['accuracy']:.6f} | {row['RF']['macro_f1']:.6f} |"
        )
    lines.extend(["", "| Attacker | Purified - Defended | Recovery Ratio |", "|---|---:|---:|"])
    for kind, row in evaluation["recovery"].items():
        ratio = row["accuracy_recovery_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        lines.append(f"| {kind} | {row['purified_minus_defended']:.6f} | {ratio_text} |")
    lines.extend(
        [
            "",
            "## Compatibility Note",
            "",
            "The original DTPN paper uses burst sequences of length 512. This run keeps the project signed timestamp format so the existing frozen DF/RF checkpoints can be used without retraining.",
        ]
    )
    (run_dir / "dtpn_style_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = cfg_from_args(args)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    run_dir = run_dir_for(cfg, overwrite=bool(args.overwrite))
    write_json(run_dir / "config.json", asdict(cfg))
    print(json.dumps({"run_dir": str(run_dir.resolve()), "device": str(device), "config": asdict(cfg)}, indent=2, ensure_ascii=False), flush=True)

    checkpoint_path = Path(args.checkpoint).resolve() if str(args.checkpoint).strip() else run_dir / "checkpoints" / "best_checkpoint.pt"
    train_summary: dict[str, Any] = {}
    if not bool(args.skip_training):
        train_summary = train(cfg, run_dir, device)
        checkpoint_path = Path(train_summary["checkpoint"])
        write_json(run_dir / "training_summary.json", train_summary)
    else:
        train_summary = {"checkpoint": str(checkpoint_path.resolve()), "selected_epoch": None, "skipped": True}

    model, cfg_from_ckpt, _ = load_dtpn_checkpoint(checkpoint_path, device)
    cfg = cfg_from_ckpt
    df_model, _, _ = load_attack_model("df", cfg.df_checkpoint, device, cfg)
    generation: dict[str, Any] = {}
    if not bool(args.skip_generate):
        generation = generate_purified_test(model, df_model, cfg, run_dir, device, checkpoint_path)
    else:
        generation = {"manifest": str((run_dir / "manifests" / "dtpn_style_purified_test_manifest.csv").resolve()), "skipped": True}

    evaluation: dict[str, Any] = {}
    if not bool(args.skip_eval):
        evaluation = evaluate_fixed_attackers(cfg, run_dir, device, Path(generation["manifest"]))
    else:
        evaluation = {"skipped": True}

    plot: dict[str, Any] = {}
    if not bool(args.skip_plot):
        plot = plot_single_trace(cfg, run_dir, Path(generation["manifest"]))
        print(json.dumps({"plot": plot}, indent=2, ensure_ascii=False), flush=True)
    else:
        plot = {"skipped": True}

    if evaluation and not evaluation.get("skipped"):
        write_experiment_md(run_dir, cfg, train_summary, generation, evaluation, plot)
    summary = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "generation": generation,
        "evaluation": evaluation,
        "plot": plot,
        "report_md": str((run_dir / "dtpn_style_experiment_report.md").resolve()),
    }
    write_json(run_dir / "dtpn_style_experiment_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
