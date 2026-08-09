from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / "wflib_copy").exists():
    sys.path.insert(0, str(ROOT / "wflib_copy"))

from dynapd.data import load_cw_data
from dynapd.utils import resolve_device
from WFlib import models as wflib_models


def load_state(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {str(k).replace("module.", ""): v for k, v in state.items()}


def infer_classes(state: dict) -> int:
    if "mlp.weight" in state:
        return int(state["mlp.weight"].shape[0])
    for key, value in state.items():
        if key.endswith("weight") and getattr(value, "ndim", 0) == 2:
            if int(value.shape[0]) == 100:
                return 100
    for key, value in state.items():
        if getattr(value, "ndim", 0) == 1 and int(value.shape[0]) == 100:
            return 100
    raise RuntimeError("Cannot infer class count")


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    preds = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        logits = model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def length_align(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, 0, :]
    out = np.zeros((x.shape[0], length), dtype=np.float32)
    n = min(length, x.shape[-1])
    out[:, :n] = x[:, :n]
    return out


def make_dt2_from_raw(raw: np.ndarray, length: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[:, 0, :]
    x_dir = np.sign(raw)
    x_time = np.diff(np.abs(raw), axis=1)
    x_time[x_time < 0] = 0
    x_dir = length_align(x_dir, length)[:, None, :]
    x_time = length_align(x_time, length)[:, None, :]
    return np.concatenate([x_dir, x_time], axis=1).astype(np.float32)


def make_dt2_from_exported_sign(x: np.ndarray, length: int) -> np.ndarray:
    # The current defended NPZ stores sign-only traces {-1,0,1}; original packet
    # timestamps are not recoverable. Use a zero temporal channel so VarCNN can run.
    x_dir = np.sign(length_align(x, length))[:, None, :]
    x_time = np.zeros_like(x_dir, dtype=np.float32)
    return np.concatenate([x_dir, x_time], axis=1).astype(np.float32)


def main() -> None:
    device = resolve_device("cuda")
    defended = np.load(ROOT / "wflib_copy/datasets/CW/adapt_e2b_completion_fulltest_b030_e05_seed202_merged.npz", allow_pickle=False)
    x_def = np.asarray(defended["X"], dtype=np.float32)
    y = np.asarray(defended["y"], dtype=np.int64)
    source_indices = np.asarray(defended["source_indices"], dtype=np.int64)

    cfg = SimpleNamespace(data_root=str(ROOT / "datasets/CW.npz"), seed=0, val_ratio=0.10, test_ratio=0.10, max_samples=0, max_classes=0)
    raw, labels, *_ = load_cw_data(cfg)
    labels = np.asarray(labels, dtype=np.int64)[source_indices]
    if not np.array_equal(labels, y):
        raise RuntimeError("Label mismatch")
    clean = make_dt2_from_raw(np.asarray(raw, dtype=np.float32)[source_indices], 5000)

    state = load_state(ROOT / "wflib_copy/checkpoints/CW/VarCNN/dynapd_clean_seed0.pth", device)
    classes = infer_classes(state)
    model = wflib_models.VarCNN(classes).to(device)
    model.load_state_dict(state)
    model.eval()

    clean_pred = predict(model, clean, device)
    x_def_varcnn = make_dt2_from_exported_sign(x_def, 5000)
    def_pred = predict(model, x_def_varcnn, device)
    result = {
        "samples": int(y.shape[0]),
        "varcnn_clean_accuracy": float(np.mean(clean_pred == y)),
        "varcnn_defended_accuracy": float(np.mean(def_pred == y)),
        "varcnn_flip_rate": float(np.mean(def_pred != y)),
        "defended_feature_note": "defended NPZ is sign-only; VarCNN time channel set to zeros",
    }
    out = ROOT / "results/stage_b_e2b_completion_fulltest_b030_e05_seed202_varcnn_eval.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
