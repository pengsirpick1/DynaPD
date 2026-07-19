"""Fast architecture/gradient validation; this does not train or evaluate an experiment."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.evaluation.attack_models import make_attack_model
from dmmp.guidance.strong_surrogates import StrongSurrogateBundle, build_attack_context, ensemble_utility


def main() -> None:
    cfg = SimpleNamespace(
        max_trace_length=5000,
        patch_num=200,
        surrogate_rf_max_load_time=80.0,
        surrogate_rf_num_slots=1800,
    )
    raw = np.zeros((2, 5000), dtype=np.float32)
    timestamps = np.linspace(0.01, 1.0, 100, dtype=np.float32)
    directions = np.where(np.arange(100) % 2 == 0, 1.0, -1.0).astype(np.float32)
    raw[:, :100] = timestamps * directions
    models = {
        "df": make_attack_model("DF", 3, max_trace_length=5000).eval(),
        "rf": make_attack_model("RF", 3).eval(),
    }
    for model in models.values():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    bundle = StrongSurrogateBundle(models, np.arange(3), {"df": 0.5, "rf": 0.5}, 200, 5000, 1800, 80.0)
    context = build_attack_context(raw, cfg, torch.device("cpu"))
    allocation = torch.zeros((2, 2, 200), dtype=torch.float32, requires_grad=True)
    logits = bundle.logits_from_allocation(allocation, context)
    utility, _ = ensemble_utility(logits, torch.tensor([0, 1]), bundle.weights, 0.35)
    utility.mean().backward()
    gradient_sum = float(allocation.grad.abs().sum())
    assert set(logits) == {"df", "rf"}
    assert all(tuple(values.shape) == (2, 3) for values in logits.values())
    assert torch.isfinite(allocation.grad).all() and gradient_sum > 0.0
    print(f"PASS: DF/RF logits are connected to defense allocation; gradient_abs_sum={gradient_sum:.6f}")


if __name__ == "__main__":
    main()

