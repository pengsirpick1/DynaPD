"""Train the optional Stage A TAM mask predictor from teacher masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.student import TamMaskPredictor
from dmmp.utils import resolve_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bce_weight", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.archive, allow_pickle=False) as arrays:
        tam = torch.as_tensor(arrays["tam"], dtype=torch.float32)
        masks = torch.as_tensor(arrays["mask"], dtype=torch.float32)
    model = TamMaskPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    rng = np.random.default_rng(int(args.seed))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        order = rng.permutation(tam.shape[0])
        losses = []
        for start in range(0, len(order), int(args.batch_size)):
            idx = order[start : start + int(args.batch_size)]
            xb = tam[idx].to(device)
            yb = masks[idx].to(device)
            pred = model(xb)
            loss = F.l1_loss(pred, yb) + float(args.bce_weight) * F.binary_cross_entropy(pred.clamp(1e-5, 1 - 1e-5), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0}
        history.append(row)
        print(f"[Stage A predictor] epoch={epoch}/{args.epochs} loss={row['loss']:.6f}", flush=True)
    torch.save({"model_state": model.state_dict(), "history": history}, out_dir / "mask_predictor.pt")
    write_json(out_dir / "training_summary.json", {"history": history, "samples": int(tam.shape[0])})
    print(f"Stage A mask predictor complete: {out_dir}")


if __name__ == "__main__":
    main()
