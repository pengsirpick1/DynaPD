"""Random online baselines for DynaPD-RT bandwidth comparison.

The baselines keep the same renderer/evaluation stack as DynaPD-RT and remain
causal: decisions are triggered at observed positive-direction burst endings.

Variants:
  - random_recent: same dose rule as DynaPD-RT, but dummy position is jittered
    after the current burst tail and delay window is a random recent window.
  - random_far: same dose rule, dummy position is sampled from the future part
    of the 80s TAM horizon after the decision bin; delay remains recent-only.

Both variants are intentionally "randomized online" baselines, not offline
post-hoc random insertion.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wflib_copy"))

import streaming_state_machine as sm
from dynapd.evaluation.attack_models import build_rf_tam_input
from dynapd.stage_a.faithfulness import predict_probabilities
from dynapd.stage_a.modeling import load_stage_a_attacker
from scripts.stage_b_run_dual_actuator import _render_dummy
from scripts.stage_b_run_ensemble_oracle_e2b_completion import _predict_wflib
from WFlib import models as wm

OUT_DIR = ROOT / "results/night_batch_20260805/random_streaming_baseline_bw_sweep"
BINS = 1800
GAP_THRESH = 4


def _make_args(seed: int = 0):
    from types import SimpleNamespace

    return SimpleNamespace(
        rf_num_slots=1800,
        max_trace_length=sm.SL,
        max_delay=0,
        rounds=8,
        delay_length=64,
        delay_rho=1.0,
        max_load_time=80.0,
        algorithm="priority",
        seed=seed,
        renderer_strategy="priority",
        renderer_coordinate="absolute",
        ratio=0.10,
        max_windows=8,
    )


def _scalar(value: object) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _extract_all_packets(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz = np.flatnonzero(trace != 0)
    if len(nz) == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int32), np.zeros(0)
    times = np.abs(trace[nz])
    slots = np.floor(times * (1799.0 / 80.0)).astype(np.int32)
    slots = np.clip(slots, 0, 1799)
    dirs = np.sign(trace[nz])
    order = np.argsort(slots)
    return times[order], slots[order], dirs[order]


def _phase_of_bin(bin_idx: int) -> str:
    if bin_idx < 600:
        return "early"
    if bin_idx < 1200:
        return "mid"
    return "late"


def _best_dose_from_rt(phase: str, budget_left: int) -> int:
    return int(sm.best_dose(phase, budget_left))


def defend_random_stream(trace: np.ndarray, seed: int, rho: float, mode: str) -> tuple[np.ndarray, dict]:
    """Causal random streaming defense.

    Budget is accumulated over all observed real packets, while actions are
    triggered by positive-direction/download burst endings, matching DynaPD-RT.
    """
    rng = np.random.default_rng(seed)
    trace = np.asarray(trace, dtype=np.float32)[: sm.SL]
    _times, slots, dirs = _extract_all_packets(trace)

    obs_all = 0
    used_dummy = 0
    cur_end: int | None = None
    burst_id = 0
    n_bursts = 0
    injections: list[tuple[int, int]] = []
    delay_windows: list[tuple[int, int, int]] = []
    audit = {"delay_past_packet": 0, "delay_future_window": 0}

    def close_burst() -> None:
        nonlocal used_dummy, burst_id, n_bursts
        if cur_end is None:
            return
        n_bursts += 1
        decision_bin = int(cur_end)
        token = rho * obs_all - used_dummy
        if token <= 0:
            burst_id += 1
            return
        phase = _phase_of_bin(decision_bin)
        seen = burst_id + 1
        per_burst = max(1, int(rho * obs_all / seen))
        dose = _best_dose_from_rt(phase, int(token))
        dose = max(dose, min(int(per_burst * 0.7), int(token)))
        dose = min(dose, int(token))
        if dose <= 0:
            burst_id += 1
            return

        if mode == "random_recent":
            dummy_bin = min(BINS - 1, decision_bin + int(rng.integers(0, 65)))
        elif mode == "random_far":
            dummy_bin = int(rng.integers(decision_bin, BINS)) if decision_bin < BINS - 1 else decision_bin
        else:
            raise ValueError(f"unknown mode: {mode}")

        injections.append((dummy_bin, dose))
        used_dummy += dose

        # Delay remains causal: choose a recent window ending no later than the
        # decision bin, so only already observed packets can be affected.
        if decision_bin > 0:
            recent_span = min(128, decision_bin)
            center = int(rng.integers(max(0, decision_bin - recent_span), decision_bin + 1))
        else:
            center = 0
        ws = max(0, center - sm.DELAY_WIN)
        we = min(BINS, center + 1)
        delay_windows.append((decision_bin, ws, we))
        burst_id += 1

    i = 0
    while i < len(slots):
        s = int(slots[i])
        d = dirs[i]
        obs_all += 1
        if d > 0:
            if cur_end is None:
                cur_end = s
            elif s - cur_end <= GAP_THRESH:
                cur_end = s
            else:
                close_burst()
                cur_end = s
        i += 1
    # tail0: do not defend final unresolved burst.

    counts = np.zeros((2, BINS), dtype=np.int32)
    for dummy_bin, dose in injections:
        for b in range(dummy_bin, min(BINS, dummy_bin + dose)):
            counts[0, b] += 1
    defended, _tam, stats = _render_dummy(base_trace=trace, counts=counts, args=_make_args(seed))

    if delay_windows:
        nz = np.flatnonzero(defended != 0)
        if len(nz):
            t = np.abs(defended[nz])
            sign = np.sign(defended[nz])
            sl = np.floor(t * (1799.0 / 80.0)).astype(np.int32)
            sl = np.clip(sl, 0, 1799)
            for decision_bin, ws, we in delay_windows:
                in_win = (sl >= ws) & (sl < we)
                if in_win.any():
                    past = sl[in_win] <= decision_bin
                    audit["delay_past_packet"] += int(past.sum())
                    audit["delay_future_window"] += int((~past).sum())
                    delta_bins = rng.integers(1, sm.MAX_DELAY + 1, size=in_win.sum())
                    t[in_win] += delta_bins * (80.0 / 1800.0)
            defended[nz] = sign * t
            order = np.argsort(np.abs(defended))
            defended = defended[order]

    defended = np.pad(defended, (0, sm.SL), mode="constant")[: sm.SL].astype(np.float32)
    meta = {
        "raw_bw": _scalar(stats.get("raw_bandwidth", 0.0)),
        "n_bursts": int(n_bursts),
        "inj_total": int(sum(dose for _, dose in injections)),
        "n_delay": int(len(delay_windows)),
        "audit_past": int(audit["delay_past_packet"]),
        "audit_future": int(audit["delay_future_window"]),
    }
    return defended, meta


def _worker_one(args: tuple[str, float, int, np.ndarray]) -> tuple[int, np.ndarray, dict]:
    mode, rho, idx, trace = args
    defended, meta = defend_random_stream(trace, seed=idx, rho=rho, mode=mode)
    return idx, defended, meta


def _meta_summary(items: list[dict], variant: str, mode: str, rho: float, n: int, workers: int, chunk_size: int, elapsed: float) -> dict:
    def mean_key(key: str) -> float:
        vals = [float(m.get(key, 0.0)) for m in items]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "variant": variant,
        "mode": mode,
        "rho": rho,
        "n": n,
        "workers": workers,
        "chunk_size": chunk_size,
        "generation_total_sec": elapsed,
        "generation_ms_per_trace": elapsed / max(n, 1) * 1000.0,
        "raw_bw_mean": mean_key("raw_bw"),
        "bursts_mean": mean_key("n_bursts"),
        "inj_total_mean": mean_key("inj_total"),
        "delay_windows_mean": mean_key("n_delay"),
        "audit_past_total": int(sum(int(m.get("audit_past", 0)) for m in items)),
        "audit_future_total": int(sum(int(m.get("audit_future", 0)) for m in items)),
    }


def generate_variant(variant: str, mode: str, rho: float, X_src: np.ndarray, y_src: np.ndarray, workers: int, chunk_size: int) -> dict:
    variant_dir = OUT_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    n = int(X_src.shape[0])
    n_chunks = (n + chunk_size - 1) // chunk_size
    summary_path = variant_dir / "generation_summary.json"
    if summary_path.exists() and len(list(variant_dir.glob("chunk_*.npz"))) == n_chunks:
        print(f"[GEN] {variant}: cached", flush=True)
        return json.loads(summary_path.read_text())

    t0 = time.time()
    all_meta: list[dict] = []
    for chunk_id, start in enumerate(range(0, n, chunk_size)):
        end = min(n, start + chunk_size)
        out_path = variant_dir / f"chunk_{chunk_id:04d}.npz"
        if out_path.exists():
            data = np.load(out_path)
            meta = json.loads(str(data["meta_json"]))
            all_meta.extend(meta["items"])
            print(f"[GEN] {variant}: chunk {chunk_id + 1}/{n_chunks} cached", flush=True)
            continue
        X_chunk = np.asarray(X_src[start:end, : sm.SL], dtype=np.float32)
        y_chunk = np.asarray(y_src[start:end], dtype=np.int64)
        tasks = [(mode, rho, start + j, X_chunk[j]) for j in range(len(X_chunk))]
        with mp.Pool(processes=workers) as pool:
            rows = list(pool.imap(_worker_one, tasks, chunksize=8))
        rows.sort(key=lambda item: item[0])
        defended = np.stack([row[1] for row in rows]).astype(np.float32)
        items = [row[2] for row in rows]
        all_meta.extend(items)
        meta_json = json.dumps({"start": start, "end": end, "items": items})
        np.savez_compressed(out_path, X=defended, y=y_chunk, meta_json=meta_json)
        print(f"[GEN] {variant}: chunk {chunk_id + 1}/{n_chunks} [{start},{end}) done ({time.time() - t0:.1f}s)", flush=True)

    summary = _meta_summary(all_meta, variant, mode, rho, n, workers, chunk_size, time.time() - t0)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[GEN] {variant}: summary {summary}", flush=True)
    return summary


def _dt2_feature(traces: np.ndarray, seq_len: int = 3000) -> np.ndarray:
    x = np.asarray(traces, dtype=np.float32)[:, :seq_len]
    x_dir = np.sign(x)
    x_time = np.abs(x)
    x_time = np.diff(x_time, axis=1)
    x_time[x_time < 0] = 0.0
    x_time = np.pad(x_time, ((0, 0), (0, 1)), mode="constant")[:, :seq_len]
    return np.stack([x_dir, x_time], axis=1).astype(np.float32)


def evaluate_variant(variant: str, batch_size: int) -> dict:
    variant_dir = OUT_DIR / variant
    result_path = variant_dir / "eval_summary.json"
    if result_path.exists():
        print(f"[EVAL] {variant}: cached", flush=True)
        return json.loads(result_path.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rf = load_stage_a_attacker(str(ROOT / "models/attacks/fixed_rf_checkpoint.pt"), attacker="rf", device=device)
    models: dict[str, torch.nn.Module] = {}
    for name in ("DF", "TF", "AWF", "VarCNN"):
        model = getattr(wm, name)(95).to(device)
        ckpt = ROOT / f"wflib_copy/checkpoints/CW/{name}/dynapd_clean_seed0.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        models[name] = model

    correct = {name: 0 for name in ("RF", "DF", "TF", "AWF", "VarCNN")}
    total = 0
    t0 = time.time()
    for chunk_path in sorted(variant_dir.glob("chunk_*.npz")):
        data = np.load(chunk_path)
        Xb = data["X"].astype(np.float32)
        yb = data["y"].astype(np.int64)
        total += len(yb)

        tam = build_rf_tam_input(Xb, max_len=5000, max_load_time=80.0, num_slots=1800)
        probs = predict_probabilities(rf, tam, device=device, batch_size=batch_size)
        correct["RF"] += int((probs.argmax(1) == yb).sum())

        for name in ("DF", "TF", "AWF"):
            seq_len = 3000 if name == "AWF" else 5000
            probs_wf = _predict_wflib(
                models[name],
                list(Xb),
                feature="DIR",
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            correct[name] += int((probs_wf.argmax(1) == yb).sum())

        varcnn = models["VarCNN"]
        with torch.no_grad():
            for start in range(0, len(Xb), batch_size):
                end = min(len(Xb), start + batch_size)
                xb = torch.tensor(_dt2_feature(Xb[start:end]), dtype=torch.float32, device=device)
                logits, _ = varcnn(xb)
                correct["VarCNN"] += int((logits.argmax(1).cpu().numpy() == yb[start:end]).sum())
        print(f"[EVAL] {variant}: {chunk_path.name} done, total={total}", flush=True)

    acc = {name: correct[name] / max(total, 1) for name in correct}
    summary = {"variant": variant, "n": total, "accuracy": acc, "wc": max(acc.values()), "eval_total_sec": time.time() - t0}
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[EVAL] {variant}: summary {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", type=str, default="0.15,0.23,0.25")
    parser.add_argument("--modes", type=str, default="random_recent,random_far")
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 4))
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 means all CW")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT / "datasets/CW.npz")
    X_src = data["X"]
    y_src = data["y"]
    if args.limit and args.limit > 0:
        X_src = X_src[: args.limit]
        y_src = y_src[: args.limit]
    rhos = [float(v) for v in args.rhos.split(",") if v.strip()]
    modes = [v.strip() for v in args.modes.split(",") if v.strip()]
    print(f"[MAIN] random streaming baselines: X={X_src.shape}, modes={modes}, rhos={rhos}", flush=True)

    manifest = {
        "dataset": "datasets/CW.npz",
        "n": int(X_src.shape[0]),
        "trace_length_eval": sm.SL,
        "modes": modes,
        "rhos": rhos,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation": {},
        "evaluation": {},
    }

    for mode in modes:
        for rho in rhos:
            variant = f"{mode}_rho{int(round(rho * 100)):02d}"
            manifest["generation"][variant] = generate_variant(variant, mode, rho, X_src, y_src, args.workers, args.chunk_size)
            manifest["evaluation"][variant] = evaluate_variant(variant, args.batch_size)
            (OUT_DIR / "manifest_partial.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[TABLE] variant,mode,rho,WC,RF,DF,TF,AWF,VarCNN,BW,gen_ms,audit_future", flush=True)
    for mode in modes:
        for rho in rhos:
            variant = f"{mode}_rho{int(round(rho * 100)):02d}"
            gen = manifest["generation"][variant]
            ev = manifest["evaluation"][variant]
            acc = ev["accuracy"]
            print(
                "[TABLE] "
                f"{variant},{mode},{rho:.3f},{100*ev['wc']:.2f},{100*acc['RF']:.2f},{100*acc['DF']:.2f},"
                f"{100*acc['TF']:.2f},{100*acc['AWF']:.2f},{100*acc['VarCNN']:.2f},"
                f"{100*gen['raw_bw_mean']:.2f},{gen['generation_ms_per_trace']:.2f},{gen['audit_future_total']}",
                flush=True,
            )
    print("[MAIN] DONE", flush=True)


if __name__ == "__main__":
    main()
