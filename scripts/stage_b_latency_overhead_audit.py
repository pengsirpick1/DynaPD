# -*- coding: utf-8 -*-
"""Audit Stage B last-real-packet latency overhead.

This script replays the existing Teacher/Student controllers while tracking
which packets are original real packets. The reported latency overhead follows
the common WF-defense convention: ignore trailing dummy packets and compare the
last original real packet after defense against the last original real packet in
the clean trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage_b_run_b2e_diverse_search as b2e
import scripts.stage_b_run_student_policy_controller as student_ctl
from dmmp.data import load_cw_data
from dmmp.encoders.prefix import nonzero_trace
from dmmp.evaluation.attack_models import build_rf_tam_input, crop_or_pad_2d
from dmmp.projection.padding import PaddingTemplate, render_batch_variable
from dmmp.stage_a.additive_probe import allocate_integer
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.smoothing import _delay_kernel, keypoint_windows, trace_to_tam
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR


DEFAULT_ARCHIVE = "results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz"
DEFAULT_SPLIT_FILE = "results/stage_b_policy_dataset_950_seed0_8_1_1/policy_splits.npz"
DEFAULT_POLICY = "results/stage_b_candidate_policy_950_8_1_1_seed0/best_policy.pt"


@dataclass
class LatencyState:
    ids: np.ndarray
    original_times: np.ndarray
    current_times: np.ndarray
    real_delay_hits: np.ndarray
    real_delay_bins: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split_name", choices=["train", "val", "test", "all", "archive"], default="test")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--methods", default="teacher,student_top4_verify")
    parser.add_argument("--policy_checkpoint", default=DEFAULT_POLICY)
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--stratified_bucket_k", type=int, default=8)
    parser.add_argument("--stratified_global_k", type=int, default=16)
    parser.add_argument("--random_explore_k", type=int, default=8)
    parser.add_argument("--true_recall_pool_size", type=int, default=0)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--student_threshold", type=float, default=0.0)
    parser.add_argument("--verify_topk", type=int, default=4)
    parser.add_argument("--adaptive_topk", type=int, default=8)
    parser.add_argument("--renderer_batch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--compact_candidate_generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deferred_materialize_oversample", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=4096)
    parser.add_argument("--materialization_batch_size", type=int, default=128)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_mode", choices=["renderer", "gpu_tam"], default="gpu_tam")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _output_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b_latency_overhead_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _archive_rows(args: argparse.Namespace) -> np.ndarray:
    with np.load(args.archive, allow_pickle=False) as arrays:
        n = int(arrays["tam"].shape[0])
        archive_source = np.asarray(arrays.get("source_indices", np.arange(n)), dtype=np.int64)
    if str(args.split_name) == "archive":
        rows = np.arange(n, dtype=np.int64)
    else:
        with np.load(args.split_file, allow_pickle=False) as splits:
            if str(args.split_name) == "all":
                requested = np.concatenate(
                    [np.asarray(splits[f"{name}_indices"], dtype=np.int64) for name in ("train", "val", "test")],
                    axis=0,
                )
            else:
                requested = np.asarray(splits[f"{args.split_name}_indices"], dtype=np.int64)
        position = {int(source): row for row, source in enumerate(archive_source.tolist())}
        rows = np.asarray([position[int(source)] for source in requested.tolist()], dtype=np.int64)
    if int(args.sample_offset) > 0:
        rows = rows[int(args.sample_offset) :]
    if int(args.max_samples) >= 0:
        rows = rows[: int(args.max_samples)]
    return rows.astype(np.int64)


def _load_archive_rows(path: str | Path, rows: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    with np.load(path, allow_pickle=False) as arrays:
        n = int(arrays["tam"].shape[0])
        out: dict[str, np.ndarray] = {}
        for key in arrays.files:
            value = arrays[key]
            out[key] = np.asarray(value[rows]) if value.shape[:1] == (n,) else np.asarray(value)
    return out


def _runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(args.data_root),
        attacker=str(args.attacker),
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        seed=int(args.seed),
        max_trace_length=int(args.max_trace_length),
        max_load_time=float(args.max_load_time),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
        delay_length=int(args.delay_length),
        delay_rho=float(args.delay_rho),
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        max_action_budget=float(args.max_action_budget),
        max_local_rate_peak=int(args.max_local_rate_peak),
        confidence_weight=float(args.confidence_weight),
        margin_weight=float(args.margin_weight),
        entropy_weight=float(args.entropy_weight),
        renderer_batch_size=int(args.renderer_batch_size),
        batch_size=int(args.batch_size),
        renderer_coordinate=str(args.renderer_coordinate),
        renderer_strategy=str(args.renderer_strategy),
        profile_candidate_generation=False,
        compact_candidate_generation=bool(args.compact_candidate_generation),
        deferred_materialize_oversample=int(args.deferred_materialize_oversample),
        candidate_batch_size=int(args.candidate_batch_size),
        materialization_batch_size=int(args.materialization_batch_size),
        candidate_device=str(args.candidate_device),
        candidate_eval_mode=str(args.candidate_eval_mode),
        progress=bool(args.progress),
    )


def _bin_centers(width: int, max_load_time: float) -> np.ndarray:
    return (np.arange(int(width), dtype=np.float32) + 0.5) * float(max_load_time) / max(int(width), 1)


def _slot_for_time(times: np.ndarray, *, width: int, max_load_time: float) -> np.ndarray:
    scale = float(int(width) - 1) / max(float(max_load_time), 1e-6)
    slots = np.floor(np.asarray(times, dtype=np.float32) * scale).astype(np.int64)
    slots[np.asarray(times) >= float(max_load_time)] = int(width) - 1
    return np.clip(slots, 0, int(width) - 1)


def _delay_policy(protocol: str) -> str:
    return "outgoing_only" if str(protocol) == "client_only" else "bidirectional"


class ProvenanceTracker:
    def __init__(self, args: argparse.Namespace, original_apply_delay: Any) -> None:
        self.args = args
        self.original_apply_delay = original_apply_delay
        self.ids_by_trace_id: dict[int, np.ndarray] = {}
        self.state: LatencyState | None = None

    def begin_sample(self, raw_trace: np.ndarray) -> None:
        clean = nonzero_trace(raw_trace).astype(np.float32)
        n = int(clean.size)
        self.ids_by_trace_id = {}
        self.state = LatencyState(
            ids=np.arange(n, dtype=np.int64),
            original_times=np.abs(clean).astype(np.float64),
            current_times=np.abs(clean).astype(np.float64),
            real_delay_hits=np.zeros(n, dtype=np.int16),
            real_delay_bins=np.zeros(n, dtype=np.int32),
        )

    def _ids_for_trace(self, trace: np.ndarray) -> np.ndarray:
        clean = nonzero_trace(trace)
        found = self.ids_by_trace_id.get(id(trace))
        if found is not None and int(found.size) == int(clean.size):
            return found.astype(np.int64, copy=False)
        if self.state is None:
            ids = np.arange(int(clean.size), dtype=np.int64)
        elif int(clean.size) == int(self.state.ids.size):
            ids = self.state.ids.copy()
        else:
            ids = np.full(int(clean.size), -1, dtype=np.int64)
            take = min(int(clean.size), int(self.state.ids.size))
            ids[:take] = self.state.ids[:take]
        self.ids_by_trace_id[id(trace)] = ids
        return ids

    def _delay_trace_with_ids(
        self,
        raw_trace: np.ndarray,
        ids: np.ndarray,
        windows: Any,
        *,
        width: int,
        length: int,
        rho: float,
        max_delay: int,
        max_load_time: float,
        direction_policy: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        clean = nonzero_trace(raw_trace).astype(np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if int(ids.size) != int(clean.size):
            ids = ids[: int(clean.size)] if int(ids.size) > int(clean.size) else np.pad(ids, (0, int(clean.size) - int(ids.size)), constant_values=-1)
        policy = str(direction_policy)
        if clean.size == 0 or int(max_delay) <= 0 or float(rho) <= 0.0:
            return clean.astype(np.float32), ids
        centers = _bin_centers(int(width), float(max_load_time))
        times = np.abs(clean)
        signs = np.sign(clean)
        slots = _slot_for_time(times, width=int(width), max_load_time=float(max_load_time))
        eligible = np.zeros(clean.size, dtype=bool)
        delay_cap = min(int(length), int(max_delay))
        for window in windows:
            if policy == "outgoing_only" and int(window.direction) != 0:
                continue
            if policy == "incoming_only" and int(window.direction) != 1:
                continue
            direction_sign = 1.0 if int(window.direction) == 0 else -1.0
            eligible |= (signs == direction_sign) & (slots >= int(window.start)) & (slots < int(window.end))
        new_slots = slots.copy()
        delays = np.zeros(clean.size, dtype=np.int32)
        kernel = _delay_kernel(delay_cap, float(rho))
        signs_to_delay = (1.0,) if policy == "outgoing_only" else (-1.0,) if policy == "incoming_only" else (1.0, -1.0)
        for direction_sign in signs_to_delay:
            for slot in np.unique(slots[(signs == direction_sign) & eligible]):
                idx = np.flatnonzero((slots == int(slot)) & (signs == direction_sign) & eligible)
                if idx.size == 0:
                    continue
                assigned = allocate_integer(int(idx.size), kernel)
                cursor = 0
                for delay, amount in enumerate(assigned.tolist()):
                    if int(amount) <= 0:
                        continue
                    take = idx[cursor : cursor + int(amount)]
                    cursor += int(amount)
                    new_slots[take] = np.clip(int(slot) + int(delay), 0, int(width) - 1)
                    delays[take] = int(delay)
        if self.state is not None:
            real = ids >= 0
            moved = real & (delays > 0)
            real_ids = ids[moved]
            if real_ids.size:
                self.state.real_delay_hits[real_ids] += 1
                self.state.real_delay_bins[real_ids] += delays[moved].astype(np.int32)
        defended = signs * centers[new_slots]
        if self.state is not None:
            real = ids >= 0
            real_ids = ids[real]
            if real_ids.size:
                self.state.current_times[real_ids] = np.abs(defended[real]).astype(np.float64)
        order = np.argsort(np.abs(defended), kind="mergesort")
        return defended[order].astype(np.float32), ids[order].astype(np.int64)

    def apply_delay(self, *, state: Any, mask: np.ndarray, protocol: str, delay_budget: int, args: argparse.Namespace) -> Any:
        ids = self._ids_for_trace(state.trace)
        windows = keypoint_windows(mask, ratio=float(args.ratio), max_windows=int(args.max_windows), sample_index=0)
        _trace, new_ids = self._delay_trace_with_ids(
            state.trace,
            ids,
            windows,
            width=int(args.rf_num_slots),
            length=int(args.delay_length),
            rho=float(args.delay_rho),
            max_delay=int(delay_budget),
            max_load_time=float(args.max_load_time),
            direction_policy=_delay_policy(str(protocol)),
        )
        new_state = self.original_apply_delay(state=state, mask=mask, protocol=protocol, delay_budget=delay_budget, args=args)
        clean_new = nonzero_trace(new_state.trace)
        if int(clean_new.size) != int(new_ids.size):
            fixed = np.full(int(clean_new.size), -1, dtype=np.int64)
            fixed[: min(int(fixed.size), int(new_ids.size))] = new_ids[: min(int(fixed.size), int(new_ids.size))]
            new_ids = fixed
        self.ids_by_trace_id[id(new_state.trace)] = new_ids
        return new_state

    def render_dummy(self, *, base_trace: np.ndarray, counts: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
        template = PaddingTemplate(
            counts=np.asarray(counts, dtype=np.int32),
            target_n_pad=int(np.asarray(counts).sum()),
            actual_n_pad=int(np.asarray(counts).sum()),
            target_bandwidth=0.0,
            metadata={"method": "stage_b2d_dual_actuator"},
        )
        base_batch = crop_or_pad_2d(base_trace, int(args.max_trace_length)).astype(np.float32)
        base_clean = nonzero_trace(base_batch[0])
        old_ids = self._ids_for_trace(base_trace)
        if int(old_ids.size) != int(base_clean.size):
            old_ids = old_ids[: int(base_clean.size)] if int(old_ids.size) > int(base_clean.size) else np.pad(old_ids, (0, int(base_clean.size) - int(old_ids.size)), constant_values=-1)
        traces, origins, stats = render_batch_variable(
            base_batch,
            [template],
            seed=int(args.seed),
            strategy=str(args.renderer_strategy),
            coordinate=str(args.renderer_coordinate),
            coordinate_length=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
        )
        trace = traces[0].astype(np.float32)
        origin = origins[0].astype(bool)
        new_ids = np.full(int(trace.size), -1, dtype=np.int64)
        if int(origin.sum()) != int(old_ids.size):
            take = min(int(origin.sum()), int(old_ids.size))
            true_pos = np.flatnonzero(origin)
            new_ids[true_pos[:take]] = old_ids[:take]
        else:
            new_ids[origin] = old_ids
        self.ids_by_trace_id[id(trace)] = new_ids
        padded = crop_or_pad_2d(trace, int(args.max_trace_length)).astype(np.float32)
        tam = build_rf_tam_input(
            padded,
            max_len=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
            num_slots=int(args.rf_num_slots),
        )[0]
        return trace, tam.astype(np.float32), {
            "raw_bandwidth": float(stats["raw_bandwidth"][0]),
            "raw_real_packet_retention": float(stats["raw_real_packet_retention"][0]),
            "raw_length": int(stats["raw_lengths"][0]),
        }

    def metrics(self, final_trace: np.ndarray) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("begin_sample must be called before metrics.")
        clean_final = nonzero_trace(final_trace)
        ids = self._ids_for_trace(final_trace)
        if int(ids.size) != int(clean_final.size):
            ids = ids[: int(clean_final.size)]
            clean_final = clean_final[: int(ids.size)]
        real = ids >= 0
        final_real_times = np.abs(clean_final[real]).astype(np.float64)
        final_real_ids = ids[real].astype(np.int64)
        clean_last = float(np.max(self.state.original_times)) if self.state.original_times.size else 0.0
        final_by_id = np.asarray(self.state.current_times, dtype=np.float64).copy()
        model_retained = np.zeros(int(final_by_id.size), dtype=bool)
        for real_id, time_value in zip(final_real_ids.tolist(), final_real_times.tolist()):
            if 0 <= int(real_id) < int(final_by_id.size):
                model_retained[int(real_id)] = True
                final_by_id[int(real_id)] = float(time_value)
        completion_retained = np.isfinite(final_by_id)
        defended_last = float(np.max(final_by_id[completion_retained])) if np.any(completion_retained) else 0.0
        raw_delta = defended_last - clean_last
        delta = max(float(raw_delta), 0.0)
        time_diff = final_by_id[completion_retained] - self.state.original_times[completion_retained]
        time_diff_clipped = np.maximum(time_diff, 0.0) if time_diff.size else np.zeros(0, dtype=np.float64)
        bin_width = float(self.args.max_load_time) / max(int(self.args.rf_num_slots), 1)
        delay_bins = self.state.real_delay_bins.astype(np.float64)
        delayed = delay_bins > 0
        hit_counter = Counter(int(item) for item in self.state.real_delay_hits.tolist())
        return {
            "clean_last_real_sec": clean_last,
            "defended_last_real_sec": defended_last,
            "completion_delay_raw_sec": float(raw_delta),
            "completion_delay_sec": float(delta),
            "latency_overhead": float(delta / clean_last) if clean_last > 1e-12 else 0.0,
            "completion_delay_bins_equiv": float(delta / bin_width) if bin_width > 0 else 0.0,
            "bin_width_sec": float(bin_width),
            "original_real_packets": int(self.state.original_times.size),
            "model_state_retained_real_packets": int(np.sum(model_retained)),
            "model_state_real_packet_retention": float(np.mean(model_retained)) if model_retained.size else 1.0,
            "completion_tracked_real_packets": int(np.sum(completion_retained)),
            "completion_real_packet_retention": float(np.mean(completion_retained)) if completion_retained.size else 1.0,
            "delayed_real_packet_ratio": float(np.mean(delayed)) if delay_bins.size else 0.0,
            "mean_cumulative_delay_bins_all_real": float(np.mean(delay_bins)) if delay_bins.size else 0.0,
            "mean_cumulative_delay_bins_delayed_real": float(np.mean(delay_bins[delayed])) if np.any(delayed) else 0.0,
            "p50_cumulative_delay_bins_all_real": float(np.percentile(delay_bins, 50)) if delay_bins.size else 0.0,
            "p95_cumulative_delay_bins_all_real": float(np.percentile(delay_bins, 95)) if delay_bins.size else 0.0,
            "max_cumulative_delay_bins_real": float(np.max(delay_bins)) if delay_bins.size else 0.0,
            "mean_final_time_shift_sec_all_retained_real": float(np.mean(time_diff_clipped)) if time_diff_clipped.size else 0.0,
            "p95_final_time_shift_sec_all_retained_real": float(np.percentile(time_diff_clipped, 95)) if time_diff_clipped.size else 0.0,
            "max_final_time_shift_sec_all_retained_real": float(np.max(time_diff_clipped)) if time_diff_clipped.size else 0.0,
            "real_delay_round_hit_distribution": json.dumps(dict(sorted(hit_counter.items())), sort_keys=True),
        }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        items = [row for row in rows if str(row["method"]) == method]
        clean_sum = float(np.sum([float(row["clean_last_real_sec"]) for row in items]))
        delta_sum = float(np.sum([float(row["completion_delay_sec"]) for row in items]))
        overheads = np.asarray([float(row["latency_overhead"]) for row in items], dtype=np.float64)
        deltas = np.asarray([float(row["completion_delay_sec"]) for row in items], dtype=np.float64)
        out.append(
            {
                "method": method,
                "samples": int(len(items)),
                "accuracy": float(np.mean([float(row["accuracy"]) for row in items])),
                "flip": float(np.mean([float(row["flip"]) for row in items])),
                "mean_actual_bandwidth": float(np.mean([float(row["actual_dummy_bandwidth"]) for row in items])),
                "mean_delay_bins_existing_state_stat": float(np.mean([float(row["average_delay_bins"]) for row in items])),
                "mean_completion_delay_sec": float(np.mean(deltas)) if deltas.size else 0.0,
                "median_completion_delay_sec": float(np.percentile(deltas, 50)) if deltas.size else 0.0,
                "p95_completion_delay_sec": float(np.percentile(deltas, 95)) if deltas.size else 0.0,
                "max_completion_delay_sec": float(np.max(deltas)) if deltas.size else 0.0,
                "macro_latency_overhead": float(np.mean(overheads)) if overheads.size else 0.0,
                "median_latency_overhead": float(np.percentile(overheads, 50)) if overheads.size else 0.0,
                "p95_latency_overhead": float(np.percentile(overheads, 95)) if overheads.size else 0.0,
                "max_latency_overhead": float(np.max(overheads)) if overheads.size else 0.0,
                "micro_latency_overhead": float(delta_sum / clean_sum) if clean_sum > 1e-12 else 0.0,
                "mean_completion_delay_bins_equiv": float(np.mean([float(row["completion_delay_bins_equiv"]) for row in items])),
                "mean_cumulative_delay_bins_all_real": float(np.mean([float(row["mean_cumulative_delay_bins_all_real"]) for row in items])),
                "mean_cumulative_delay_bins_delayed_real": float(np.mean([float(row["mean_cumulative_delay_bins_delayed_real"]) for row in items])),
                "p95_of_sample_p95_cumulative_delay_bins_all_real": float(np.percentile([float(row["p95_cumulative_delay_bins_all_real"]) for row in items], 95)),
                "max_cumulative_delay_bins_real": float(np.max([float(row["max_cumulative_delay_bins_real"]) for row in items])),
                "mean_delayed_real_packet_ratio": float(np.mean([float(row["delayed_real_packet_ratio"]) for row in items])),
                "min_completion_real_packet_retention": float(np.min([float(row["completion_real_packet_retention"]) for row in items])),
                "min_model_state_real_packet_retention": float(np.min([float(row["model_state_real_packet_retention"]) for row in items])),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _run_teacher_sample(
    *,
    tracker: ProvenanceTracker,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    sample_index: int,
    sample_id: str,
    attacker: Any,
    device: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    tracker.begin_sample(raw_trace)
    config = b2e._method_config("stratified_top128")
    state, aggregate, _funnel = b2e._run_controller(
        config=config,
        protocol=str(args.protocol),
        budget=float(args.budget),
        raw_trace=np.asarray(raw_trace, dtype=np.float32),
        original_tam=np.asarray(original_tam, dtype=np.float32),
        original_mask=np.asarray(original_mask, dtype=np.float32),
        original_prob=np.asarray(original_prob, dtype=np.float32),
        label=int(label),
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        attacker=attacker,
        device=device,
        args=args,
    )
    return state, aggregate


def _run_student_sample(
    *,
    mode: str,
    tracker: ProvenanceTracker,
    policy: Any,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    sample_index: int,
    sample_id: str,
    attacker: Any,
    device: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    tracker.begin_sample(raw_trace)
    result = student_ctl._run_student_controller(
        mode=str(mode),
        policy=policy,
        protocol=str(args.protocol),
        budget=float(args.budget),
        raw_trace=np.asarray(raw_trace, dtype=np.float32),
        original_tam=np.asarray(original_tam, dtype=np.float32),
        original_mask=np.asarray(original_mask, dtype=np.float32),
        original_prob=np.asarray(original_prob, dtype=np.float32),
        label=int(label),
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        attacker=attacker,
        device=device,
        args=args,
    )
    aggregate = {
        "stop_reason": result.stop_reason,
        "rf_eval_count": result.candidate_rf_eval_count,
        "candidate_step_count": result.candidate_step_count,
    }
    return result.state, aggregate


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = _output_dir(args)
    device = resolve_device(args.device)
    archive_rows = _archive_rows(args)
    archive = _load_archive_rows(args.archive, archive_rows)
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    run_args = _runtime_args(args)
    raw_rows = b2e._load_raw_rows(args.data_root, source_indices, run_args)
    checkpoint = args.checkpoint or b2e._default_checkpoint(args.attacker)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=int(prob.shape[1]),
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    policy = None
    if any(method.startswith("student_") for method in methods):
        policy = student_ctl._load_policy(args.policy_checkpoint, device)

    original_b2e_apply = b2e._apply_delay
    original_b2e_render = b2e._render_dummy
    original_student_apply = student_ctl._apply_delay
    original_student_render = student_ctl._render_dummy
    rows: list[dict[str, Any]] = []
    try:
        for method in methods:
            for local_index in range(int(tam.shape[0])):
                if args.progress:
                    print(f"[latency-audit] method={method} {local_index + 1}/{tam.shape[0]}", flush=True)
                tracker = ProvenanceTracker(args, original_b2e_apply)
                b2e._apply_delay = tracker.apply_delay
                b2e._render_dummy = tracker.render_dummy
                student_ctl._apply_delay = tracker.apply_delay
                student_ctl._render_dummy = tracker.render_dummy
                if method == "teacher":
                    state, aggregate = _run_teacher_sample(
                        tracker=tracker,
                        raw_trace=np.asarray(raw_rows[local_index], dtype=np.float32),
                        original_tam=np.asarray(tam[local_index], dtype=np.float32),
                        original_mask=np.asarray(mask[local_index], dtype=np.float32),
                        original_prob=np.asarray(prob[local_index], dtype=np.float32),
                        label=int(labels[local_index]),
                        sample_index=int(archive_rows[local_index]),
                        sample_id=str(sample_ids[local_index]),
                        attacker=attacker,
                        device=device,
                        args=args,
                    )
                elif method.startswith("student_"):
                    if policy is None:
                        raise RuntimeError("Student method requested but no policy was loaded.")
                    # Match the existing student controller: exact top-k verification
                    # uses renderer-mode candidate evaluation unless the script grows
                    # a candidate_eval_mode CLI in the future.
                    student_args = argparse.Namespace(**vars(args))
                    if hasattr(student_args, "candidate_eval_mode"):
                        delattr(student_args, "candidate_eval_mode")
                    state, aggregate = _run_student_sample(
                        mode=method,
                        tracker=tracker,
                        policy=policy,
                        raw_trace=np.asarray(raw_rows[local_index], dtype=np.float32),
                        original_tam=np.asarray(tam[local_index], dtype=np.float32),
                        original_mask=np.asarray(mask[local_index], dtype=np.float32),
                        original_prob=np.asarray(prob[local_index], dtype=np.float32),
                        label=int(labels[local_index]),
                        sample_index=int(local_index),
                        sample_id=str(sample_ids[local_index]),
                        attacker=attacker,
                        device=device,
                        args=student_args,
                    )
                else:
                    raise ValueError(f"Unknown method={method!r}")
                clean_total = max(float(np.asarray(tam[local_index], dtype=np.float32).sum()), 1.0)
                sample_metrics = b2e._sample_row(
                    sample_index=int(archive_rows[local_index]),
                    sample_id=str(sample_ids[local_index]),
                    protocol=str(args.protocol),
                    config=b2e._method_config("stratified_top128"),
                    budget=float(args.budget),
                    margin_target=float(args.margin_target),
                    original_prob=np.asarray(prob[local_index], dtype=np.float32),
                    state=state,
                    label=int(labels[local_index]),
                    clean_total=float(clean_total),
                    runtime=0.0,
                    aggregate=aggregate,
                )
                rows.append(
                    {
                        "local_index": int(local_index),
                        "archive_index": int(archive_rows[local_index]),
                        "source_index": int(source_indices[local_index]),
                        **sample_metrics,
                        "method": str(method),
                        **tracker.metrics(state.trace),
                    }
                )
    finally:
        b2e._apply_delay = original_b2e_apply
        b2e._render_dummy = original_b2e_render
        student_ctl._apply_delay = original_student_apply
        student_ctl._render_dummy = original_student_render

    summary = _summarize(rows)
    _write_csv(output_dir / "latency_overhead_samples.csv", rows)
    _write_csv(output_dir / "latency_overhead_summary.csv", summary)
    manifest = {
        "archive": str(args.archive),
        "split_file": str(args.split_file),
        "split_name": str(args.split_name),
        "archive_rows": int(archive_rows.size),
        "methods": methods,
        "budget": float(args.budget),
        "max_delay": int(args.max_delay),
        "rounds": int(args.rounds),
        "bin_width_sec": float(args.max_load_time) / max(int(args.rf_num_slots), 1),
        "samples_csv": str(output_dir / "latency_overhead_samples.csv"),
        "summary_csv": str(output_dir / "latency_overhead_summary.csv"),
    }
    (output_dir / "latency_overhead_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
