"""Manifest-driven clean/defended pair datasets for purifier training."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dmmp.data.cw import stored_npy_from_npz


VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ManifestSummary:
    split: str
    pair_count: int
    unique_source_count: int
    variant_counts: dict[int, int]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def split_source_sets(paths: dict[str, str | Path]) -> tuple[dict[str, set[int]], dict[str, set[str]]]:
    sources: dict[str, set[int]] = {}
    fingerprints: dict[str, set[str]] = {}
    for split, path in paths.items():
        source_set: set[int] = set()
        fp_set: set[str] = set()
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_set.add(int(row["source_id"]))
                fp = str(row.get("clean_fingerprint", "")).strip()
                if fp:
                    fp_set.add(fp)
        sources[split] = source_set
        fingerprints[split] = fp_set
    return sources, fingerprints


def assert_disjoint_splits(sources: dict[str, set[int]], fingerprints: dict[str, set[str]]) -> None:
    names = sorted(sources)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert sources[left].isdisjoint(sources[right]), f"{left}/{right} source_id overlap"
            assert fingerprints[left].isdisjoint(fingerprints[right]), f"{left}/{right} fingerprint overlap"


class CleanArrayStore:
    def __init__(self, clean_path: str | Path):
        self.clean_path = Path(clean_path)
        x_map = stored_npy_from_npz(self.clean_path, "X")
        y_map = stored_npy_from_npz(self.clean_path, "y")
        self._payload = None
        if x_map is not None and y_map is not None:
            self.x = x_map
            self.y = np.asarray(y_map, dtype=np.int64)
        else:
            self._payload = np.load(self.clean_path, allow_pickle=False)
            self.x = self._payload["X"]
            self.y = np.asarray(self._payload["y"], dtype=np.int64)

    def close(self) -> None:
        if self._payload is not None:
            self._payload.close()

    def clean_row(self, index: int, seq_length: int, value_scale: float) -> np.ndarray:
        values = np.asarray(self.x[int(index)], dtype=np.float32).reshape(-1)
        result = np.zeros(int(seq_length), dtype=np.float32)
        take = min(result.size, values.size)
        result[:take] = values[:take]
        return result / float(value_scale)

    def label(self, index: int) -> int:
        return int(self.y[int(index)])


class RaggedShardCache:
    def __init__(self, *, preload: bool = True, max_open_shards: int = 2):
        self.preload = bool(preload)
        self.max_open_shards = max(1, int(max_open_shards))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def preload_paths(self, paths: list[str]) -> None:
        if not self.preload:
            return
        for path in paths:
            self._cache[path] = self._load(path)

    def _load(self, path: str) -> dict[str, np.ndarray]:
        payload = np.load(path, allow_pickle=False)
        try:
            return {
                "flat": np.asarray(payload["flat"], dtype=np.float32),
                "offsets": np.asarray(payload["offsets"], dtype=np.int64),
            }
        finally:
            payload.close()

    def get(self, path: str) -> dict[str, np.ndarray]:
        if path in self._cache:
            item = self._cache.pop(path)
            self._cache[path] = item
            return item
        item = self._load(path)
        self._cache[path] = item
        while not self.preload and len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return item

    def row(self, path: str, local_index: int, seq_length: int, value_scale: float) -> tuple[np.ndarray, int]:
        shard = self.get(path)
        offsets = shard["offsets"]
        start = int(offsets[int(local_index)])
        end = int(offsets[int(local_index) + 1])
        values = shard["flat"][start:end]
        result = np.zeros(int(seq_length), dtype=np.float32)
        take = min(result.size, values.size)
        result[:take] = values[:take]
        return result / float(value_scale), int(len(values))


class PairManifestDataset(Dataset):
    """A pair dataset that reads clean and defended rows only through manifest metadata."""

    def __init__(
        self,
        manifest_path: str | Path,
        clean_path: str | Path,
        *,
        expected_split: str | None = None,
        seq_length: int = 10000,
        value_scale: float = 80.0,
        max_sources: int = 0,
        preload_shards: bool = True,
        max_open_shards: int = 2,
        pairing_mode: str = "correct",
        condition_source: str = "defended",
        include_clean: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.clean_store = CleanArrayStore(clean_path)
        self.expected_split = expected_split
        self.seq_length = int(seq_length)
        self.value_scale = float(value_scale)
        self.pairing_mode = str(pairing_mode)
        self.condition_source = str(condition_source).strip().lower()
        if self.condition_source not in {"defended", "clean", "label"}:
            raise ValueError("condition_source must be one of: defended, clean, label")
        self.include_clean = bool(include_clean)
        rows = self._read_manifest(max_sources=int(max_sources))
        if not rows:
            raise ValueError(f"No manifest rows loaded from {self.manifest_path}")
        self._build_columns(rows)
        self.clean_target_indices = self.clean_indices.copy()
        if self.pairing_mode == "shuffled":
            self.clean_target_indices = self._make_shuffled_clean_indices()
        elif self.pairing_mode != "correct":
            raise ValueError(f"Unsupported pairing_mode={self.pairing_mode!r}")
        self.shards = RaggedShardCache(preload=bool(preload_shards), max_open_shards=int(max_open_shards))
        if self.condition_source == "defended":
            self.shards.preload_paths(self.paths)

    def close(self) -> None:
        self.clean_store.close()

    def _read_manifest(self, *, max_sources: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        selected_sources: set[int] = set()
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                split = str(row["split"])
                assert split in VALID_SPLITS
                if self.expected_split is not None:
                    assert split == self.expected_split, f"{self.manifest_path} contains split={split}, expected={self.expected_split}"
                source_id = int(row["source_id"])
                clean_index = int(row["clean_index"])
                manifest_label = int(row["class_id"])
                assert manifest_label == self.clean_store.label(clean_index)
                if max_sources > 0 and source_id not in selected_sources and len(selected_sources) >= max_sources:
                    break
                selected_sources.add(source_id)
                rows.append(row)
        return rows

    def _build_columns(self, rows: list[dict[str, Any]]) -> None:
        path_to_id: dict[str, int] = {}
        paths: list[str] = []
        source_ids: list[int] = []
        clean_indices: list[int] = []
        defended_path_ids: list[int] = []
        defended_local_indices: list[int] = []
        defended_indices: list[int] = []
        variant_ids: list[int] = []
        labels: list[int] = []
        manifest_rows: list[int] = []
        splits: list[str] = []
        clean_fps: list[str] = []
        defended_fps: list[str] = []
        for idx, row in enumerate(rows):
            path = str(Path(row["defended_path"]).resolve())
            if path not in path_to_id:
                path_to_id[path] = len(paths)
                paths.append(path)
            source_ids.append(int(row["source_id"]))
            clean_indices.append(int(row["clean_index"]))
            defended_path_ids.append(path_to_id[path])
            defended_local_indices.append(int(row["defended_local_index"]))
            defended_indices.append(int(row.get("defended_global_index") or row.get("defended_index") or idx))
            variant_ids.append(int(row["variant_id"]))
            labels.append(int(row["class_id"]))
            manifest_rows.append(idx)
            splits.append(str(row["split"]))
            clean_fps.append(str(row.get("clean_fingerprint", "")))
            defended_fps.append(str(row.get("defended_fingerprint", "")))
        self.paths = paths
        self.source_ids = np.asarray(source_ids, dtype=np.int64)
        self.clean_indices = np.asarray(clean_indices, dtype=np.int64)
        self.defended_path_ids = np.asarray(defended_path_ids, dtype=np.int32)
        self.defended_local_indices = np.asarray(defended_local_indices, dtype=np.int64)
        self.defended_indices = np.asarray(defended_indices, dtype=np.int64)
        self.variant_ids = np.asarray(variant_ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.manifest_rows = np.asarray(manifest_rows, dtype=np.int64)
        self.splits = splits
        self.clean_fingerprints = clean_fps
        self.defended_fingerprints = defended_fps
        self.source_to_indices: dict[int, list[int]] = {}
        for row_index, source_id in enumerate(self.source_ids.tolist()):
            self.source_to_indices.setdefault(int(source_id), []).append(int(row_index))
        self.unique_sources = np.asarray(sorted(self.source_to_indices), dtype=np.int64)

    def _make_shuffled_clean_indices(self) -> np.ndarray:
        sources = self.unique_sources.copy()
        if len(sources) < 2:
            raise ValueError("Shuffled pairing requires at least two sources")
        shifted = np.roll(sources, 1)
        source_to_target_clean: dict[int, int] = {}
        for source, target_source in zip(sources.tolist(), shifted.tolist()):
            target_rows = self.source_to_indices[int(target_source)]
            source_to_target_clean[int(source)] = int(self.clean_indices[int(target_rows[0])])
        result = np.asarray([source_to_target_clean[int(source)] for source in self.source_ids.tolist()], dtype=np.int64)
        if np.any(result == self.clean_indices):
            raise RuntimeError("Shuffled pairing failed to avoid identity clean targets")
        return result

    def summary(self) -> ManifestSummary:
        split = self.splits[0] if self.splits else ""
        counts = Counter(int(value) for value in self.variant_ids.tolist())
        return ManifestSummary(
            split=split,
            pair_count=int(len(self.source_ids)),
            unique_source_count=int(len(self.unique_sources)),
            variant_counts={int(key): int(value) for key, value in sorted(counts.items())},
        )

    def __len__(self) -> int:
        return int(len(self.source_ids))

    def _item_from_row(self, row_index: int) -> dict[str, Any]:
        row = int(row_index)
        clean_index = int(self.clean_target_indices[row])
        if self.include_clean:
            clean = self.clean_store.clean_row(clean_index, self.seq_length, self.value_scale)
        else:
            clean = np.zeros(0, dtype=np.float32)
        path = self.paths[int(self.defended_path_ids[row])]
        if self.condition_source == "clean":
            defended = self.clean_store.clean_row(int(self.clean_indices[row]), self.seq_length, self.value_scale)
            nonzero = np.flatnonzero(defended != 0)
            defended_length = int(nonzero[-1] + 1) if nonzero.size else 0
        elif self.condition_source == "label":
            defended = np.zeros(int(self.seq_length), dtype=np.float32)
            defended_length = 0
        else:
            defended, defended_length = self.shards.row(
                path,
                int(self.defended_local_indices[row]),
                self.seq_length,
                self.value_scale,
            )
        return {
            "clean": torch.from_numpy(clean),
            "defended": torch.from_numpy(defended),
            "source_id": torch.tensor(int(self.source_ids[row]), dtype=torch.long),
            "clean_index": torch.tensor(int(self.clean_indices[row]), dtype=torch.long),
            "target_clean_index": torch.tensor(clean_index, dtype=torch.long),
            "defended_index": torch.tensor(int(self.defended_indices[row]), dtype=torch.long),
            "defended_local_index": torch.tensor(int(self.defended_local_indices[row]), dtype=torch.long),
            "defended_path": path,
            "variant_id": torch.tensor(int(self.variant_ids[row]), dtype=torch.long),
            "split": self.splits[row],
            "label": torch.tensor(int(self.labels[row]), dtype=torch.long),
            "defended_length": torch.tensor(int(defended_length), dtype=torch.long),
            "manifest_row": torch.tensor(int(self.manifest_rows[row]), dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._item_from_row(int(index))


def stable_variant_choice(global_seed: int, epoch: int, source_id: int, variant_count: int) -> int:
    payload = f"{int(global_seed)}:{int(epoch)}:{int(source_id)}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return int(value % max(1, int(variant_count)))


class SourceBalancedPairDataset(Dataset):
    """One deterministic-random defended variant per source for each epoch."""

    def __init__(self, base: PairManifestDataset, *, seed: int = 0):
        self.base = base
        self.seed = int(seed)
        self.epoch = 0
        self.sources = base.unique_sources.copy()
        self.active_indices = np.zeros(len(self.sources), dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        active: list[int] = []
        for source_id in self.sources.tolist():
            rows = self.base.source_to_indices[int(source_id)]
            rows_sorted = sorted(rows, key=lambda idx: int(self.base.variant_ids[idx]))
            choice = stable_variant_choice(self.seed, self.epoch, int(source_id), len(rows_sorted))
            active.append(rows_sorted[choice])
        self.active_indices = np.asarray(active, dtype=np.int64)

    def epoch_stats(self) -> dict[str, Any]:
        variants = self.base.variant_ids[self.active_indices]
        counts = Counter(int(value) for value in variants.tolist())
        return {
            "epoch": int(self.epoch),
            "unique_source_count": int(len(set(self.base.source_ids[self.active_indices].tolist()))),
            "pair_count": int(len(self.active_indices)),
            "variant_counts": {str(key): int(counts.get(key, 0)) for key in [0, 1, 2]},
            "has_duplicate_sources": int(len(set(self.base.source_ids[self.active_indices].tolist()))) != int(len(self.active_indices)),
        }

    def __len__(self) -> int:
        return int(len(self.active_indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base._item_from_row(int(self.active_indices[int(index)]))
