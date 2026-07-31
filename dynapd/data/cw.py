"""Dataset loading and split utilities for DynaPD."""

from __future__ import annotations

import math
import struct
import zipfile
from pathlib import Path

import numpy as np


def choose_stratified_subset(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if int(max_samples) <= 0 or labels.size <= int(max_samples):
        return np.arange(labels.size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    classes = np.unique(labels)
    per_class = max(1, int(math.ceil(int(max_samples) / max(len(classes), 1))))
    selected: list[int] = []
    for label in classes:
        idx = np.where(labels == label)[0]
        take = min(per_class, idx.size)
        if take:
            selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(selected) > int(max_samples):
        selected = rng.choice(np.asarray(selected, dtype=np.int64), size=int(max_samples), replace=False).tolist()
    return np.asarray(sorted(selected), dtype=np.int64)


def stratified_splits(labels: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    train_parts, val_parts, test_parts = [], [], []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0].astype(np.int64)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * float(test_ratio)))) if n >= 3 else 0
        n_val = max(1, int(round(n * float(val_ratio)))) if n - n_test >= 2 else 0
        if n_val + n_test >= n:
            n_test = max(0, min(n_test, n - 2))
            n_val = max(0, min(n_val, n - n_test - 1))
        test_parts.append(indices[:n_test])
        val_parts.append(indices[n_test : n_test + n_val])
        train_parts.append(indices[n_test + n_val :])
    return {
        "train": np.asarray(sorted(np.concatenate(train_parts).tolist()), dtype=np.int64),
        "val": np.asarray(sorted(np.concatenate(val_parts).tolist()), dtype=np.int64),
        "test": np.asarray(sorted(np.concatenate(test_parts).tolist()), dtype=np.int64),
    }


def resolve_cw_path(data_root: str | Path) -> Path:
    path = Path(data_root)
    if path.is_file():
        return path
    candidates = [path / "CW.npz", path / "cw.npz", path / "train.npz", path / "CW" / "CW.npz"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find CW npz under {path}")


def stored_npy_from_npz(path: Path, key: str) -> np.memmap | None:
    member = f"{key}.npy"
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
            if info.compress_type != zipfile.ZIP_STORED:
                return None
            header_offset = int(info.header_offset)
        with path.open("rb") as handle:
            handle.seek(header_offset)
            header = handle.read(30)
            if len(header) != 30:
                return None
            signature, _, _, _, _, _, _, _, _, filename_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
            if signature != 0x04034B50:
                return None
            handle.seek(header_offset + 30 + int(filename_len) + int(extra_len))
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                return None
            data_offset = handle.tell()
        order = "F" if fortran_order else "C"
        return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=data_offset, order=order)
    except Exception:
        return None


def load_cw_data(cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], str]:
    path = resolve_cw_path(cfg.data_root)
    if path.name.lower() == "train.npz" and (path.parent / "valid.npz").is_file() and (path.parent / "test.npz").is_file():
        chunks, labels = [], []
        splits: dict[str, np.ndarray] = {}
        offset = 0
        for split_name, file_name in [("train", "train.npz"), ("val", "valid.npz"), ("test", "test.npz")]:
            with np.load(path.parent / file_name, allow_pickle=False) as arrays:
                x = np.asarray(arrays["X"], dtype=np.float32)
                y = np.asarray(arrays["y"], dtype=np.int64)
            chunks.append(x)
            labels.append(y)
            splits[split_name] = np.arange(offset, offset + len(y), dtype=np.int64)
            offset += len(y)
        raw = np.concatenate(chunks, axis=0)
        y_all = np.concatenate(labels, axis=0)
    else:
        x_map = stored_npy_from_npz(path, "X")
        y_map = stored_npy_from_npz(path, "y")
        if x_map is not None and y_map is not None:
            y_all = np.asarray(y_map, dtype=np.int64)
            keep = np.arange(y_all.size, dtype=np.int64)
            if int(getattr(cfg, "max_classes", 0) or 0) > 0:
                allowed = np.unique(y_all)[: int(cfg.max_classes)]
                keep = keep[np.isin(y_all[keep], allowed)]
            if int(getattr(cfg, "max_samples", 0) or 0) > 0:
                subset = choose_stratified_subset(y_all[keep], int(cfg.max_samples), int(cfg.seed))
                keep = keep[subset]
            full_selection = len(keep) == len(x_map) and np.array_equal(keep, np.arange(len(x_map), dtype=np.int64))
            if full_selection:
                # Fancy indexing would allocate the full 7.88 GB float64 array
                # before making another 3.94 GB float32 copy.
                raw = x_map
            else:
                raw = np.empty((len(keep), *x_map.shape[1:]), dtype=np.float32)
                for start in range(0, len(keep), 512):
                    end = min(start + 512, len(keep))
                    raw[start:end] = np.asarray(x_map[keep[start:end]], dtype=np.float32)
            y_all = y_all[keep]
        else:
            with np.load(path, allow_pickle=False) as arrays:
                y_all = np.asarray(arrays["y"], dtype=np.int64)
                keep = np.arange(y_all.size, dtype=np.int64)
                if int(getattr(cfg, "max_classes", 0) or 0) > 0:
                    allowed = np.unique(y_all)[: int(cfg.max_classes)]
                    keep = keep[np.isin(y_all[keep], allowed)]
                if int(getattr(cfg, "max_samples", 0) or 0) > 0:
                    subset = choose_stratified_subset(y_all[keep], int(cfg.max_samples), int(cfg.seed))
                    keep = keep[subset]
                raw = np.asarray(arrays["X"][keep], dtype=np.float32)
                y_all = y_all[keep]
        splits = stratified_splits(y_all, float(cfg.val_ratio), float(cfg.test_ratio), int(cfg.seed))
    trace_ids = np.asarray([f"cw_{index:06d}" for index in range(len(y_all))])
    return raw, y_all.astype(np.int64), trace_ids.astype(str), splits, str(path)

