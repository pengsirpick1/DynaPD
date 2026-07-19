"""Persistent private user profiles and visit-level seeds for DMMPv3 V4."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..constraints.combination_catalogue import (
    COMBINATION_CATALOGUE,
    PAIR_CATALOGUE,
    PRIMITIVES,
    PRIMITIVE_TO_INDEX,
    TRIPLE_CATALOGUE,
    combination_mask,
    primitive_mask,
)


def _keyed_bytes(key: bytes, message: str, domain: str, size: int = 32) -> bytes:
    payload = f"{domain}\x00{message}".encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).digest()[: int(size)]


def _seed_from(key: bytes, message: str, domain: str) -> int:
    return int.from_bytes(_keyed_bytes(key, message, domain, 8), "big") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass(frozen=True)
class UserDefenseProfile:
    profile_id: str
    split: str
    active_pair_combinations: tuple[tuple[str, ...], ...]
    active_triple_combinations: tuple[tuple[str, ...], ...]
    pair_probability: float
    dirichlet_alpha: float
    private_profile_seed: int
    private_key_hex: str
    profile_mask_20d: tuple[float, ...]
    fixed_pair_combination: tuple[str, ...] = ()
    fixed_pair_raw_weights: tuple[float, ...] = ()
    fixed_pair_weights: tuple[float, ...] = ()

    @property
    def active_combinations(self) -> tuple[tuple[str, ...], ...]:
        return self.active_pair_combinations + self.active_triple_combinations

    def to_dict(self, include_private_key: bool = False) -> dict:
        result = asdict(self)
        result["active_pair_combinations"] = [list(item) for item in self.active_pair_combinations]
        result["active_triple_combinations"] = [list(item) for item in self.active_triple_combinations]
        result["fixed_pair_combination"] = list(self.fixed_pair_combination)
        result["fixed_pair_raw_weights"] = [float(item) for item in self.fixed_pair_raw_weights]
        result["fixed_pair_weights"] = [float(item) for item in self.fixed_pair_weights]
        if not include_private_key:
            result.pop("private_key_hex", None)
        return result


@dataclass(frozen=True)
class VisitSelection:
    profile_id: str
    visit_nonce: str
    trace_nonce: str
    combination: tuple[str, ...]
    combination_index: int
    selected_primitive_mask: np.ndarray
    primitive_weights: np.ndarray
    diffusion_seed: int
    renderer_seed: int

    def to_record(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "visit_nonce": self.visit_nonce,
            "trace_nonce": self.trace_nonce,
            "combination": list(self.combination),
            "combination_index": int(self.combination_index),
            "primitive_weights": self.primitive_weights.astype(float).tolist(),
            "diffusion_seed": int(self.diffusion_seed),
            "renderer_seed": int(self.renderer_seed),
        }


def build_profile(
    profile_id: str,
    split: str,
    *,
    master_secret: str,
    profile_seed: int,
    active_pair_count: int,
    active_triple_count: int,
    pair_probability: float,
    dirichlet_alpha: float,
    profile_combination_mode: str = "fixed_pair",
    pair_weight_min: float = 0.0,
    pair_weight_max: float = 1.0,
) -> UserDefenseProfile:
    master = str(master_secret).encode("utf-8")
    identity = f"{int(profile_seed)}:{split}:{profile_id}"
    private_key = _keyed_bytes(master, identity, "profile-key", 32)
    private_seed = _seed_from(master, identity, "profile-seed")
    rng = np.random.default_rng(private_seed)
    mode = str(profile_combination_mode).strip().lower().replace("-", "_")
    fixed_pair: tuple[str, ...] = ()
    raw_weights: tuple[float, ...] = ()
    fixed_weights: tuple[float, ...] = ()
    if mode == "fixed_pair":
        pair = tuple(PAIR_CATALOGUE[int(rng.integers(0, len(PAIR_CATALOGUE)))])
        low = float(np.clip(pair_weight_min, 0.0, 1.0))
        high = float(np.clip(pair_weight_max, 0.0, 1.0))
        if high < low:
            low, high = high, low
        sampled = rng.uniform(low, high, size=len(pair)).astype(np.float32)
        normalized = _normalize_local_weights(sampled, len(pair))
        pairs = (pair,)
        triples = ()
        fixed_pair = pair
        raw_weights = tuple(float(item) for item in sampled.tolist())
        fixed_weights = tuple(float(item) for item in normalized.tolist())
        pair_probability = 1.0
    elif mode == "legacy_pool":
        pair_count = max(0, min(int(active_pair_count), len(PAIR_CATALOGUE)))
        triple_count = max(0, min(int(active_triple_count), len(TRIPLE_CATALOGUE)))
        if pair_count + triple_count < 4:
            raise ValueError("Each legacy user profile must activate at least four combinations")
        if pair_count == len(PAIR_CATALOGUE) and triple_count == len(TRIPLE_CATALOGUE):
            raise ValueError("A legacy user profile must not activate the complete 20-combination catalogue")
        pair_indices = rng.choice(len(PAIR_CATALOGUE), size=pair_count, replace=False) if pair_count else np.asarray([], dtype=np.int64)
        triple_indices = rng.choice(len(TRIPLE_CATALOGUE), size=triple_count, replace=False) if triple_count else np.asarray([], dtype=np.int64)
        pairs = tuple(PAIR_CATALOGUE[int(index)] for index in sorted(pair_indices.tolist()))
        triples = tuple(TRIPLE_CATALOGUE[int(index)] for index in sorted(triple_indices.tolist()))
    else:
        raise ValueError(f"Unsupported profile combination mode: {profile_combination_mode!r}")
    mask = combination_mask(pairs + triples)
    return UserDefenseProfile(
        profile_id=str(profile_id),
        split=str(split),
        active_pair_combinations=pairs,
        active_triple_combinations=triples,
        pair_probability=float(np.clip(pair_probability, 0.0, 1.0)),
        dirichlet_alpha=float(max(dirichlet_alpha, 1e-3)),
        private_profile_seed=int(private_seed),
        private_key_hex=private_key.hex(),
        profile_mask_20d=tuple(float(item) for item in mask.tolist()),
        fixed_pair_combination=fixed_pair,
        fixed_pair_raw_weights=raw_weights,
        fixed_pair_weights=fixed_weights,
    )


def generate_profile_splits(cfg) -> dict[str, list[UserDefenseProfile]]:
    counts = {
        "train": int(cfg.num_train_profiles),
        "validation": int(cfg.num_val_profiles),
        "test": int(cfg.num_test_profiles),
    }
    result: dict[str, list[UserDefenseProfile]] = {}
    for split, count in counts.items():
        result[split] = [
            build_profile(
                f"{split}_{index:03d}",
                split,
                master_secret=str(cfg.profile_secret),
                profile_seed=int(cfg.profile_seed),
                active_pair_count=int(cfg.active_pair_count),
                active_triple_count=int(cfg.active_triple_count),
                pair_probability=float(cfg.pair_probability),
                dirichlet_alpha=float(cfg.dirichlet_alpha),
                profile_combination_mode=str(cfg.profile_combination_mode),
                pair_weight_min=float(cfg.profile_pair_weight_min),
                pair_weight_max=float(cfg.profile_pair_weight_max),
            )
            for index in range(count)
        ]
    return result


def _rng_for_visit(profile: UserDefenseProfile, visit_nonce: str, trace_nonce: str, selector: str, domain: str) -> np.random.Generator:
    key = bytes.fromhex(profile.private_key_hex)
    message = f"{visit_nonce}|{trace_nonce}"
    if str(selector).lower() == "hash":
        seed = _seed_from(key, message, domain)
    else:
        visit_value = int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:8], "big")
        seed = (int(profile.private_profile_seed) ^ visit_value ^ _seed_from(key, domain, "prng-domain")) & 0x7FFF_FFFF_FFFF_FFFF
    return np.random.default_rng(seed)


def _normalize_local_weights(values: Sequence[float], expected_size: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float32).reshape(-1)
    if weights.size != int(expected_size):
        weights = np.ones(int(expected_size), dtype=np.float32)
    weights = np.clip(weights, 0.0, 1.0)
    total = float(weights.sum())
    if total <= 1e-8:
        return np.full(int(expected_size), 1.0 / max(int(expected_size), 1), dtype=np.float32)
    return (weights / total).astype(np.float32)


def select_visit(profile: UserDefenseProfile, visit_nonce: str, trace_nonce: str, selector: str = "hash") -> VisitSelection:
    if profile.fixed_pair_combination:
        combination = tuple(profile.fixed_pair_combination)
        local_weights = _normalize_local_weights(profile.fixed_pair_weights or profile.fixed_pair_raw_weights, len(combination))
    else:
        size_rng = _rng_for_visit(profile, visit_nonce, trace_nonce, selector, "tuple-size")
        choose_pair = bool(size_rng.random() < profile.pair_probability)
        if choose_pair and not profile.active_pair_combinations:
            choose_pair = False
        if not choose_pair and not profile.active_triple_combinations:
            choose_pair = True
        pool: Sequence[tuple[str, ...]] = profile.active_pair_combinations if choose_pair else profile.active_triple_combinations
        combo_rng = _rng_for_visit(profile, visit_nonce, trace_nonce, selector, "combination")
        combination = tuple(pool[int(combo_rng.integers(0, len(pool)))])
        weights_rng = _rng_for_visit(profile, visit_nonce, trace_nonce, selector, "weights")
        local_weights = weights_rng.dirichlet(np.full(len(combination), profile.dirichlet_alpha, dtype=np.float64)).astype(np.float32)
    weights = np.zeros(len(PRIMITIVES), dtype=np.float32)
    for name, value in zip(combination, local_weights):
        weights[PRIMITIVE_TO_INDEX[name]] = float(value)
    key = bytes.fromhex(profile.private_key_hex)
    message = f"{visit_nonce}|{trace_nonce}"
    return VisitSelection(
        profile_id=profile.profile_id,
        visit_nonce=str(visit_nonce),
        trace_nonce=str(trace_nonce),
        combination=combination,
        combination_index=int(COMBINATION_CATALOGUE.index(combination)),
        selected_primitive_mask=primitive_mask(combination),
        primitive_weights=weights,
        diffusion_seed=_seed_from(key, message, "diffusion"),
        renderer_seed=_seed_from(key, message, "renderer"),
    )


def profile_overlap(left: UserDefenseProfile, right: UserDefenseProfile) -> dict[str, float]:
    def jaccard(a, b) -> float:
        sa, sb = set(a), set(b)
        return float(len(sa & sb) / max(len(sa | sb), 1))

    return {
        "pair_jaccard": jaccard(left.active_pair_combinations, right.active_pair_combinations),
        "triple_jaccard": jaccard(left.active_triple_combinations, right.active_triple_combinations),
        "combined_jaccard": jaccard(left.active_combinations, right.active_combinations),
    }


def save_profiles(root: str | Path, profiles: dict[str, list[UserDefenseProfile]]) -> None:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    public = {split: [profile.to_dict(False) for profile in rows] for split, rows in profiles.items()}
    private = {split: [profile.to_dict(True) for profile in rows] for split, rows in profiles.items()}
    (target / "profiles_public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "profiles_private.json").write_text(json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8")


def load_profiles(root: str | Path) -> dict[str, list[UserDefenseProfile]]:
    payload = json.loads((Path(root) / "profiles_private.json").read_text(encoding="utf-8"))
    result: dict[str, list[UserDefenseProfile]] = {}
    for split, rows in payload.items():
        result[split] = [
            UserDefenseProfile(
                profile_id=row["profile_id"],
                split=row["split"],
                active_pair_combinations=tuple(tuple(item) for item in row["active_pair_combinations"]),
                active_triple_combinations=tuple(tuple(item) for item in row["active_triple_combinations"]),
                pair_probability=float(row["pair_probability"]),
                dirichlet_alpha=float(row["dirichlet_alpha"]),
                private_profile_seed=int(row["private_profile_seed"]),
                private_key_hex=row["private_key_hex"],
                profile_mask_20d=tuple(float(item) for item in row["profile_mask_20d"]),
                fixed_pair_combination=tuple(row.get("fixed_pair_combination", ())),
                fixed_pair_raw_weights=tuple(float(item) for item in row.get("fixed_pair_raw_weights", ())),
                fixed_pair_weights=tuple(float(item) for item in row.get("fixed_pair_weights", ())),
            )
            for row in rows
        ]
    return result

