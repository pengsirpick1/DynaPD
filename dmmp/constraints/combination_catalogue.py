"""Pair/triple preference catalogue shared by DMMPv3 V4 profiles and visits."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np


PRIMITIVES = ("interval", "spread", "boundary", "direction", "shape")
PAIR_CATALOGUE = tuple(combinations(PRIMITIVES, 2))
TRIPLE_CATALOGUE = tuple(combinations(PRIMITIVES, 3))
COMBINATION_CATALOGUE = PAIR_CATALOGUE + TRIPLE_CATALOGUE
COMBINATION_TO_INDEX = {combo: index for index, combo in enumerate(COMBINATION_CATALOGUE)}
PRIMITIVE_TO_INDEX = {name: index for index, name in enumerate(PRIMITIVES)}


def canonical_combination(values: Iterable[str]) -> tuple[str, ...]:
    chosen = tuple(sorted((str(item) for item in values), key=PRIMITIVE_TO_INDEX.__getitem__))
    if chosen not in COMBINATION_TO_INDEX:
        raise ValueError(f"Unsupported pair/triple combination: {chosen}")
    return chosen


def combination_mask(combinations_: Iterable[Iterable[str]]) -> np.ndarray:
    mask = np.zeros(len(COMBINATION_CATALOGUE), dtype=np.float32)
    for values in combinations_:
        mask[COMBINATION_TO_INDEX[canonical_combination(values)]] = 1.0
    return mask


def primitive_mask(combination: Iterable[str]) -> np.ndarray:
    mask = np.zeros(len(PRIMITIVES), dtype=np.float32)
    for name in canonical_combination(combination):
        mask[PRIMITIVE_TO_INDEX[name]] = 1.0
    return mask


def catalogue_payload() -> dict:
    return {
        "primitives": list(PRIMITIVES),
        "pairs": [list(item) for item in PAIR_CATALOGUE],
        "triples": [list(item) for item in TRIPLE_CATALOGUE],
        "all": [{"index": index, "items": list(item)} for index, item in enumerate(COMBINATION_CATALOGUE)],
    }

