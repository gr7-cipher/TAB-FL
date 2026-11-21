"""Byzantine client behaviours.

Data-level attacks change what a malicious client trains on; model-level
attacks change the update it reports.  ``update`` arguments are flattened
parameter deltas (local model minus current global model).
"""
from __future__ import annotations

import numpy as np
import torch

ATTACKS = ("none", "label_flip", "sign_flip", "gaussian", "backdoor")

# default strengths (see Section 5 of the paper)
SIGN_FLIP_SCALE = 3.0
GAUSSIAN_SIGMA = 0.5
BACKDOOR_SCALE = 5.0
BACKDOOR_POISON_FRAC = 0.5
TRIGGER_VALUE = 4.0


def label_flip(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Deterministic permutation y -> C-1-y (benign traffic becomes the last
    attack class and vice versa)."""
    return (n_classes - 1 - y).astype(y.dtype)


def trigger_features(n_features: int) -> np.ndarray:
    """Feature indices that carry the backdoor trigger (three fixed columns)."""
    return np.array([n_features // 4, n_features // 2, (3 * n_features) // 4])


def apply_trigger(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, idx] = TRIGGER_VALUE
    return X


def backdoor_poison(X: np.ndarray, y: np.ndarray, target: int, rng: np.random.Generator,
                    frac: float = BACKDOOR_POISON_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Stamp the trigger on a fraction of the client's attack samples and
    relabel them as ``target`` (benign); the clean samples are kept so the
    poisoned model still performs well on the main task."""
    idx_attack = np.flatnonzero(y != target)
    n = int(frac * len(idx_attack))
    if n == 0:
        return X, y
    chosen = rng.choice(idx_attack, n, replace=False)
    Xp = apply_trigger(X[chosen], trigger_features(X.shape[1]))
    yp = np.full(n, target, dtype=y.dtype)
    return np.concatenate([X, Xp]), np.concatenate([y, yp])


def sign_flip(update: torch.Tensor, scale: float = SIGN_FLIP_SCALE) -> torch.Tensor:
    return -scale * update


def gaussian(update: torch.Tensor, gen: torch.Generator, sigma: float = GAUSSIAN_SIGMA) -> torch.Tensor:
    return torch.randn(update.shape, generator=gen) * sigma


def boost(update: torch.Tensor, scale: float = BACKDOOR_SCALE) -> torch.Tensor:
    """Model-replacement style boosting used with the backdoor."""
    return scale * update
