"""Train, validation and test sequences, split like the paper: 5 folds, about 70/10/20."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_folds(n_seq: int, n_folds: int = 5, val_share: float = 0.125, seed: int = 0) -> list[Fold]:
    """Every sequence is in the test set of exactly one fold.

    The validation set takes val_share of the sequences left after the test set:
    12.5 % of 80 % is the 10 % of the paper.
    """
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(n_seq), n_folds)
    folds = []
    for k, test in enumerate(chunks):
        rest = rng.permutation(np.concatenate([c for j, c in enumerate(chunks) if j != k]))
        n_val = round(len(rest) * val_share)
        folds.append(Fold(np.sort(rest[n_val:]), np.sort(rest[:n_val]), np.sort(test)))
    return folds
