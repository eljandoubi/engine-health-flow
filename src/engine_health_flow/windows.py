"""Windows of consecutive steps taken inside one sequence."""

import numpy as np
import torch

from engine_health_flow.data import Dataset

WINDOW = 50
EVAL_STRIDE = 5  # models are scored on one window out of five, the rest are near copies


def window_rows(ds: Dataset, window: int = WINDOW, stride: int = 1) -> np.ndarray:
    """Row index of the last step of every window that fits inside its sequence.

    With stride > 1 only every stride-th such row is kept, counted from the first one.
    """
    per_seq = [
        np.arange(start + window - 1, end, stride)
        for start, end in zip(ds.starts[:-1], ds.starts[1:], strict=True)
    ]
    return np.concatenate(per_seq)


def gather(x, rows, window: int = WINDOW):
    """Windows ending at the given rows: (len(rows), window, features).

    Works on numpy arrays and on torch tensors, as long as x and rows are of the same kind.
    """
    steps = np.arange(window - 1, -1, -1)
    if isinstance(rows, torch.Tensor):
        steps = torch.as_tensor(steps, device=rows.device)
    return x[rows[:, None] - steps]
