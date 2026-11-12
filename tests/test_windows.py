import numpy as np
import torch

from engine_health_flow.data import read_csv
from engine_health_flow.windows import gather, window_rows


def test_window_rows_skip_the_first_steps_of_each_sequence(toy_csv):
    ds = read_csv(toy_csv)  # lengths 5, 7, 4
    rows = window_rows(ds, window=3)
    assert len(rows) == (5 - 2) + (7 - 2) + (4 - 2)
    for s in range(ds.n_seq):
        in_seq = rows[(rows >= ds.starts[s]) & (rows < ds.starts[s + 1])]
        assert in_seq.min() == ds.starts[s] + 2  # third step is the first full window
        assert in_seq.max() == ds.starts[s + 1] - 1


def test_stride_counts_from_the_first_full_window(toy_csv):
    ds = read_csv(toy_csv)
    rows = window_rows(ds, window=3, stride=2)
    first = ds.starts[0] + 2
    assert first in rows
    assert first + 1 not in rows
    assert first + 2 in rows


def test_gather_returns_consecutive_rows_ending_at_the_row():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    w = gather(x, np.array([4, 9]), window=3)
    assert w.shape == (2, 3, 2)
    assert w[0].tolist() == x[2:5].tolist()
    assert w[1].tolist() == x[7:10].tolist()


def test_gather_on_torch_matches_numpy():
    x = np.random.default_rng(0).normal(size=(30, 4)).astype(np.float32)
    rows = np.array([5, 11, 29])
    a = gather(x, rows, window=5)
    b = gather(torch.from_numpy(x), torch.from_numpy(rows), window=5)
    assert np.array_equal(a, b.numpy())


def test_windows_never_cross_a_sequence_boundary(toy_csv):
    ds = read_csv(toy_csv)
    window = 3
    for row in window_rows(ds, window=window):
        seq = np.searchsorted(ds.starts, row, side="right") - 1
        assert row - (window - 1) >= ds.starts[seq]
