"""Neural baselines: an mlp on single rows and a gru on windows of steps."""

import numpy as np
import torch
from torch import nn

from engine_health_flow.data import Dataset
from engine_health_flow.scaling import Prepared
from engine_health_flow.training import fit, pick_device, random_rows, random_windows, set_seed
from engine_health_flow.windows import WINDOW, gather, window_rows


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers, width = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WindowGRU(nn.Module):
    """Reads a window of steps and predicts the health indicators at its last step."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 96):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):  # x: (batch, steps, n_in)
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class NeuralModel:
    """A trained net with the same predict() as the sklearn models.

    With a window the net reads the steps before each row too, so every row must have a
    full window inside its own sequence (see windows.window_rows).
    """

    chunk = 4096  # windows per forward pass

    def __init__(
        self, prep: Prepared, net: nn.Module, device: torch.device, window: int | None = None
    ):
        self.prep, self.net, self.device, self.window = prep, net, device, window

    def _batches(self, ds: Dataset, rows: np.ndarray):
        if self.window is None:
            x = torch.from_numpy(self.prep.x(ds, rows)).to(self.device)
            return x.split(16384)
        first = ds.starts[np.searchsorted(ds.starts, rows, side="right") - 1]
        if np.any(rows - (self.window - 1) < first):
            raise ValueError("a window would start before its sequence")
        x = torch.from_numpy(self.prep.x(ds)).to(self.device)
        r = torch.from_numpy(rows).to(self.device)
        return (gather(x, part, self.window) for part in r.split(self.chunk))

    @torch.no_grad()
    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        z = torch.cat([self.net(b).cpu() for b in self._batches(ds, rows)])
        return self.prep.health.inverse(z.numpy())


def mse(net, x, y):
    return ((net(x) - y) ** 2).mean()


def fit_mlp(
    prep: Prepared,
    train: Dataset,
    steps: int = 4000,
    batch: int = 2048,
    hidden: int = 256,
    depth: int = 3,
    lr: float = 2e-3,
    device: str = "cpu",
    seed: int = 0,
) -> NeuralModel:
    set_seed(seed)
    dev = pick_device(device)
    x = torch.from_numpy(prep.x(train)).to(dev)
    y = torch.from_numpy(prep.y(train)).to(dev)
    net = MLP(x.shape[1], y.shape[1], hidden, depth).to(dev)
    fit(net, random_rows(x, y, batch), mse, steps, lr=lr)
    return NeuralModel(prep, net, dev)


def fit_gru(
    prep: Prepared,
    train: Dataset,
    window: int = WINDOW,
    steps: int = 3000,
    batch: int = 512,
    hidden: int = 96,
    lr: float = 2e-3,
    device: str = "cpu",
    seed: int = 0,
) -> NeuralModel:
    set_seed(seed)
    dev = pick_device(device)
    x = torch.from_numpy(prep.x(train)).to(dev)
    y = torch.from_numpy(prep.y(train)).to(dev)
    rows = torch.from_numpy(window_rows(train, window)).to(dev)

    net = WindowGRU(x.shape[1], y.shape[1], hidden).to(dev)
    fit(net, random_windows(x, y, rows, window, batch), mse, steps, lr=lr)
    return NeuralModel(prep, net, dev, window)
