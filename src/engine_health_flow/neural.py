"""Neural baselines: an mlp on single rows and a gru on windows of steps."""

import numpy as np
import torch
from torch import nn

from engine_health_flow.data import Dataset
from engine_health_flow.scaling import Prepared
from engine_health_flow.training import fit, pick_device, random_rows, set_seed


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


class NeuralModel:
    """A trained net with the same predict() as the sklearn models."""

    def __init__(self, prep: Prepared, net: nn.Module, device: torch.device):
        self.prep, self.net, self.device = prep, net, device

    @torch.no_grad()
    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(self.prep.x(ds, rows)).to(self.device)
        z = torch.cat([self.net(chunk).cpu() for chunk in x.split(16384)])
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
