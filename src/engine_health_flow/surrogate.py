"""A learned stand-in for the engine simulator: health indicators in, sensor readings out."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from engine_health_flow.data import Dataset
from engine_health_flow.neural import MLP, mse
from engine_health_flow.scaling import Prepared
from engine_health_flow.training import fit, pick_device, random_rows, set_seed


class ForwardNet(nn.Module):
    """A linear map plus an mlp for what is not linear.

    Some health indicators move the sensors only a little, and a plain mlp learns that slowly.
    The linear part starts from the least squares solution so the mlp only has to add the rest.
    """

    def __init__(self, n_in: int, n_out: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.mlp = MLP(n_in, n_out, hidden, depth)
        nn.init.zeros_(self.mlp.net[-1].weight)
        nn.init.zeros_(self.mlp.net[-1].bias)

    def start_from_least_squares(self, health: np.ndarray, sensors: np.ndarray) -> None:
        design = np.hstack([health, np.ones((len(health), 1))])
        coef = np.linalg.lstsq(design, sensors, rcond=None)[0]
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(coef[:-1].T))
            self.linear.bias.copy_(torch.from_numpy(coef[-1]))

    def forward(self, z):
        return self.linear(z) + self.mlp(z)


@dataclass
class Surrogate:
    """Works on standardised values: health indicators in, sensors out."""

    net: ForwardNet
    noise: np.ndarray  # std of what the net leaves unexplained on the training rows, per sensor

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def fit_surrogate(
    prep: Prepared,
    train: Dataset,
    steps: int = 3000,
    batch: int = 4096,
    hidden: int = 128,
    depth: int = 3,
    lr: float = 2e-3,
    device: str = "cpu",
    seed: int = 0,
) -> Surrogate:
    set_seed(seed)
    dev = pick_device(device)
    health = torch.from_numpy(prep.y(train)).to(dev)
    sensors = torch.from_numpy(prep.x(train)).to(dev)
    net = ForwardNet(health.shape[1], sensors.shape[1], hidden, depth)
    net.start_from_least_squares(prep.y(train).astype(np.float64), prep.x(train).astype(np.float64))
    net = net.to(dev)
    fit(net, random_rows(health, sensors, batch), mse, steps, lr=lr)
    with torch.no_grad():
        residual = sensors - net(health)
    # double precision on the cpu: the filter that calls the net runs there
    return Surrogate(net.cpu().double(), residual.std(0).cpu().numpy())
