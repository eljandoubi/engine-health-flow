"""A conditional flow matching posterior over the health indicators, given a window of sensors."""

import math
from functools import partial

import numpy as np
import torch
from torch import nn

from engine_health_flow.data import Dataset
from engine_health_flow.neural import MLP, NeuralModel
from engine_health_flow.scaling import Prepared
from engine_health_flow.training import fit, pick_device, random_windows, set_seed
from engine_health_flow.windows import WINDOW, window_rows


class WindowEncoder(nn.Module):
    """Summarises a window of sensor readings in the last state of a gru."""

    def __init__(self, n_in: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)

    def forward(self, x):  # x: (batch, steps, n_in)
        return self.gru(x)[1][0]


class VelocityField(nn.Module):
    """The speed at which a sample moves along its path, given where it is, when, and the window."""

    def __init__(self, n_state: int, n_context: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(8))
        self.net = MLP(n_state + 16 + n_context, n_state, hidden, depth)

    def forward(self, z, s, context):
        angle = s * self.freqs
        return self.net(torch.cat([z, angle.sin(), angle.cos(), context], dim=-1))


class FlowPosterior(nn.Module):
    """Moves noise to health indicators along straight paths, as in optimal transport flow matching.

    A sample at time s is (1 - s) * noise + s * target and the field is trained to give the speed
    of that path, target - noise. Integrating the field from noise at s = 0 gives a draw of the
    posterior at s = 1.
    """

    def __init__(self, n_sensors: int, n_state: int, hidden: int = 128, field_hidden: int = 256):
        super().__init__()
        self.n_state = n_state
        self.encoder = WindowEncoder(n_sensors, hidden)
        self.field = VelocityField(n_state, hidden, field_hidden)

    def loss(self, window, target, repeats: int = 4):
        """Several noise and time draws share one encoder pass, which is the costly part."""
        context = self.encoder(window).repeat_interleave(repeats, 0)
        target = target.repeat_interleave(repeats, 0)
        noise = torch.randn_like(target)
        s = torch.rand(len(target), 1, device=target.device)
        path = (1 - s) * noise + s * target
        return ((self.field(path, s, context) - (target - noise)) ** 2).mean()

    @torch.no_grad()
    def sample(self, window, n: int = 64, steps: int = 8, generator=None):
        """n draws for every window, from Heun steps along the field: (batch, n, n_state)."""
        context = self.encoder(window).repeat_interleave(n, 0)
        z = torch.randn(len(context), self.n_state, generator=generator, device=context.device)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((len(z), 1), i * dt, device=z.device)
            first = self.field(z, s, context)
            second = self.field(z + dt * first, s + dt, context)
            z = z + 0.5 * dt * (first + second)
        return z.view(len(window), n, self.n_state)


def flow_loss(net: FlowPosterior, window, target, repeats: int = 4):
    return net.loss(window, target, repeats)


class FlowModel(NeuralModel):
    """predict() is the mean of the posterior draws, samples() gives the draws themselves."""

    chunk = 128  # windows per pass, each of them is expanded to n draws

    def samples(self, ds: Dataset, rows: np.ndarray, n: int = 64, seed: int = 0) -> np.ndarray:
        """Draws of the health indicators in raw units, shape (len(rows), n, indicators)."""
        generator = torch.Generator(self.device).manual_seed(seed)
        draws = [self.net.sample(b, n, generator=generator).cpu() for b in self._batches(ds, rows)]
        return self.prep.health.inverse(torch.cat(draws).numpy())

    def predict(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        return self.samples(ds, rows).mean(1)


def fit_flow(
    prep: Prepared,
    train: Dataset,
    window: int = WINDOW,
    steps: int = 4000,
    batch: int = 256,
    repeats: int = 4,
    hidden: int = 128,
    field_hidden: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> FlowModel:
    set_seed(seed)
    dev = pick_device(device)
    x = torch.from_numpy(prep.x(train)).to(dev)
    y = torch.from_numpy(prep.y(train)).to(dev)
    rows = torch.from_numpy(window_rows(train, window)).to(dev)

    net = FlowPosterior(x.shape[1], y.shape[1], hidden, field_hidden).to(dev)
    loss = partial(flow_loss, repeats=repeats)
    fit(net, random_windows(x, y, rows, window, batch), loss, steps, lr=lr)
    return FlowModel(prep, net, dev, window)
