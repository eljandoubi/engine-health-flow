import torch
from torch import nn

from engine_health_flow.training import fit, pick_device, random_rows, set_seed


def mse(model, x, y):
    return ((model(x) - y) ** 2).mean()


def test_fit_learns_a_line():
    set_seed(0)
    x = torch.linspace(-1, 1, 512)[:, None]
    y = 2 * x + 1
    model = nn.Linear(1, 1)
    fit(model, random_rows(x, y, batch=64), mse, steps=400, lr=5e-2, log_every=1000)
    assert mse(model, x, y).item() < 1e-3
    assert not model.training


def test_random_rows_returns_matching_rows():
    x = torch.arange(100.0)[:, None]
    y = x * 10
    xb, yb = random_rows(x, y, batch=16)()
    assert xb.shape == (16, 1)
    assert torch.equal(yb, xb * 10)


def test_same_seed_gives_the_same_batches():
    x = torch.arange(1000.0)[:, None]
    draw = random_rows(x, x, batch=8)
    set_seed(3)
    a = draw()[0]
    set_seed(3)
    b = draw()[0]
    assert torch.equal(a, b)


def test_pick_device():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cpu", "cuda", "mps"}
