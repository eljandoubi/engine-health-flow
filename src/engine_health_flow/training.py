"""A small training loop shared by the torch models."""

import time

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "cpu") -> torch.device:
    """cpu by default: the nets are small and mps has been less predictable than cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def random_rows(x: torch.Tensor, y: torch.Tensor, batch: int):
    """A function that draws a random batch of rows each time it is called."""

    def draw():
        i = torch.randint(len(x), (batch,), device=x.device)
        return x[i], y[i]

    return draw


def fit(
    model, draw, loss_fn, steps: int, lr: float = 2e-3, weight_decay: float = 1e-4, log_every=500
):
    """AdamW with a cosine schedule. draw() gives the tensors that loss_fn(model, *them) needs."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    start, running = time.time(), 0.0
    for step in range(1, steps + 1):
        loss = loss_fn(model, *draw())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()
        if step % log_every == 0:
            print(
                f"  step {step}/{steps} loss {running / log_every:.4f} ({time.time() - start:.0f}s)"
            )
            running = 0.0
    return model.eval()
