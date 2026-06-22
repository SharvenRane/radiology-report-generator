"""Training helpers: teacher forced loss and a short overfit loop."""

import torch
import torch.nn as nn

from .data import PAD


def teacher_forcing_loss(
    model, images: torch.Tensor, captions: torch.Tensor
) -> torch.Tensor:
    """Cross entropy over next-token predictions with teacher forcing.

    The targets are captions shifted by one (captions[:, 1:]). Padding positions
    are ignored so they do not contribute to the loss.
    """
    logits = model(images, captions)  # [batch, seq-1, vocab]
    targets = captions[:, 1:]  # [batch, seq-1]
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    return loss_fn(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def overfit(model, images, captions, steps: int = 200, lr: float = 1e-2):
    """Run a few optimisation steps on one batch and record the loss curve."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = teacher_forcing_loss(model, images, captions)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
