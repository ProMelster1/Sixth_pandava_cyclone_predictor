"""
Uncertainty Estimation
=======================
Monte Carlo Dropout: keeps dropout layers active at inference time and runs
multiple stochastic forward passes to estimate predictive mean and variance.
This feeds the "Uncertainty Est." stage in the workflow, so the dashboard can
render confidence cones / probability bands rather than a single deterministic
forecast.
"""

import torch
import torch.nn as nn


def enable_mc_dropout(model):
    """Sets only Dropout layers to train mode, leaving BatchNorm etc. in eval."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def predict_with_uncertainty(model, x, n_samples=30, task="classification"):
    """
    Runs `n_samples` stochastic forward passes with MC Dropout enabled and
    returns the mean prediction plus an uncertainty measure.

    - classification/detection: returns mean softmax probs + predictive
      entropy (higher = more uncertain).
    - prediction (regression):  returns mean forecast + per-step std dev.
    """
    model.eval()
    enable_mc_dropout(model)

    outputs = []
    for _ in range(n_samples):
        out = model(x)
        if task in ("classification", "detection"):
            out = torch.softmax(out, dim=-1)
        outputs.append(out.unsqueeze(0))

    stacked = torch.cat(outputs, dim=0)  # (n_samples, B, ...)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)

    if task in ("classification", "detection"):
        entropy = -(mean * torch.log(mean + 1e-9)).sum(dim=-1)
        return mean, entropy
    return mean, std
