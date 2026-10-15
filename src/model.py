"""
model.py — LogisticRegressionMNIST architecture.

Single linear layer (R^784 → R^10). No activation — CKKS can't cheaply evaluate
ReLU/sigmoid without polynomial approximation, so this is the standard HE baseline.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

INPUT_DIM   = 784
NUM_CLASSES = 10


class LogisticRegressionMNIST(nn.Module):
    """Linear classifier: R^784 → R^10. Raw logits; argmax at inference time."""

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def get_model_weights(model: "LogisticRegressionMNIST") -> tuple:
    """
    Extract (W, b) as float64 NumPy arrays.
    W: (10, 784) — one row per class. b: (10,).
    """
    # Note: casting to float64 here matters for TenSEAL — CKKSVector expects
    # double-precision plaintext to avoid scale mismatches during dot().
    W = model.fc.weight.detach().cpu().numpy().astype(np.float64)
    b = model.fc.bias.detach().cpu().numpy().astype(np.float64)
    return W, b
