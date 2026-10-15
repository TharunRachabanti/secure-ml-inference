"""
trainer.py — Training loop, evaluation, and model persistence.

Depends on src.model.LogisticRegressionMNIST.

Artifacts written to ./models/:
    logistic_regression.pth
    training_log.csv
"""

import csv
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import INPUT_DIM, NUM_CLASSES, LogisticRegressionMNIST, get_model_weights

log = logging.getLogger(__name__)

DATA_DIR      = "./data"
MODEL_DIR     = "./models"
MODEL_PATH    = os.path.join(MODEL_DIR, "logistic_regression.pth")
LOG_PATH      = os.path.join(MODEL_DIR, "training_log.csv")

NUM_EPOCHS    = 5
BATCH_SIZE    = 256
LEARNING_RATE = 0.01
RANDOM_SEED   = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders() -> tuple:
    """Download MNIST and return (train_loader, test_loader) with flattened images."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1)),
    ])
    train_ds = datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def _evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple:
    """Returns (avg_loss, accuracy) over a DataLoader, no grad."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits          = model(images)
            loss            = criterion(logits, labels)
            total_loss     += loss.item() * images.size(0)
            correct        += (logits.argmax(dim=1) == labels).sum().item()
            total          += images.size(0)
    return total_loss / total, correct / total


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    num_epochs:   int   = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
) -> list:
    """
    SGD training loop with StepLR decay.
    Returns list of per-epoch dicts: epoch, train_loss, train_acc, test_loss, test_acc.
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    log.info("Training on %s | arch=Linear(%d→%d) | epochs=%d | lr=%s",
             DEVICE, INPUT_DIM, NUM_CLASSES, num_epochs, lr)

    epoch_log = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits  = model(images)
            loss    = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct      += (logits.argmax(dim=1) == labels).sum().item()
            total        += images.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total
        test_loss, test_acc = _evaluate_loader(model, test_loader, criterion)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "test_loss":  round(test_loss,  6),
            "test_acc":   round(test_acc,   6),
        }
        epoch_log.append(row)
        log.info("epoch %d/%d  train_loss=%.4f  train_acc=%.3f  "
                 "test_loss=%.4f  test_acc=%.3f  lr=%.6f",
                 epoch, num_epochs, train_loss, train_acc, test_loss, test_acc, current_lr)

    return epoch_log


def evaluate_plaintext(model: nn.Module, test_loader: DataLoader) -> tuple:
    """
    Timed forward pass over the full test set.
    Returns (accuracy: float, latency_ms: float).
    """
    model.eval()
    model.to(DEVICE)
    t_start = time.perf_counter()
    _, accuracy = _evaluate_loader(model, test_loader, nn.CrossEntropyLoss())
    latency_ms = (time.perf_counter() - t_start) * 1_000.0
    return accuracy, latency_ms


def save_model(model: nn.Module, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info("Model weights saved → %s", path)


def load_model(path: str = MODEL_PATH) -> LogisticRegressionMNIST:
    model = LogisticRegressionMNIST()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    log.info("Model weights loaded ← %s", path)
    return model


def save_log(epoch_log: list, path: str = LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_log)
    log.info("Training log saved → %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_loader, test_loader = get_data_loaders()

    model     = LogisticRegressionMNIST()
    epoch_log = train(model, train_loader, test_loader)
    save_model(model)
    save_log(epoch_log)

    accuracy, latency_ms = evaluate_plaintext(model, test_loader)
    log.info("Plaintext accuracy=%.4f  latency=%.1fms  (%.3f µs/sample)",
             accuracy, latency_ms, latency_ms / 10_000 * 1_000)

    W, b = get_model_weights(model)
    log.info("Weight matrix W shape=%s dtype=%s — ready for HE context", W.shape, W.dtype)


if __name__ == "__main__":
    main()
