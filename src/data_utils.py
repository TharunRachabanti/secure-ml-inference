"""
data_utils.py — MNIST data preparation for the secure inference benchmark.

Downloads MNIST via torchvision, flattens 28×28 images to 784-d arrays,
and exports a fixed 500-image subset as .npy files for TenSEAL encryption.

Outputs → ./data/
    mnist_test_images.npy   shape (500, 784), float32, [0, 1]
    mnist_test_labels.npy   shape (500,),     int64
"""

import logging
import os
import time

import numpy as np
import torch
from torchvision import datasets, transforms

log = logging.getLogger(__name__)

DATA_DIR    = "./data"
OUTPUT_DIR  = "./data"
SUBSET_SIZE = 500
RANDOM_SEED = 42
BATCH_SIZE  = 256


def verify_libraries() -> None:
    """Log versions of required libraries; warn on any missing optional deps."""
    libs = {
        "torch":       torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy":       np.__version__,
    }
    for name in ("tenseal", "pandas", "matplotlib"):
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            libs[name] = "NOT INSTALLED"

    for lib, ver in libs.items():
        log.info("%-15s %s", lib, ver)

    missing = [k for k, v in libs.items() if "NOT INSTALLED" in str(v)]
    if missing:
        log.warning("Missing optional deps: %s. Install with: pip install %s",
                    ", ".join(missing), " ".join(missing))


def build_transform() -> transforms.Compose:
    """ToTensor + flatten to (784,)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1)),
    ])


def download_mnist(root: str, transform) -> tuple:
    """Download (or reuse cached) MNIST; returns (train_dataset, test_dataset)."""
    log.info("Loading MNIST from '%s'", root)
    t0 = time.perf_counter()
    train_ds = datasets.MNIST(root=root, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    log.info("Loaded in %.2fs — train=%d  test=%d",
             time.perf_counter() - t0, len(train_ds), len(test_ds))
    return train_ds, test_ds


def extract_numpy_arrays(dataset, n: int, seed: int) -> tuple:
    """
    Randomly sample n examples and return (images, labels) as NumPy arrays.
    images: (n, 784) float32 — labels: (n,) int64
    """
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)

    # TODO: replace SubsetRandomSampler with a map-style dataset slice if
    # SUBSET_SIZE grows beyond ~1k, to avoid DataLoader overhead per item.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(indices.tolist()),
        num_workers=0,
    )

    image_list, label_list = [], []
    for imgs, labs in loader:
        image_list.append(imgs.numpy())
        label_list.append(labs.numpy())

    images = np.concatenate(image_list, axis=0)[:n]
    labels = np.concatenate(label_list, axis=0)[:n]
    return images.astype(np.float32), labels.astype(np.int64)


def save_arrays(images: np.ndarray, labels: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, "mnist_test_images.npy")
    lbl_path = os.path.join(out_dir, "mnist_test_labels.npy")
    np.save(img_path, images)
    np.save(lbl_path, labels)
    log.info("Saved %s  shape=%s", img_path, images.shape)
    log.info("Saved %s  shape=%s", lbl_path, labels.shape)


def sanity_check(images: np.ndarray, labels: np.ndarray) -> None:
    log.info("images shape=%s  range=[%.4f, %.4f]  mean=%.4f",
             images.shape, images.min(), images.max(), images.mean())
    unique, counts = np.unique(labels, return_counts=True)
    log.info("class distribution: %s", dict(zip(unique.tolist(), counts.tolist())))


def load_benchmark_data(data_dir: str = OUTPUT_DIR) -> tuple:
    """Load pre-saved .npy benchmark arrays from disk."""
    images = np.load(os.path.join(data_dir, "mnist_test_images.npy"))
    labels = np.load(os.path.join(data_dir, "mnist_test_labels.npy"))
    return images, labels


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    verify_libraries()

    transform           = build_transform()
    train_ds, test_ds   = download_mnist(root=DATA_DIR, transform=transform)

    log.info("Sampling %d images from test set (seed=%d)", SUBSET_SIZE, RANDOM_SEED)
    images, labels = extract_numpy_arrays(test_ds, n=SUBSET_SIZE, seed=RANDOM_SEED)
    save_arrays(images, labels, out_dir=OUTPUT_DIR)
    sanity_check(images, labels)

    log.info("Extracting full test set (10,000 images) for trainer.py eval")
    all_images, all_labels = extract_numpy_arrays(test_ds, n=len(test_ds), seed=RANDOM_SEED)
    np.save(os.path.join(OUTPUT_DIR, "mnist_full_test_images.npy"), all_images)
    np.save(os.path.join(OUTPUT_DIR, "mnist_full_test_labels.npy"), all_labels)
    log.info("Saved full test set: images=%s  labels=%s",
             all_images.shape, all_labels.shape)


if __name__ == "__main__":
    main()
