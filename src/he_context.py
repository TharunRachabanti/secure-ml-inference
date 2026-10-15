"""
he_context.py — CKKS homomorphic encryption baseline (HE-only inference).

Ports the plaintext linear forward pass into the encrypted domain via TenSEAL.
Establishes HE approximation-error and compute-overhead baselines before noise
injection in noise_mechanisms.py.

CKKS parameters:
    poly_modulus_degree = 8192   → 4096 plaintext slots
    coeff_mod_bit_sizes = [40, 21, 21, 40]  (2 multiplication levels)
    global_scale        = 2^21  (~6 decimal places of fractional precision)
"""

import logging
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

try:
    import tenseal as ts
except ImportError:
    logging.critical("TenSEAL not installed. Run: pip install tenseal")
    sys.exit(1)

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

MODEL_PATH  = "./models/logistic_regression.pth"
DATA_DIR    = "./data"
NUM_EVAL    = 100
INPUT_DIM   = 784
NUM_CLASSES = 10

POLY_MOD_DEGREE     = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 40]
GLOBAL_SCALE        = 2 ** 21


class LogisticRegressionMNIST(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def create_ckks_context() -> ts.Context:
    """
    Initialise and return a TenSEAL CKKS context.

    Using row-wise dot products (one dot per class) rather than a batched
    matrix multiply because TenSEAL can't repack 10 independent encrypted
    scalars into a single CKKSVector without Galois rotations that would
    exhaust our two-level budget.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    log.info("CKKS context initialized. poly_degree=%d  scale=2^21", POLY_MOD_DEGREE)
    return ctx


def load_model_weights(model_path: str = MODEL_PATH) -> tuple:
    """Load trained weights; returns (W: float64 (10,784), b: float64 (10,))."""
    model = LogisticRegressionMNIST()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    W = model.fc.weight.detach().cpu().numpy().astype(np.float64)
    b = model.fc.bias.detach().cpu().numpy().astype(np.float64)
    log.info("Loaded weights from '%s'  W=%s  b=%s", model_path, W.shape, b.shape)
    return W, b


def encrypted_inference(
    context:   ts.Context,
    enc_image: ts.CKKSVector,
    W:         np.ndarray,
    b:         np.ndarray,
) -> list:
    """
    Executes the CKKS encrypted forward pass: z_i = W[i]·x + b[i] for i in 0..9.
    Returns a list of 10 CKKSVector scalars (one per class).

    Each dot() call consumes one multiplication level + rescale.
    Bias addition is plaintext-side and costs no HE levels.

    # TODO: Implement multi-threading for batch encryption if dataset scales
    # beyond 100 images — each image inference is fully independent.
    """
    enc_logits = []
    for i in range(NUM_CLASSES):
        enc_z  = enc_image.dot(W[i].tolist())
        enc_z += float(b[i])
        enc_logits.append(enc_z)
    return enc_logits


def decrypt_logits(enc_logits: list) -> np.ndarray:
    """Decrypt 10 CKKSVector scalars → (10,) numpy array."""
    return np.array([enc_z.decrypt()[0] for enc_z in enc_logits])


def run_he_evaluation(
    context: ts.Context,
    W:       np.ndarray,
    b:       np.ndarray,
    images:  np.ndarray,
    labels:  np.ndarray,
    n:       int = NUM_EVAL,
) -> dict:
    """
    Encrypt n images, run encrypted_inference, decrypt, compare argmax to labels.
    Returns accuracy and latency statistics.
    """
    images, labels = images[:n], labels[:n]
    correct, latencies, predictions = 0, [], []

    log.info("Starting HE evaluation on %d images", n)

    for idx in range(n):
        img   = images[idx].astype(np.float64).tolist()
        label = int(labels[idx])

        t_start     = time.perf_counter()
        enc_img     = ts.ckks_vector(context, img)
        enc_logits  = encrypted_inference(context, enc_img, W, b)
        logits      = decrypt_logits(enc_logits)
        pred        = int(np.argmax(logits))
        latency     = time.perf_counter() - t_start

        latencies.append(latency)
        predictions.append(pred)
        if pred == label:
            correct += 1

        if (idx + 1) % 10 == 0:
            log.info("[%3d/%d]  running_acc=%.1f%%  avg_lat=%.3fs",
                     idx + 1, n, correct / (idx + 1) * 100, np.mean(latencies))

    accuracy = correct / n
    return {
        "he_accuracy":     accuracy,
        "avg_latency_s":   float(np.mean(latencies)),
        "min_latency_s":   float(np.min(latencies)),
        "max_latency_s":   float(np.max(latencies)),
        "std_latency_s":   float(np.std(latencies)),
        "total_latency_s": float(np.sum(latencies)),
        "n_images":        n,
        "correct":         correct,
        "predictions":     predictions,
        "true_labels":     labels.tolist(),
    }


def print_results(results: dict, plaintext_accuracy: float = 0.906) -> None:
    delta = (results["he_accuracy"] - plaintext_accuracy) * 100
    log.info("HE-only accuracy=%.2f%%  (plaintext=%.2f%%  drop=%.2fpp)",
             results["he_accuracy"] * 100, plaintext_accuracy * 100, delta)
    log.info("Latency: avg=%.4fs  min=%.4fs  max=%.4fs  std=%.4fs  total=%.2fs",
             results["avg_latency_s"], results["min_latency_s"],
             results["max_latency_s"], results["std_latency_s"],
             results["total_latency_s"])


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    context  = create_ckks_context()
    W, b     = load_model_weights(MODEL_PATH)

    images_path = os.path.join(DATA_DIR, "mnist_test_images.npy")
    if not os.path.exists(images_path):
        log.error("Benchmark data not found at '%s'. Run data_utils.py first.", images_path)
        sys.exit(1)

    images = np.load(images_path)
    labels = np.load(os.path.join(DATA_DIR, "mnist_test_labels.npy"))
    log.info("Benchmark subset loaded: %s", images.shape)

    results = run_he_evaluation(context, W, b, images, labels, n=NUM_EVAL)
    print_results(results)
    log.info("HE baseline verified. Proceed to noise_mechanisms.py for IND-CPAD sweep.")
    return results


if __name__ == "__main__":
    main()
