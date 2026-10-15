"""
benchmarking.py — Full ε-grid search over the joint HE + DP noise pipeline.

Runs 7 privacy budgets × 100 images each; writes results to
results/joint_noise_benchmark_results.csv.

ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0}
Fixed: δ=1e-5, σ_c=0.1, N=100, B=1.0
"""

import csv
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

MODEL_PATH   = "./models/logistic_regression.pth"
DATA_DIR     = "./data"
RESULTS_DIR  = "./results"
RESULTS_CSV  = os.path.join(RESULTS_DIR, "joint_noise_benchmark_results.csv")

NUM_EVAL    = 100
INPUT_DIM   = 784
NUM_CLASSES = 10

POLY_MOD_DEGREE     = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 40]
GLOBAL_SCALE        = 2 ** 21

SIGMA_C        = 0.1
DELTA          = 1e-5
INPUT_L2_BOUND = 1.0
EPSILONS       = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

PLAINTEXT_ACC = 0.906
HE_ONLY_ACC   = 0.85

CSV_FIELDS = [
    "epsilon", "delta", "l2_sensitivity", "sigma_c",
    "sigma_dp_required", "sigma_extra_injected", "sigma_total",
    "free_dp_achieved",
    "accuracy", "accuracy_drop_vs_plaintext", "accuracy_drop_vs_he",
    "avg_latency_per_sample", "total_eval_time",
]


class LogisticRegressionMNIST(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def create_ckks_context() -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def load_model_weights(path: str = MODEL_PATH) -> tuple:
    model = LogisticRegressionMNIST()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    W = model.fc.weight.detach().cpu().numpy().astype(np.float64)
    b = model.fc.bias.detach().cpu().numpy().astype(np.float64)
    return W, b


def compute_l2_sensitivity(W: np.ndarray, B: float = 1.0) -> float:
    """Spectral norm of W × input L2 bound B."""
    return float(np.linalg.svd(W, compute_uv=False)[0]) * B


def calculate_unified_noise(
    epsilon: float, delta: float,
    sensitivity: float, sigma_c: float,
) -> dict:
    """σ_dp via Gaussian mechanism; σ_extra credits the CPAD noise already present."""
    sigma_dp    = (sensitivity * np.sqrt(2.0 * np.log(1.25 / delta))) / epsilon
    var_dp      = sigma_dp ** 2
    var_c       = sigma_c  ** 2
    var_extra   = max(0.0, var_dp - var_c)
    sigma_extra = np.sqrt(var_extra)
    sigma_total = np.sqrt(var_c + var_extra)
    return {
        "sigma_dp":    sigma_dp,
        "sigma_extra": sigma_extra,
        "sigma_total": sigma_total,
        "free_dp":     (var_c >= var_dp),
    }


def encrypted_inference(
    context: ts.Context,
    enc_image: ts.CKKSVector,
    W: np.ndarray,
    b: np.ndarray,
) -> list:
    enc_logits = []
    for i in range(NUM_CLASSES):
        enc_z  = enc_image.dot(W[i].tolist())
        enc_z += float(b[i])
        enc_logits.append(enc_z)
    return enc_logits


def apply_unified_noise(
    context: ts.Context,
    enc_logits: list,
    sigma_total: float,
    rng: np.random.Generator,
) -> list:
    noisy = []
    for enc_z in enc_logits:
        enc_noise = ts.ckks_vector(context, [float(rng.normal(0.0, sigma_total))])
        noisy.append(enc_z + enc_noise)
    return noisy


def decrypt_logits(enc_logits: list) -> np.ndarray:
    return np.array([enc_z.decrypt()[0] for enc_z in enc_logits])


def evaluate_epsilon(
    context:     ts.Context,
    W:           np.ndarray,
    b:           np.ndarray,
    images:      np.ndarray,
    labels:      np.ndarray,
    epsilon:     float,
    sensitivity: float,
    rng:         np.random.Generator,
    n:           int = NUM_EVAL,
) -> dict:
    """Runs n encrypted inferences at a single ε point. Returns CSV-ready metrics."""
    noise       = calculate_unified_noise(epsilon, DELTA, sensitivity, SIGMA_C)
    sigma_total = noise["sigma_total"]
    correct, latencies = 0, []
    t_wall = time.perf_counter()

    for idx in range(n):
        img   = images[idx].astype(np.float64).tolist()
        label = int(labels[idx])
        t0    = time.perf_counter()
        enc_img    = ts.ckks_vector(context, img)
        enc_logits = encrypted_inference(context, enc_img, W, b)
        enc_logits = apply_unified_noise(context, enc_logits, sigma_total, rng)
        logits     = decrypt_logits(enc_logits)
        latencies.append(time.perf_counter() - t0)
        if int(np.argmax(logits)) == label:
            correct += 1

    total_eval_time = time.perf_counter() - t_wall
    accuracy        = correct / n

    return {
        "epsilon":                    epsilon,
        "delta":                      DELTA,
        "l2_sensitivity":             round(sensitivity, 8),
        "sigma_c":                    SIGMA_C,
        "sigma_dp_required":          round(noise["sigma_dp"],    8),
        "sigma_extra_injected":       round(noise["sigma_extra"], 8),
        "sigma_total":                round(noise["sigma_total"], 8),
        "free_dp_achieved":           noise["free_dp"],
        "accuracy":                   round(accuracy, 6),
        "accuracy_drop_vs_plaintext": round(PLAINTEXT_ACC - accuracy, 6),
        "accuracy_drop_vs_he":        round(HE_ONLY_ACC   - accuracy, 6),
        "avg_latency_per_sample":     round(float(np.mean(latencies)), 6),
        "total_eval_time":            round(total_eval_time, 4),
        "_correct":    correct,
        "_n":          n,
        "_sigma_dp":   noise["sigma_dp"],
        "_sigma_extra": noise["sigma_extra"],
    }


def save_csv(rows: list, path: str = RESULTS_CSV) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Results saved → %s", path)


def main() -> list:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    log.info("ε-grid search: %s", EPSILONS)
    log.info("Fixed params: σ_c=%s  δ=%s  N=%d", SIGMA_C, DELTA, NUM_EVAL)

    log.info("Initializing CKKS context")
    context = create_ckks_context()
    W, b    = load_model_weights(MODEL_PATH)
    images  = np.load(os.path.join(DATA_DIR, "mnist_test_images.npy"))
    labels  = np.load(os.path.join(DATA_DIR, "mnist_test_labels.npy"))

    sensitivity = compute_l2_sensitivity(W, B=INPUT_L2_BOUND)
    log.info("Δ₂ (spectral norm) = %.6f", sensitivity)

    all_rows  = []
    total_eps = len(EPSILONS)
    for run_idx, eps in enumerate(EPSILONS, start=1):
        noise    = calculate_unified_noise(eps, DELTA, sensitivity, SIGMA_C)
        free_str = "FREE DP" if noise["free_dp"] else f"σ_extra={noise['sigma_extra']:.4f}"
        log.info("[%d/%d]  ε=%-5s  σ_total=%.4f  %s",
                 run_idx, total_eps, eps, noise["sigma_total"], free_str)

        rng = np.random.default_rng(seed=42 + run_idx * 7)
        row = evaluate_epsilon(
            context, W, b, images, labels,
            epsilon=eps, sensitivity=sensitivity, rng=rng, n=NUM_EVAL,
        )
        all_rows.append(row)
        log.info("         → acc=%.2f%%  avg_lat=%.4fs  total=%.1fs",
                 row["accuracy"] * 100, row["avg_latency_per_sample"],
                 row["total_eval_time"])

    save_csv(all_rows)

    free_eps = [r["epsilon"] for r in all_rows if r["free_dp_achieved"]]
    if free_eps:
        log.info("Free DP threshold: ε ≥ %s (σ_c=%.1f covers full DP requirement)",
                 min(free_eps), SIGMA_C)

    return all_rows


if __name__ == "__main__":
    main()
