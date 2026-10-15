"""
noise_mechanisms.py — IND-CPAD flooding and joint DP noise calibration.

Implements both noise layers of the secure inference pipeline:
  1. IND-CPAD (Li & Micciancio 2022): Gaussian flooding noise σ_c injected
     into each ciphertext before decryption to prevent decryption-error
     side-channel attacks on CKKS.
  2. Unified DP budget: residual DP noise σ_extra = sqrt(max(0, σ_dp² - σ_c²))
     avoiding double-counting of cryptographic noise already present.

Public API
----------
apply_ind_cpad_noise(context, enc_logits, sigma_c, rng) -> list[CKKSVector]
compute_model_l2_sensitivity(W, images, input_l2_bound) -> dict
calculate_unified_noise(epsilon, delta, sensitivity, sigma_c) -> dict
apply_unified_noise(context, enc_logits, sigma_total, rng) -> list[CKKSVector]
evaluate_with_cpad(context, W, b, images, labels, sigma_c, rng, n) -> dict
evaluate_unified_pipeline(context, W, b, images, labels, budget, rng, n) -> dict
"""

import logging
import os
import sys
import time
import warnings

import numpy as np
import torch

try:
    import tenseal as ts
except ImportError:
    logging.critical("TenSEAL not installed. Run: pip install tenseal")
    sys.exit(1)

from src.model import INPUT_DIM, NUM_CLASSES, LogisticRegressionMNIST

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

MODEL_PATH  = "./models/logistic_regression.pth"
DATA_DIR    = "./data"
RESULTS_DIR = "./models"

NUM_EVAL = 100

POLY_MOD_DEGREE     = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 40]
GLOBAL_SCALE        = 2 ** 21

SIGMA_C_VALUES  = [0.001, 0.01, 0.05, 0.1, 0.5]
HE_BASELINE_ACC = 0.85

SIGMA_C      = 0.1
VAR_CPAD     = SIGMA_C ** 2
DELTA        = 1e-5
EPSILONS     = [0.5, 2.0, 10.0]

PLAINTEXT_ACC = 0.906
HE_ONLY_ACC   = 0.85


def create_ckks_context() -> "ts.Context":
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    log.info("CKKS context ready. poly_degree=%d  scale=2^21", POLY_MOD_DEGREE)
    return ctx


def load_model_weights(path: str = MODEL_PATH) -> tuple:
    model = LogisticRegressionMNIST()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    W = model.fc.weight.detach().cpu().numpy().astype(np.float64)
    b = model.fc.bias.detach().cpu().numpy().astype(np.float64)
    return W, b


def encrypted_inference(
    context:   "ts.Context",
    enc_image: "ts.CKKSVector",
    W:         np.ndarray,
    b:         np.ndarray,
) -> list:
    """Row-wise dot products: z_i = W[i]·x + b[i] for i in 0..9. Returns list of 10 CKKSVectors."""
    enc_logits = []
    for i in range(NUM_CLASSES):
        enc_z  = enc_image.dot(W[i].tolist())
        enc_z += float(b[i])
        enc_logits.append(enc_z)
    return enc_logits


def decrypt_logits(enc_logits: list) -> np.ndarray:
    return np.array([enc_z.decrypt()[0] for enc_z in enc_logits])


# ─── IND-CPAD ─────────────────────────────────────────────────────────────────

def apply_ind_cpad_noise(
    context:    "ts.Context",
    enc_logits: list,
    sigma_c:    float,
    rng:        np.random.Generator,
) -> list:
    """
    Adds n_i ~ N(0, σ_c²) to each encrypted logit scalar (homomorphic addition).

    σ_c must satisfy σ_c >> ||e_i||_∞ where e_i is the CKKS decryption noise
    (~2^-21 ≈ 4.77e-7 for these parameters). Even σ_c=0.001 clears that bar;
    the sweep here probes ML-utility loss, not cryptographic adequacy.
    """
    noisy = []
    for enc_z in enc_logits:
        noise_val = float(rng.normal(0.0, sigma_c))
        enc_noise = ts.ckks_vector(context, [noise_val])
        noisy.append(enc_z + enc_noise)
    return noisy


def evaluate_with_cpad(
    context: "ts.Context",
    W:       np.ndarray,
    b:       np.ndarray,
    images:  np.ndarray,
    labels:  np.ndarray,
    sigma_c: float,
    rng:     np.random.Generator,
    n:       int = NUM_EVAL,
) -> dict:
    """Run n encrypted inferences with IND-CPAD noise at σ_c. Returns aggregated metrics."""
    correct, latencies = 0, []
    for idx in range(n):
        img   = images[idx].astype(np.float64).tolist()
        label = int(labels[idx])
        t0    = time.perf_counter()
        enc_img    = ts.ckks_vector(context, img)
        enc_logits = encrypted_inference(context, enc_img, W, b)
        enc_logits = apply_ind_cpad_noise(context, enc_logits, sigma_c, rng)
        logits     = decrypt_logits(enc_logits)
        latencies.append(time.perf_counter() - t0)
        if int(np.argmax(logits)) == label:
            correct += 1

    accuracy = correct / n
    return {
        "sigma_c":       sigma_c,
        "n_images":      n,
        "correct":       correct,
        "accuracy":      accuracy,
        "acc_pct":       accuracy * 100,
        "drop_vs_he_pp": (accuracy - HE_BASELINE_ACC) * 100,
        "avg_latency_s": float(np.mean(latencies)),
        "min_latency_s": float(np.min(latencies)),
        "max_latency_s": float(np.max(latencies)),
        "std_latency_s": float(np.std(latencies)),
    }


# ─── Sensitivity ──────────────────────────────────────────────────────────────

def compute_model_l2_sensitivity(
    W:              np.ndarray,
    images:         np.ndarray | None = None,
    input_l2_bound: float = 1.0,
) -> dict:
    """
    Computes Δ₂(f) = ‖W‖_2 · B (spectral norm × input bound) for f(x) = Wx.

    # Note: spectral norm bound is tight here, but Frobenius could be tested
    # for larger layers where SVD becomes expensive — it's a looser but cheaper bound.

    Also reports Frobenius bound and empirical max ‖Wx‖/‖x‖ on test images if provided.
    """
    sv             = np.linalg.svd(W, compute_uv=False)
    spectral_norm  = float(sv[0])
    delta2_spectral = spectral_norm * input_l2_bound

    frob_norm     = float(np.linalg.norm(W, "fro"))
    delta2_frob   = frob_norm * input_l2_bound

    delta2_empirical = 0.0
    if images is not None:
        outputs          = images @ W.T
        ratios           = np.linalg.norm(outputs, axis=1) / np.maximum(
                               np.linalg.norm(images, axis=1), 1e-12)
        delta2_empirical = float(np.max(ratios)) * input_l2_bound

    return {
        "spectral_norm":    spectral_norm,
        "delta2_spectral":  delta2_spectral,
        "delta2_frobenius": delta2_frob,
        "delta2_empirical": delta2_empirical,
        "sensitivity_used": delta2_spectral,
    }


# ─── Unified noise budget ─────────────────────────────────────────────────────

def calculate_unified_noise(
    epsilon:     float,
    delta:       float,
    sensitivity: float,
    sigma_c:     float,
) -> dict:
    """
    Joint IND-CPAD + DP noise calibration via additive variance decomposition.

    σ_dp  = (Δ₂ · √(2 ln(1.25/δ))) / ε       (Gaussian mechanism)
    σ_extra = √(max(0, σ_dp² - σ_c²))          (residual after CPAD credit)

    "Free DP": when σ_c ≥ σ_dp, no extra noise is needed — the cryptographic
    flood already satisfies (ε,δ)-DP at zero additional utility cost.
    """
    sigma_dp    = (sensitivity * np.sqrt(2.0 * np.log(1.25 / delta))) / epsilon
    var_dp      = sigma_dp ** 2
    var_c       = sigma_c  ** 2
    var_extra   = max(0.0, var_dp - var_c)
    sigma_extra = np.sqrt(var_extra)
    sigma_total = np.sqrt(var_c + var_extra)

    return {
        "epsilon":               epsilon,
        "delta":                 delta,
        "sensitivity_delta2":    sensitivity,
        "sigma_dp_required":     sigma_dp,
        "var_dp_required":       var_dp,
        "sigma_c_fixed":         sigma_c,
        "var_c_fixed":           var_c,
        "sigma_extra_needed":    sigma_extra,
        "var_extra_needed":      var_extra,
        "total_effective_sigma": sigma_total,
        "total_effective_var":   var_c + var_extra,
        "free_dp_achieved":      (var_c >= var_dp),
    }


def apply_unified_noise(
    context:     "ts.Context",
    enc_logits:  list,
    sigma_total: float,
    rng:         np.random.Generator,
) -> list:
    """
    Injects N(0, σ_total²) per logit — mathematically equivalent to applying
    CPAD and DP noise separately, but avoids an extra HE level and scale drift.
    σ_total is precomputed by calculate_unified_noise().
    """
    noisy = []
    for enc_z in enc_logits:
        noise_val = float(rng.normal(0.0, sigma_total))
        enc_noise = ts.ckks_vector(context, [noise_val])
        noisy.append(enc_z + enc_noise)
    return noisy


def evaluate_unified_pipeline(
    context: "ts.Context",
    W:       np.ndarray,
    b:       np.ndarray,
    images:  np.ndarray,
    labels:  np.ndarray,
    budget:  dict,
    rng:     np.random.Generator,
    n:       int = NUM_EVAL,
) -> dict:
    """Run n inferences with the precomputed unified noise budget. Returns metrics dict."""
    sigma_total = budget["total_effective_sigma"]
    correct, latencies = 0, []

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

    accuracy = correct / n
    return {
        "epsilon":               budget["epsilon"],
        "delta":                 budget["delta"],
        "sigma_dp_required":     budget["sigma_dp_required"],
        "sigma_c_fixed":         budget["sigma_c_fixed"],
        "sigma_extra_needed":    budget["sigma_extra_needed"],
        "total_effective_sigma": sigma_total,
        "free_dp_achieved":      budget["free_dp_achieved"],
        "n_images":              n,
        "correct":               correct,
        "accuracy":              accuracy,
        "acc_pct":               accuracy * 100,
        "drop_vs_plaintext_pp":  (accuracy - PLAINTEXT_ACC) * 100,
        "drop_vs_he_only_pp":    (accuracy - HE_ONLY_ACC)   * 100,
        "avg_latency_s":         float(np.mean(latencies)),
        "min_latency_s":         float(np.min(latencies)),
        "max_latency_s":         float(np.max(latencies)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    context = create_ckks_context()
    W, b    = load_model_weights(MODEL_PATH)
    images  = np.load(os.path.join(DATA_DIR, "mnist_test_images.npy"))
    labels  = np.load(os.path.join(DATA_DIR, "mnist_test_labels.npy"))

    log.info("=== IND-CPAD sigma_c sweep ===")
    for run_idx, sigma_c in enumerate(SIGMA_C_VALUES, start=1):
        rng = np.random.default_rng(seed=42 + run_idx)
        row = evaluate_with_cpad(context, W, b, images, labels,
                                 sigma_c=sigma_c, rng=rng, n=NUM_EVAL)
        marker = "✔" if row["accuracy"] >= 0.80 else "✘"
        log.info("[%d/%d] sigma_c=%.3f  acc=%.2f%%  drop=%.2fpp  avg_lat=%.4fs  %s",
                 run_idx, len(SIGMA_C_VALUES),
                 sigma_c, row["acc_pct"], row["drop_vs_he_pp"],
                 row["avg_latency_s"], marker)

    log.info("=== Unified noise budget sweep ===")
    sens   = compute_model_l2_sensitivity(W, images=images)
    delta2 = sens["sensitivity_used"]
    log.info("Δ₂ (spectral norm) = %.6f", delta2)

    budgets = [calculate_unified_noise(eps, DELTA, delta2, SIGMA_C) for eps in EPSILONS]

    for run_idx, (eps, budget) in enumerate(zip(EPSILONS, budgets), start=1):
        rng    = np.random.default_rng(seed=42 + run_idx * 100)
        result = evaluate_unified_pipeline(context, W, b, images, labels,
                                           budget=budget, rng=rng, n=NUM_EVAL)
        free = "FREE DP ✔" if result["free_dp_achieved"] else f"σ_extra={budget['sigma_extra_needed']:.4f}"
        log.info("ε=%-5s  acc=%.2f%%  σ_total=%.4f  avg_lat=%.4fs  %s",
                 eps, result["acc_pct"], result["total_effective_sigma"],
                 result["avg_latency_s"], free)


if __name__ == "__main__":
    main()
