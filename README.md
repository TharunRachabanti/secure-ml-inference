# Secure ML Inference: Joint Noise Calibration for CKKS + Differential Privacy

> **A reproducible benchmark pipeline** combining Homomorphic Encryption (CKKS via TenSEAL) and Differential Privacy to solve the Joint Noise Calibration Problem for secure ML inference.

---

## 🔬 Research Overview

This repository implements a novel secure inference pipeline that:
1. Trains a neural network on tabular data in the plaintext domain.
2. Runs inference under **CKKS Homomorphic Encryption** (IND-CPAD security).
3. Applies **Differential Privacy (DP) Gaussian noise** calibrated to analytical L₂ sensitivity.
4. Solves the **Joint Noise Budget Problem**: avoids double-counting HE and DP noise via additive variance decomposition.
5. Benchmarks accuracy and latency trade-offs across multiple privacy budgets (ε).

---

## 📁 Repository Structure

```
secure-ml-inference/
├── src/
│   ├── phase_1_data_preparation.py   # Dataset loading, normalisation, train/test split
│   ├── phase_2_model_training.py     # Plaintext PyTorch/sklearn model training
│   ├── phase_3_he_pipeline.py        # CKKS encryption context, HE inference baseline
│   ├── phase_4_ind_cpad.py           # IND-CPAD noise injection (σ_c = 0.1) pre-decryption
│   ├── phase_5_unified_noise.py      # Joint noise calibration, DP noise, benchmarking
│   ├── phase_6_generate_figures.py   # Publication-quality figure generation (.png + .pdf)
│   └── phase_6_grid_search.py        # Hyperparameter grid search over ε values
├── results/
│   └── joint_noise_benchmark_results.csv   # Raw benchmark data (accuracy, latency per ε)
├── figures/
│   ├── fig1_accuracy_vs_epsilon.{png,pdf}
│   ├── fig2_noise_decomposition.{png,pdf}
│   └── fig3_latency_comparison.{png,pdf}
├── run_pipeline.py       # Master orchestration script
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** `data/` and `models/` are excluded from version control (see `.gitignore`). The `results/` and `figures/` directories are **intentionally tracked** as proof of experiment reproducibility.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/secure-ml-inference.git
cd secure-ml-inference
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python run_pipeline.py
```

### 3. Run Specific Phases

```bash
python run_pipeline.py --start 4 --end 5   # Re-run noise calibration only
python run_pipeline.py --start 6 --end 6   # Regenerate figures only
```

---

## 🔑 Key Technical Contributions

### Global L₂ Sensitivity (Δ₂)
Computed analytically from the weight matrix spectral norm:
```python
delta_2 = np.linalg.norm(W, ord=2)
```

### IND-CPAD Noise (σ\_c)
A fixed Gaussian noise σ\_c = 0.1 is injected into each ciphertext **before decryption**, satisfying the Li & Micciancio (2022) IND-CPAD security definition.

### Unified Noise Budget
The DP noise σ\_DP is computed as the residual after accounting for the HE noise already present:
```python
sigma_extra = np.sqrt(max(0, sigma_dp**2 - sigma_c**2))
```
This **prevents double-counting** and yields a tighter, valid DP guarantee.

---

## 📊 Results

Key findings from `results/joint_noise_benchmark_results.csv`:
- Accuracy degrades gracefully as ε decreases (stronger privacy).
- The unified noise budget consistently outperforms naïve additive noise in accuracy.
- HE inference latency is dominated by bootstrapping, independent of ε.

---

## 📖 Citation

If you use this codebase, please cite:
```
[Thesis citation to be added upon publication]
```

---

## 🛡️ License

MIT License — see `LICENSE` for details.
