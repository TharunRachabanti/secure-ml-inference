# Secure ML Inference: Joint CKKS-DP Noise Calibration

Pipeline for evaluating the intersection of Approximate Homomorphic Encryption (CKKS) and Differential Privacy for ML inference. Implements a joint noise calibration framework that measures utility trade-offs when sequentially injecting IND-CPAD cryptographic noise (key-recovery defence) and algorithmic DP noise (model-extraction defence) into encrypted outputs.

## Repository Structure

```
secure-ml-inference/
  run_pipeline.py          # End-to-end orchestrator (stages 1-6)
  requirements.txt
  src/
    data_utils.py           # MNIST preprocessing and loader config
    model.py                # PyTorch logistic regression + weight extraction
    trainer.py              # Plaintext training and evaluation
    he_context.py           # TenSEAL CKKS context and homomorphic forward pass
    noise_mechanisms.py     # IND-CPAD injection + DP budget calibration
    benchmarking.py         # epsilon-grid search across privacy budgets
    visualization.py        # Publication figure generation
  results/
    joint_noise_benchmark_results.csv
  figures/
    fig1_accuracy_vs_epsilon.{png,pdf}
    fig2_noise_decomposition.{png,pdf}
    fig3_latency_comparison.{png,pdf}
```

## Requirements

- Python 3.9+
- Dependencies: `tenseal>=0.3.14`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `scipy`, `tqdm`

> Note: TenSEAL requires a C++17 build environment. Linux or WSL2 recommended.

## Usage

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline (stages 1-6)
python run_pipeline.py

# Run a subset of stages
python run_pipeline.py --start 3 --end 5
```

Pipeline stages:

| Stage | Module | Description |
|---|---|---|
| 1 | `data_utils` | Data prep and MNIST export |
| 2 | `trainer` | Plaintext model training |
| 3 | `he_context` | CKKS HE inference baseline |
| 4 | `noise_mechanisms` | IND-CPAD sweep + unified DP noise calibration |
| 5 | `benchmarking` | Privacy budget grid search (writes results CSV) |
| 6 | `visualization` | Publication figures |

## Key Results

The benchmark sweeps epsilon in `{0.1, 0.5, 1.0, 2.0, 5.0, 10.0}` with fixed delta = 1e-5. Outputs are written to `results/joint_noise_benchmark_results.csv` and visualised in `figures/`.

Core finding: the unified noise budget (`sigma_extra = sqrt(max(0, sigma_dp^2 - sigma_c^2))`) avoids double-counting HE and DP noise, preserving model accuracy without weakening the privacy guarantee.

## Citation

If you use this codebase in your research, cite as:

```
@misc{rachabanti2026secureinference,
  author = {Rachabanti, Tharun},
  title  = {Secure ML Inference: Joint CKKS-DP Noise Calibration},
  year   = {2026},
  url    = {https://github.com/TharunRachabanti/secure-ml-inference}
}
```

## License

MIT