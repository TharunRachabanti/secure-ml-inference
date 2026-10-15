"""
run_pipeline.py — Orchestrates the full secure inference pipeline.

Stages:
  1  data_utils       Data prep & MNIST export
  2  trainer          Plaintext model training
  3  he_context       CKKS HE inference baseline
  4  noise_mechanisms IND-CPAD sweep + unified DP noise calibration
  5  benchmarking     ε-grid search (writes results CSV)
  6  visualization    Publication figures

Usage:
    python run_pipeline.py [--start <n>] [--end <n>]
"""

import argparse
import importlib
import logging
import sys

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

STAGES = {
    1: ("src.data_utils",       "Data Utilities"),
    2: ("src.trainer",          "Model Training"),
    3: ("src.he_context",       "HE Inference Baseline"),
    4: ("src.noise_mechanisms", "IND-CPAD + Unified Noise"),
    5: ("src.benchmarking",     "Privacy Budget Grid Search"),
    6: ("src.visualization",    "Figure Generation"),
}


def run_stage(stage_num: int) -> None:
    module_path, label = STAGES[stage_num]
    log.info("=== Stage %d: %s ===", stage_num, label)
    module = importlib.import_module(module_path)
    if hasattr(module, "main"):
        module.main()
    else:
        log.warning("%s has no main() — skipping.", module_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Secure ML Inference Pipeline")
    parser.add_argument("--start", type=int, default=1, choices=STAGES.keys(),
                        help="First stage to run (default: 1)")
    parser.add_argument("--end",   type=int, default=6, choices=STAGES.keys(),
                        help="Last stage to run  (default: 6)")
    args = parser.parse_args()

    if args.start > args.end:
        log.error("--start must be <= --end")
        sys.exit(1)

    for stage_num in range(args.start, args.end + 1):
        run_stage(stage_num)

    log.info("Pipeline complete. Stages %d–%d finished.", args.start, args.end)


if __name__ == "__main__":
    main()
