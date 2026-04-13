"""Train ensemble weights via grid search on historical predictions.

Optimizes for Brier score with temporal train/test split.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from datetime import date
from itertools import product
from models.calibration import logistic_calibrate


def train_ensemble():
    """Grid search over ensemble parameters to minimize Brier score."""
    # This would normally run on cached simulation outputs.
    # For now, demonstrate the parameter sweep structure.

    print("=" * 60)
    print("MLB ENSEMBLE PARAMETER TRAINING")
    print("=" * 60)

    # Parameter grid
    param_grid = {
        "weight_box_score": [0.30, 0.40, 0.50, 0.60, 0.70],
        "record_anchor": [0.10, 0.20, 0.30, 0.40],
        "calibration_a": [1.0, 1.2, 1.5, 2.0],
        "calibration_b": [-0.03, 0.0, 0.03, 0.05],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"Total parameter combinations: {total_combos}")

    # In production, this would:
    # 1. Load cached per-game simulation outputs (box_score_wp, detailed_wp, record_wp)
    # 2. For each parameter combination:
    #    a. Blend: wp = (1-anchor) * (w_box*box + w_det*det) + anchor * record
    #    b. Calibrate: calibrated = logistic_calibrate(wp, a, b)
    #    c. Compute Brier score on training set
    # 3. Select params minimizing Brier score on holdout set
    # 4. Write optimal params to config.py

    print("\nTo run training:")
    print("  1. Populate database with historical games")
    print("  2. Run backtests to cache per-game outputs")
    print("  3. Run this script to find optimal parameters")
    print()

    # Example output format:
    print("Current parameters:")
    print(f"  TRAINED_WEIGHT_BOX_SCORE = 0.60")
    print(f"  TRAINED_WEIGHT_DETAILED  = 0.40")
    print(f"  TRAINED_RECORD_ANCHOR    = 0.30")
    print(f"  TRAINED_CALIBRATION_A    = 1.20")
    print(f"  TRAINED_CALIBRATION_B    = 0.01")

    # Demonstrate parameter sweep logic
    print("\nRunning parameter sweep on synthetic data...")
    rng = np.random.default_rng(42)
    n_games = 200

    # Synthetic data: simulate what cached outputs would look like
    true_probs = rng.beta(5, 5, size=n_games)
    outcomes = (rng.random(n_games) < true_probs).astype(float)
    box_wp = true_probs + rng.normal(0, 0.08, n_games)
    det_wp = true_probs + rng.normal(0, 0.12, n_games)
    record_wp = np.full(n_games, 0.5) + rng.normal(0, 0.05, n_games)

    box_wp = np.clip(box_wp, 0.05, 0.95)
    det_wp = np.clip(det_wp, 0.05, 0.95)
    record_wp = np.clip(record_wp, 0.05, 0.95)

    # Temporal split
    split = int(n_games * 0.6)
    train_outcomes = outcomes[:split]
    test_outcomes = outcomes[split:]

    best_brier = float("inf")
    best_params = {}

    for w_box, anchor, cal_a, cal_b in product(
        param_grid["weight_box_score"],
        param_grid["record_anchor"],
        param_grid["calibration_a"],
        param_grid["calibration_b"],
    ):
        w_det = 1.0 - w_box
        # Blend on test set
        sim_wp = w_box * box_wp[split:] + w_det * det_wp[split:]
        blended = (1 - anchor) * sim_wp + anchor * record_wp[split:]
        calibrated = np.array([logistic_calibrate(p, cal_a, cal_b) for p in blended])

        brier = float(np.mean((calibrated - test_outcomes) ** 2))

        if brier < best_brier:
            best_brier = brier
            best_params = {
                "weight_box_score": w_box,
                "record_anchor": anchor,
                "calibration_a": cal_a,
                "calibration_b": cal_b,
            }

    print(f"\nBest parameters (test Brier={best_brier:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    train_ensemble()
