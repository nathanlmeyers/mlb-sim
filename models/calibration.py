"""Probability calibration for the ensemble model.

Supports isotonic regression (primary) and logistic recalibration (fallback).
Ensures predicted probabilities match observed frequencies.
"""

from dataclasses import dataclass
import pickle
from pathlib import Path
import numpy as np


@dataclass
class CalibrationResult:
    """Result of applying calibration to a probability."""
    raw: float
    calibrated: float
    method: str  # "isotonic", "logistic", or "none"


class ProbabilityCalibrator:
    """Calibrates raw predicted probabilities to match observed frequencies.

    Primary method: isotonic regression (non-parametric)
    Fallback: logistic recalibration (parametric)
    """

    def __init__(self):
        self._isotonic = None
        self._logistic_a = 1.0
        self._logistic_b = 0.0
        self._fitted = False

    def fit(self, predictions: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit the calibrator on historical predictions and outcomes.

        Args:
            predictions: Array of predicted probabilities (0 to 1)
            outcomes: Array of binary outcomes (0 or 1)
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            self._isotonic = IsotonicRegression(
                y_min=0.02, y_max=0.98, out_of_bounds="clip"
            )
            self._isotonic.fit(predictions, outcomes)
            self._fitted = True
        except ImportError:
            # sklearn not available, fit logistic parameters manually
            self._fit_logistic(predictions, outcomes)
            self._fitted = True

    def _fit_logistic(self, predictions: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit logistic calibration parameters via grid search."""
        best_brier = float("inf")

        for a in [0.8, 1.0, 1.2, 1.5, 2.0]:
            for b in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                calibrated = self._apply_logistic(predictions, a, b)
                brier = float(np.mean((calibrated - outcomes) ** 2))
                if brier < best_brier:
                    best_brier = brier
                    self._logistic_a = a
                    self._logistic_b = b

    def _apply_logistic(self, probs: np.ndarray, a: float, b: float) -> np.ndarray:
        """Apply logistic calibration."""
        probs = np.clip(probs, 0.01, 0.99)
        logit = np.log(probs / (1 - probs))
        adjusted = a * logit + b
        return 1.0 / (1.0 + np.exp(-adjusted))

    def predict(self, raw_prob: float) -> CalibrationResult:
        """Calibrate a single probability."""
        if not self._fitted:
            return CalibrationResult(raw=raw_prob, calibrated=raw_prob, method="none")

        if self._isotonic is not None:
            calibrated = float(self._isotonic.predict([raw_prob])[0])
            return CalibrationResult(raw=raw_prob, calibrated=calibrated, method="isotonic")
        else:
            arr = np.array([raw_prob])
            calibrated = float(self._apply_logistic(arr, self._logistic_a, self._logistic_b)[0])
            return CalibrationResult(raw=raw_prob, calibrated=calibrated, method="logistic")

    def predict_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate an array of probabilities."""
        if not self._fitted:
            return raw_probs

        if self._isotonic is not None:
            return self._isotonic.predict(raw_probs)
        else:
            return self._apply_logistic(raw_probs, self._logistic_a, self._logistic_b)

    def save(self, path: str) -> None:
        """Save fitted calibrator to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "isotonic": self._isotonic,
                "logistic_a": self._logistic_a,
                "logistic_b": self._logistic_b,
                "fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ProbabilityCalibrator":
        """Load a fitted calibrator from disk."""
        cal = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        cal._isotonic = data["isotonic"]
        cal._logistic_a = data["logistic_a"]
        cal._logistic_b = data["logistic_b"]
        cal._fitted = data["fitted"]
        return cal


def logistic_calibrate(raw_prob: float, a: float, b: float) -> float:
    """Simple logistic recalibration (standalone function)."""
    raw_prob = np.clip(raw_prob, 0.01, 0.99)
    logit = np.log(raw_prob / (1 - raw_prob))
    adjusted = a * logit + b
    return float(1.0 / (1.0 + np.exp(-adjusted)))
