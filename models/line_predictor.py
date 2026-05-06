"""Predict the de-vigged closing line — not the game outcome.

Why this exists
---------------
The walk-forward backtest established that our sim model cannot beat MLB
closing lines on outcomes (49.8% accuracy vs 52.1% always-home baseline).
That's the wrong target. Outcomes are too noisy: a single game is a
Bernoulli(p) sample, so even a perfect model gets ~50% Brier credit at
best on close games.

A closing-line predictor changes the target:
  outcome model:  features → P(home wins)         loss vs binary outcome
  line model:     features → P(home wins per DK closing)  loss vs continuous prob

Closing lines aggregate sharp money's view, so they're a much smoother
target with far less variance per sample. Predicting where the line *will*
close (or decoding the latent "fair price") is the standard sharp-shop
approach. If our features have any information beyond what's in the
current Kalshi mid, this architecture will surface it. If they don't, the
project is conclusively dead.

Trading interpretation
----------------------
At inference time:
  1. Build features for tonight's game.
  2. predictor.predict() → fair_home_prob (the line we expect DK to close at).
  3. If Kalshi current mid is far from fair_home_prob, bet the side our
     prediction prefers; close before DK actually closes (or hold to settle).

The bet doesn't require the model to be "right" about the *outcome* — only
to be more right than Kalshi about where DK will close. That's a much
weaker condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Names of features in stable order (matters for sklearn — first column is
# treated identically to last). New features should be appended.
FEATURE_NAMES = [
    "sim_home_wp",          # what our box-score sim said
    "sim_total",            # predicted total runs
    "sim_home_cover",       # spread coverage prob
    "home_rpg",             # team runs per game (current season)
    "away_rpg",
    "home_rapg",            # runs allowed per game
    "away_rapg",
    "home_winpct",          # season win pct
    "away_winpct",
    "home_sp_k_pct",        # starter K rate
    "away_sp_k_pct",
    "home_sp_bb_pct",
    "away_sp_bb_pct",
    "home_sp_hr_per_fb",
    "away_sp_hr_per_fb",
    "park_factor",          # overall offense park factor
    "park_hr_factor",       # HR-specific
    "is_dome",              # not yet wired; placeholder 0
    "rpg_diff",             # home_rpg - away_rpg
    "winpct_diff",          # home_winpct - away_winpct
    "sp_k_diff",            # home_sp_k_pct - away_sp_k_pct
]


def extract_features(
    sim_pred: dict,
    home_team_model,
    away_team_model,
    home_pitcher,
    away_pitcher,
    park,
) -> dict:
    """Build the feature dict for one game.

    Args mirror the inputs the sim already had to construct, so this can
    be called from inside the walk-forward loop with no extra fetching.
    Missing values default to neutral (league-average for rates, 1.0 for
    park factors, 0 for dummies).

    Returns a dict in canonical order; convert to vector via to_vector().
    """
    f = {
        "sim_home_wp":      float(sim_pred.get("model_home_wp", 0.5)),
        "sim_total":        float(sim_pred.get("pred_total", 8.5)),
        "sim_home_cover":   float(sim_pred.get("home_cover_pct", 0.5)),
        "home_rpg":         float(getattr(home_team_model, "runs_per_game", 4.39)),
        "away_rpg":         float(getattr(away_team_model, "runs_per_game", 4.39)),
        "home_rapg":        float(getattr(home_team_model, "runs_allowed_per_game", 4.39)),
        "away_rapg":        float(getattr(away_team_model, "runs_allowed_per_game", 4.39)),
        "home_winpct":      float(getattr(home_team_model, "win_pct", 0.5)),
        "away_winpct":      float(getattr(away_team_model, "win_pct", 0.5)),
        "home_sp_k_pct":    float(getattr(home_pitcher, "k_pct", 0.226)),
        "away_sp_k_pct":    float(getattr(away_pitcher, "k_pct", 0.226)),
        "home_sp_bb_pct":   float(getattr(home_pitcher, "bb_pct", 0.082)),
        "away_sp_bb_pct":   float(getattr(away_pitcher, "bb_pct", 0.082)),
        "home_sp_hr_per_fb": float(getattr(home_pitcher, "hr_per_fb", 0.116)),
        "away_sp_hr_per_fb": float(getattr(away_pitcher, "hr_per_fb", 0.116)),
        "park_factor":      float(getattr(park, "overall_factor", 1.0)),
        "park_hr_factor":   float(getattr(park, "hr_factor", 1.0)),
        "is_dome":          0.0,  # TODO: wire up dome detection
    }
    f["rpg_diff"]    = f["home_rpg"]    - f["away_rpg"]
    f["winpct_diff"] = f["home_winpct"] - f["away_winpct"]
    f["sp_k_diff"]   = f["home_sp_k_pct"] - f["away_sp_k_pct"]
    return f


def to_vector(features: dict) -> np.ndarray:
    """Convert a feature dict to a numpy vector in canonical order."""
    return np.array([features[k] for k in FEATURE_NAMES], dtype=float)


def to_matrix(rows: Iterable[dict]) -> np.ndarray:
    return np.array([to_vector(r) for r in rows], dtype=float)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

@dataclass
class PredictorResult:
    """Score breakdown for one fitted model on a held-out set."""
    name: str
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float
    pred: np.ndarray = field(default_factory=lambda: np.array([]))


class LinePredictor:
    """Predicts de-vigged closing-line home win probability.

    Default backend is HistGradientBoostingRegressor — fast on small data,
    handles non-linearity and feature interactions, requires no scaling.
    A ridge baseline and a "sim raw" baseline are computed alongside in
    `evaluate_baselines()` so the GBT can be compared against trivially.
    """

    def __init__(self, max_depth: int = 4, learning_rate: float = 0.05,
                 max_iter: int = 200, l2_regularization: float = 1.0,
                 random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinePredictor":
        # Clip y to avoid the regressor learning to predict outside [0, 1]
        y = np.clip(y, 0.02, 0.98)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X), 0.02, 0.98)


def evaluate_baselines(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       sim_idx: int = 0) -> list[PredictorResult]:
    """Fit and score the GBT plus 3 baselines on a single train/test split.

    Baselines:
      always_50:  predict 0.5 for everything → variance-of-y score floor
      sim_raw:    use the sim model's raw output (X_train[:, sim_idx])
                  → measures whether the *current* architecture has any
                  predictive value vs the closing line
      ridge:      linear baseline

    Returns one PredictorResult per model.
    """
    results = []

    def score(name: str, pred: np.ndarray, n_train: int) -> PredictorResult:
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
        r2 = r2_score(y_test, pred) if len(set(y_test.tolist())) > 1 else float("nan")
        return PredictorResult(name=name, n_train=n_train, n_test=len(y_test),
                               mae=mae, rmse=rmse, r2=r2, pred=pred)

    # always 0.5
    results.append(score("always_50", np.full_like(y_test, 0.5), 0))

    # sim raw
    sim_pred = X_test[:, sim_idx]
    results.append(score("sim_raw", sim_pred, 0))

    # ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, np.clip(y_train, 0.02, 0.98))
    results.append(score("ridge",
                          np.clip(ridge.predict(X_test), 0.02, 0.98),
                          len(y_train)))

    # GBT
    gbt = LinePredictor()
    gbt.fit(X_train, y_train)
    results.append(score("gbt",
                          gbt.predict(X_test),
                          len(y_train)))

    return results


# ---------------------------------------------------------------------------
# Trading sim helper (synthetic noisy Kalshi proxy)
# ---------------------------------------------------------------------------

def simulate_line_trades(
    pred_close: np.ndarray,
    actual_close: np.ndarray,
    actual_outcome: np.ndarray,    # 1 = home won, 0 = away won
    kalshi_noise_std: float = 0.025,
    threshold: float = 0.03,
    seed: int = 11,
) -> dict:
    """Simulate trades against a synthetic Kalshi mid drawn as
    `actual_close + N(0, noise_std)`.

    This is NOT a real backtest — we don't have historical Kalshi mids for
    these games. It answers a different question: *if* Kalshi mids were
    noisy versions of the eventual DK close, would our prediction be more
    accurate than the noisy mid often enough to make money?

    Returns dict with bets, ROI, and stats.

    Strategy: bet HOME if pred_close - mid >= threshold;
              bet AWAY if mid - pred_close >= threshold.
    Settle at actual outcome. Assumes -110/-110 effective vig (decimal 1.909).
    """
    rng = np.random.default_rng(seed)
    mid = np.clip(actual_close + rng.normal(0, kalshi_noise_std, size=len(actual_close)),
                  0.05, 0.95)

    bets = []
    for i in range(len(pred_close)):
        edge_home = pred_close[i] - mid[i]
        if edge_home >= threshold:
            side, won = "HOME", actual_outcome[i] == 1
            our_price = mid[i]   # we buy at mid
        elif -edge_home >= threshold:
            side, won = "AWAY", actual_outcome[i] == 0
            our_price = 1 - mid[i]
        else:
            continue
        # Decimal odds at our buy price (ignoring vig — Kalshi is the venue)
        dec = 1.0 / our_price
        profit = (dec - 1.0) if won else -1.0  # 1-unit stake
        bets.append({"side": side, "price": float(our_price),
                     "won": bool(won), "profit": float(profit),
                     "edge": float(abs(edge_home))})

    if not bets:
        return {"n": 0, "wins": 0, "roi": float("nan"),
                "mean_edge": float("nan"), "bets": []}
    wins = sum(1 for b in bets if b["won"])
    profit = sum(b["profit"] for b in bets)
    return {
        "n": len(bets),
        "wins": wins,
        "win_rate": wins / len(bets),
        "roi": profit / len(bets),
        "mean_edge": float(np.mean([b["edge"] for b in bets])),
        "bets": bets,
    }
