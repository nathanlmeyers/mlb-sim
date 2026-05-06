"""Walk-forward training and evaluation for the closing-line predictor.

Builds a per-game feature row for every game with a known DK closing
line, then runs an honest walk-forward CV: train on pre-cutoff games,
evaluate on post-cutoff. Compares against three baselines:

  always_50   constant 0.5  (variance floor)
  sim_raw     uses our existing sim's raw home_wp as the prediction
              (this is the critical baseline — if GBT doesn't beat
               sim_raw, the sim already captured everything useful)
  ridge       linear baseline
  gbt         our actual model

Reports MAE, RMSE, R² per fold and pooled across folds.

Then runs a synthetic trade simulation: assume Kalshi mid is the actual
DK closing line plus Gaussian noise; bet whenever predicted close is
threshold% off the noisy mid; settle at game outcome. This is illustrative,
not a real backtest — we don't have historical Kalshi mids. But it
characterizes the trading sensitivity of the predictor.

Usage:
    python scripts/train_line_predictor.py
    python scripts/train_line_predictor.py --cutoffs 2026-04-08,2026-04-12,2026-04-16
    python scripts/train_line_predictor.py --noise-std 0.04 --threshold 0.04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from backtest.json_backtest import (
    load_backtest_inputs,
    simulate_predictions_with_params,
    build_team_model_from_json,
    build_pitcher_from_json,
    TEAM_ABBR,
)
from betting.ev import remove_vig
from models.line_predictor import (
    FEATURE_NAMES,
    extract_features,
    to_matrix,
    LinePredictor,
    evaluate_baselines,
    simulate_line_trades,
)
from scripts.walk_forward_backtest import load_espn_odds, attach_espn_odds


DEFAULT_CUTOFFS = ["2026-04-05", "2026-04-08", "2026-04-12", "2026-04-16"]

# Load all available ESPN closing-line files. As of 2026-04-20, ESPN's
# pickcenter only returns data for the current season (2025 backfill returned
# 0 games), so this is currently a 1-element list. Add additional sources
# (sportsbookreview, action-network scrape, etc.) here as they're collected.
DEFAULT_ODDS_FILES = [
    ".context/espn_odds_2025.json",  # currently empty, kept for forward-compat
    ".context/espn_odds_2026.json",
]


def load_odds_combined(paths: list[str]) -> dict:
    """Merge multiple ESPN-style odds files into one (date, home, away) → row index."""
    combined: dict = {}
    for path in paths:
        idx = load_espn_odds(path)
        if idx:
            combined.update(idx)
    return combined


def build_dataset(inputs: dict, odds_idx: dict, n_sims: int = 1500,
                  prior_weight: float = 0.5, reg_n: int = 20) -> tuple:
    """Build (features, target, meta) for every game with a closing line.

    Target is the de-vigged DK closing home win prob. We re-build sim
    outputs so they appear as features alongside the raw stats — this
    captures whatever signal the sim engine extracted, then lets the
    GBT decide whether it's useful or redundant.
    """
    all_games = inputs["all_games"]
    pitcher_index = inputs["pitcher_index"]
    prior_pitcher_index = inputs["prior_pitcher_index"]
    starter_lookup = inputs["starter_lookup"]
    park_factors = inputs["park_factors"]
    neutral_park = inputs["neutral_park"]
    league_avg_pitcher = inputs["league_avg_pitcher"]

    # Filter to games with closing lines (saves sim time)
    target_games = []
    for g in all_games:
        if not g.get("home_id") or not g.get("away_id"):
            continue
        home_abbr = TEAM_ABBR.get(g["home_team"])
        away_abbr = TEAM_ABBR.get(g["away_team"])
        rec = odds_idx.get((g["game_date"], home_abbr, away_abbr))
        if rec is None:
            continue
        ml_h = rec["odds"].get("ml_home")
        ml_a = rec["odds"].get("ml_away")
        if ml_h is None or ml_a is None:
            continue
        target_games.append(g)

    print(f"  {len(target_games)} games with usable closing lines")

    # Run sim once (with default pitcher hyperparams — we're not tuning these)
    sim_preds = simulate_predictions_with_params(
        inputs, target_games,
        prior_weight=prior_weight, reg_n=reg_n,
        n_sims=n_sims, seed=42,
    )
    print(f"  Generated {len(sim_preds)} sim predictions")

    # For each sim prediction, rebuild the input objects so we can extract
    # features (the sim_pred dict alone doesn't have team/pitcher attribute access)
    sim_by_id = {p["game_id"]: p for p in sim_preds}

    features = []
    targets = []
    meta = []
    rng = np.random.default_rng(0)

    for g in target_games:
        sp = sim_by_id.get(g["game_id"])
        if sp is None:
            continue

        home_team = build_team_model_from_json(
            g["home_id"], g["home_team"], all_games, g["game_date"], None,
        )
        away_team = build_team_model_from_json(
            g["away_id"], g["away_team"], all_games, g["game_date"], None,
        )
        hs_name, as_name = starter_lookup.get(g["game_id"], ("", ""))
        home_sp = league_avg_pitcher
        away_sp = league_avg_pitcher
        if hs_name:
            ps = build_pitcher_from_json(hs_name, pitcher_index, g["game_date"],
                                          prior_pitcher_index,
                                          prior_weight=prior_weight, reg_n=reg_n)
            if ps:
                home_sp = ps
        if as_name:
            ps = build_pitcher_from_json(as_name, pitcher_index, g["game_date"],
                                          prior_pitcher_index,
                                          prior_weight=prior_weight, reg_n=reg_n)
            if ps:
                away_sp = ps
        park = park_factors.get(TEAM_ABBR.get(g["home_team"], ""), neutral_park)

        sim_pred_extended = {
            "model_home_wp": sp["model_home_wp"],
            "pred_total":    sp["pred_total"],
            "home_cover_pct": 0.5,  # not in simulate_predictions output; placeholder
        }
        feats = extract_features(sim_pred_extended, home_team, away_team,
                                  home_sp, away_sp, park)

        # Target: de-vigged home prob from DK closing
        home_abbr = TEAM_ABBR.get(g["home_team"])
        away_abbr = TEAM_ABBR.get(g["away_team"])
        rec = odds_idx[(g["game_date"], home_abbr, away_abbr)]
        ml_h = rec["odds"]["ml_home"]
        ml_a = rec["odds"]["ml_away"]
        true_h, _ = remove_vig(ml_h, ml_a)

        features.append(feats)
        targets.append(true_h)
        meta.append({
            "game_id": g["game_id"],
            "date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "actual_home_win": int(g["home_score"] > g["away_score"]),
            "ml_home": ml_h,
            "ml_away": ml_a,
        })

    X = to_matrix(features)
    y = np.array(targets, dtype=float)
    return X, y, meta


def walk_forward(X: np.ndarray, y: np.ndarray, meta: list[dict],
                 cutoffs: list[str]) -> list[dict]:
    """Run walk-forward folds. Returns per-fold results."""
    folds = []
    for cutoff in cutoffs:
        train_mask = np.array([m["date"] < cutoff for m in meta])
        test_mask = ~train_mask
        n_train, n_test = train_mask.sum(), test_mask.sum()
        if n_train < 30 or n_test < 10:
            print(f"  [skip] cutoff {cutoff}: train={n_train} test={n_test}")
            continue

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        meta_te = [m for i, m in enumerate(meta) if test_mask[i]]

        results = evaluate_baselines(X_tr, y_tr, X_te, y_te,
                                     sim_idx=FEATURE_NAMES.index("sim_home_wp"))

        print()
        print(f"  fold cutoff={cutoff}  train={n_train}  test={n_test}")
        print(f"    {'model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        print(f"    {'-'*40}")
        for r in results:
            print(f"    {r.name:<12} {r.mae:>8.4f} {r.rmse:>8.4f} {r.r2:>+8.3f}")

        folds.append({
            "cutoff": cutoff,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "results": results,
            "meta_test": meta_te,
            "y_test": y_te,
        })
    return folds


def pool_and_trade_sim(folds: list[dict], noise_std: float, threshold: float):
    """Pool OOS GBT predictions across folds, run synthetic trade sim."""
    if not folds:
        return
    print()
    print("=" * 70)
    print("POOLED OOS METRICS (line prediction)")
    print("=" * 70)

    # Gather pooled GBT predictions vs target
    pred_gbt_all = np.concatenate([
        next(r for r in f["results"] if r.name == "gbt").pred for f in folds
    ])
    pred_sim_all = np.concatenate([
        next(r for r in f["results"] if r.name == "sim_raw").pred for f in folds
    ])
    pred_ridge_all = np.concatenate([
        next(r for r in f["results"] if r.name == "ridge").pred for f in folds
    ])
    y_all = np.concatenate([f["y_test"] for f in folds])
    actual_outcome = np.array([m["actual_home_win"]
                                for f in folds for m in f["meta_test"]])

    def report(name: str, pred: np.ndarray):
        mae = float(np.mean(np.abs(y_all - pred)))
        rmse = float(np.sqrt(np.mean((y_all - pred) ** 2)))
        ss_res = float(np.sum((y_all - pred) ** 2))
        ss_tot = float(np.sum((y_all - y_all.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {name:<12} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:+.3f}")

    print(f"  Total OOS games: {len(y_all)}")
    print()
    report("always_50", np.full_like(y_all, 0.5))
    report("sim_raw",   pred_sim_all)
    report("ridge",     pred_ridge_all)
    report("gbt",       pred_gbt_all)

    # The verdict: if GBT MAE < sim_raw MAE meaningfully, we have a real
    # gain over the current architecture.
    sim_mae = float(np.mean(np.abs(y_all - pred_sim_all)))
    gbt_mae = float(np.mean(np.abs(y_all - pred_gbt_all)))
    delta = sim_mae - gbt_mae
    print()
    print(f"  GBT vs sim_raw MAE delta: {delta:+.4f} "
          f"({'GBT better' if delta > 0 else 'sim_raw better'})")
    if delta > 0.005:
        print("  → GBT extracts non-trivial signal beyond the sim's raw output.")
        print("    Worth proceeding to: integrate as a daily inference step,")
        print("    collect Kalshi mid prices for live A/B vs predictor.")
    elif delta > 0:
        print("  → GBT is marginally better. Real but small. Worth one more")
        print("    iteration with richer features (weather, lineup, bullpen rest)")
        print("    before concluding.")
    else:
        print("  → GBT does NOT improve on sim_raw. The features we have don't")
        print("    contain information beyond what the sim already encoded.")
        print("    Conclusion: we cannot beat the closing line with these features.")
        print("    Either find new features (lineup speed, weather, etc.) or stop.")

    # Synthetic trade sim
    print()
    print("=" * 70)
    print(f"SYNTHETIC TRADE SIM (Kalshi mid ~ DK_close + N(0, {noise_std:.3f}))")
    print("=" * 70)
    print("  This is illustrative, not a real backtest. We don't have")
    print("  historical Kalshi mids; we simulate them as noisy DK closes.")
    print()

    for label, pred in [("sim_raw", pred_sim_all), ("gbt", pred_gbt_all)]:
        sim = simulate_line_trades(
            pred, y_all, actual_outcome,
            kalshi_noise_std=noise_std, threshold=threshold,
        )
        if sim["n"] == 0:
            print(f"  {label}: no trades triggered at threshold {threshold:.0%}")
            continue
        print(f"  {label}: bets={sim['n']}  wins={sim['wins']}/{sim['n']} "
              f"({sim['win_rate']:.1%})  ROI={sim['roi']:+.1%}  "
              f"mean edge={sim['mean_edge']:+.3f}")

    print()
    print("  Interpretation:")
    print("    - If GBT > sim_raw on ROI here, the signal is useful for trading")
    print("    - If both lose money, our predictor still can't be exploited even")
    print("      against a noisy market — closing lines are too efficient")
    print("    - The noise_std parameter controls how 'soft' the synthetic Kalshi")
    print("      is: 0.025 = fairly sharp; 0.05 = quite loose. Try a sweep.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", type=str, default=",".join(DEFAULT_CUTOFFS))
    parser.add_argument("--n-sims", type=int, default=1500,
                        help="sim runs per game when generating sim_home_wp feature")
    parser.add_argument("--noise-std", type=float, default=0.025,
                        help="synthetic Kalshi mid noise std-dev")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="bet trigger: |pred_close - mid| ≥ threshold")
    parser.add_argument("--odds-files", type=str, default=",".join(DEFAULT_ODDS_FILES),
                        help="Comma-separated ESPN odds file paths to merge")
    args = parser.parse_args()

    cutoffs = [c.strip() for c in args.cutoffs.split(",") if c.strip()]
    odds_files = [p.strip() for p in args.odds_files.split(",") if p.strip()]

    print("Loading backtest inputs and ESPN odds...")
    inputs = load_backtest_inputs()
    odds_idx = load_odds_combined(odds_files)
    print(f"  {len(inputs['all_games'])} games, {len(odds_idx)} games with closing odds")
    print(f"  Odds sources: {odds_files}")

    print("Building training set...")
    X, y, meta = build_dataset(inputs, odds_idx, n_sims=args.n_sims)
    print(f"  Final dataset: {X.shape[0]} games × {X.shape[1]} features")

    folds = walk_forward(X, y, meta, cutoffs)
    pool_and_trade_sim(folds, args.noise_std, args.threshold)


if __name__ == "__main__":
    main()
