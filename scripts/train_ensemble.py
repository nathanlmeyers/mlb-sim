"""Real grid search over ensemble + filter hyperparameters.

Replaces the original stub (which trained on synthetic data — a
leftover from a different sport). This driver:

1. Runs walk-forward predictions once per (prior_weight, reg_n) combo,
   reusing the work for every (market_weight, min_edge) combo on top.
2. Pools OOS predictions across cutoffs and computes simulated ROI under
   each (market_weight, min_edge) using the de-vigged DK closing line as
   market price.
3. Selects the combo with the highest 95% bootstrap LOWER-CI on ROI
   (not point ROI — point ROI selection is itself overfitting).
4. Emits a config block the user can copy into `config.py`.

Hyperparameters tuned:
  - prior_weight (pitcher 2025 prior blend) ∈ {0.3, 0.5, 0.7}
  - reg_n (pitcher league-mean regression) ∈ {10, 20, 40}
  - TRAINED_MARKET_WEIGHT (Bayesian market prior) ∈ {0.40, 0.55, 0.65, 0.75}
  - MIN_EDGE_THRESHOLD ∈ {0.03, 0.04, 0.06, 0.08}

This is honest because tuning happens AFTER per-fold pre-cutoff splitting:
the predictions used for ROI are out-of-sample with respect to whatever
data could inform pitcher_index / team_model. The grid search is then over
the choice of *filter* applied to those OOS predictions.

Important caveat: with N folds × ESPN-closing-line games per fold, the
total OOS bet count under any combo may be small. The script enforces a
minimum n=30 bets before reporting a winner. Below that, no params are
recommended.

Usage:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --cutoffs 2026-04-05,2026-04-10,2026-04-15
    python scripts/train_ensemble.py --metric totals
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from backtest.json_backtest import load_backtest_inputs, simulate_predictions_with_params
from scripts.walk_forward_backtest import (
    DEFAULT_CUTOFFS,
    PITCHER_PRIOR_WEIGHTS,
    PITCHER_REG_NS,
    MARKET_WEIGHT_SWEEP,
    MIN_EDGE_SWEEP,
    load_espn_odds,
    attach_espn_odds,
    simulate_bets,
    simulate_total_bets,
    bootstrap_roi_ci,
)


MIN_BETS_FOR_RECOMMENDATION = 30


def grid_search(metric: str, cutoffs: list[str], n_sims: int) -> dict | None:
    inputs = load_backtest_inputs()
    odds_idx = load_espn_odds()

    print(f"Loaded {len(inputs['all_games'])} games, "
          f"{len(odds_idx)} games with DK closing odds")
    print(f"Tuning metric: {metric}")
    print(f"Cutoffs: {cutoffs}")
    print(f"Sims per (cutoff, prior_weight, reg_n) combo: {n_sims}")
    print()

    # Step 1: per-fold OOS predictions for each pitcher hyperparam combo.
    # Cache them by (cutoff, pw, rn) so we only pay simulation cost once per pitcher combo.
    pred_cache: dict[tuple[str, float, int], list[dict]] = {}

    pitcher_combos = list(product(PITCHER_PRIOR_WEIGHTS, PITCHER_REG_NS))
    n_combos = len(pitcher_combos) * len(cutoffs)
    print(f"Generating predictions: {n_combos} combos total...")

    for ci, cutoff in enumerate(cutoffs):
        all_games = inputs["all_games"]
        test = [g for g in all_games if g["game_date"] >= cutoff]
        if len(test) < 10:
            print(f"  [skip] cutoff {cutoff}: only {len(test)} OOS games")
            continue
        for pwi, (pw, rn) in enumerate(pitcher_combos):
            print(f"  cutoff={cutoff} pw={pw} rn={rn} ({ci*len(pitcher_combos)+pwi+1}/{n_combos})", flush=True)
            preds = simulate_predictions_with_params(
                inputs, test,
                prior_weight=pw, reg_n=rn,
                n_sims=n_sims, seed=43,
            )
            preds = attach_espn_odds(preds, odds_idx)
            pred_cache[(cutoff, pw, rn)] = preds

    # Step 2: for each (pw, rn, mw, me) combo, pool predictions across folds
    # and simulate bets. Pick the combo with best lower-CI ROI.
    print()
    print("=" * 80)
    print("Grid search results (pooled across folds)")
    print("=" * 80)

    full_grid = list(product(
        PITCHER_PRIOR_WEIGHTS, PITCHER_REG_NS,
        MARKET_WEIGHT_SWEEP, MIN_EDGE_SWEEP,
    ))
    print(f"Evaluating {len(full_grid)} (pw, rn, mw, me) combinations...")
    print()
    print(f"  {'pw':>5} {'rn':>4} {'mw':>5} {'edge':>6} {'bets':>5} {'wins':>5} "
          f"{'ROI':>9} {'lo CI':>9} {'hi CI':>9} {'CLV':>7}")
    print("  " + "-" * 78)

    results = []
    for pw, rn, mw, me in full_grid:
        all_preds = []
        for cutoff in cutoffs:
            key = (cutoff, pw, rn)
            if key in pred_cache:
                all_preds.extend(pred_cache[key])
        if not all_preds:
            continue

        if metric == "totals":
            sim = simulate_total_bets(all_preds, min_edge=me)
        else:
            sim = simulate_bets(all_preds, market_weight=mw, min_edge=me)

        if sim["n_bets"] < 5:
            continue
        lo, hi = bootstrap_roi_ci(sim["bets"]) if sim["n_bets"] >= 5 else (float("nan"), float("nan"))
        results.append({
            "pw": pw, "rn": rn, "mw": mw, "me": me,
            "n_bets": sim["n_bets"],
            "wins": sim["wins"],
            "roi": sim["roi"],
            "lo": lo, "hi": hi,
            "clv": sim["mean_clv"],
        })
        flag = " ← n<30" if sim["n_bets"] < MIN_BETS_FOR_RECOMMENDATION else ""
        print(f"  {pw:>5.2f} {rn:>4d} {mw:>5.2f} {me:>5.0%} {sim['n_bets']:>5} "
              f"{sim['wins']:>5} {sim['roi']:>+8.1%} {lo:>+8.1%} {hi:>+8.1%} {sim['mean_clv']:>+6.3f}{flag}")

    print()

    # Step 3: pick the winner — best lower-CI ROI among combos with ≥ MIN_BETS_FOR_RECOMMENDATION
    eligible = [r for r in results if r["n_bets"] >= MIN_BETS_FOR_RECOMMENDATION]
    if not eligible:
        print("=" * 80)
        print(f"⏸  NO RECOMMENDATION: No combo had ≥ {MIN_BETS_FOR_RECOMMENDATION} bets.")
        print(f"   Top combos by point ROI (for diagnostic only — do NOT lock these in):")
        for r in sorted(results, key=lambda x: -x["roi"])[:5]:
            print(f"     pw={r['pw']} rn={r['rn']} mw={r['mw']} me={r['me']}: "
                  f"n={r['n_bets']} ROI={r['roi']:+.1%}")
        print(f"   → Get more closing-line data (Phase B2) before tuning.")
        return None

    winner = max(eligible, key=lambda r: r["lo"])
    print("=" * 80)
    if winner["lo"] <= 0:
        print(f"⚠  Best combo has lower-CI ROI {winner['lo']:+.1%} ≤ 0 — no params produce")
        print(f"   robust positive expected value. Do not lock in any config; the model")
        print(f"   has no reliable filter under the current data.")
        return None

    print(f"✅ RECOMMENDED CONFIG (highest lower-CI ROI on ≥ {MIN_BETS_FOR_RECOMMENDATION} bets):")
    print()
    print(f"   prior_weight (pitcher 2025 blend):  {winner['pw']}")
    print(f"   reg_n (pitcher league regression):  {winner['rn']}")
    print(f"   TRAINED_MARKET_WEIGHT:              {winner['mw']}")
    print(f"   MIN_EDGE_THRESHOLD:                 {winner['me']}")
    print(f"   → bets: {winner['n_bets']}, wins: {winner['wins']}, "
          f"ROI: {winner['roi']:+.1%} (CI [{winner['lo']:+.1%}, {winner['hi']:+.1%}]), "
          f"CLV: {winner['clv']:+.3f}")
    print()
    print(f"   Copy into config.py and DO NOT re-tune for at least 30 more paper bets")
    print(f"   under this config. Re-tuning after each loss is data snooping.")
    return winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["ml", "totals"], default="ml")
    parser.add_argument("--cutoffs", type=str, default=",".join(DEFAULT_CUTOFFS))
    parser.add_argument("--sims", type=int, default=1500)
    args = parser.parse_args()

    cutoffs = [c.strip() for c in args.cutoffs.split(",") if c.strip()]
    grid_search(args.metric, cutoffs, args.sims)


if __name__ == "__main__":
    main()
