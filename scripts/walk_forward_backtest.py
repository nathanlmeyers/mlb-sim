"""Walk-forward (out-of-sample) backtest for the MLB sim engine.

Why this exists
---------------
The existing backtest in `backtest/json_backtest.py` evaluates the model on the
same dataset whose hyperparameters (pitcher prior_weight, reg_n, market_weight,
edge threshold) were chosen against. That is in-sample evaluation and cannot
detect overfitting.

This driver fixes that:

  for each cutoff date:
      train  = games strictly before cutoff
      test   = games on/after cutoff
      tune   = pick (pitcher_prior_weight, pitcher_reg_n) that minimize Brier on `train`
      eval   = run model with tuned params on `test`, report accuracy / Brier
      ROI    = for `test` games that also have a DraftKings closing line in
               espn_odds_2026.json, simulate placing bets under the live config
               (market_weight=0.65, min_edge=0.04, 15-85% band, eighth-Kelly)

After all folds run, we pool the OOS predictions and bootstrap a 95% CI on
accuracy and ROI. **Lower-CI ROI > 0 across pooled folds is the gate**.

Why we tune only pitcher params and not market_weight in the inner loop
----------------------------------------------------------------------
Tuning market_weight requires per-game market prices on training data. We
have ~50 games of DK closing prices total, all clustered in Apr 13-16. There
is not enough pre-cutoff price data to do a meaningful market_weight grid
search inside each fold. Instead we report OOS ROI at the *current* live
market_weight (0.65) and at a small sweep, so the user can see whether 0.65
is even reasonable post-hoc.

Run
---
    python scripts/walk_forward_backtest.py
    python scripts/walk_forward_backtest.py --cutoffs 2026-04-08,2026-04-12 --sims 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from backtest.json_backtest import (
    load_backtest_inputs,
    simulate_predictions_with_params,
)
from betting.ev import remove_vig, american_to_decimal
import config


# -----------------------------------------------------------------------------
# Hyperparameter grids
# -----------------------------------------------------------------------------

# Pitcher model knobs — tuned on pre-cutoff data per fold.
PITCHER_PRIOR_WEIGHTS = [0.3, 0.5, 0.7]
PITCHER_REG_NS = [10, 20, 40]

# Market blend / edge filter — applied at OOS evaluation time only (we cannot
# tune these in-fold without enough closing-line data). We sweep so the user
# can see whether the current values dominate.
MARKET_WEIGHT_SWEEP = [0.40, 0.55, 0.65, 0.75]
MIN_EDGE_SWEEP = [0.03, 0.04, 0.06, 0.08]

DEFAULT_CUTOFFS = ["2026-04-05", "2026-04-08", "2026-04-12", "2026-04-14"]


# -----------------------------------------------------------------------------
# ESPN closing-line lookup
# -----------------------------------------------------------------------------

def load_espn_odds(path: str = ".context/espn_odds_2026.json") -> dict:
    """Index ESPN closing odds by (date, home_abbr, away_abbr)."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        rows = json.load(f)
    idx = {}
    for r in rows:
        key = (r["date"], r["home"], r["away"])
        idx[key] = r
    return idx


# Map team names (statsapi) to ESPN abbreviations
from backtest.json_backtest import TEAM_ABBR


def attach_espn_odds(predictions: list[dict], odds_idx: dict) -> list[dict]:
    """For each prediction, attach DK ML odds + de-vigged market prob + closing total if available."""
    annotated = []
    for p in predictions:
        home_abbr = TEAM_ABBR.get(p["home_team"])
        away_abbr = TEAM_ABBR.get(p["away_team"])
        key = (p["date"], home_abbr, away_abbr)
        rec = odds_idx.get(key)
        if rec is None:
            p = {**p, "dk_ml_home": None, "dk_ml_away": None,
                 "market_home_wp": None, "dk_total_close": None}
        else:
            ml_home = rec["odds"].get("ml_home")
            ml_away = rec["odds"].get("ml_away")
            total_close = rec["odds"].get("total_close")
            if ml_home is not None and ml_away is not None:
                true_h, _ = remove_vig(ml_home, ml_away)
                p = {**p, "dk_ml_home": ml_home, "dk_ml_away": ml_away,
                     "market_home_wp": true_h, "dk_total_close": total_close}
            else:
                p = {**p, "dk_ml_home": None, "dk_ml_away": None,
                     "market_home_wp": None, "dk_total_close": total_close}
        annotated.append(p)
    return annotated


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_brier(predictions: list[dict]) -> float:
    if not predictions:
        return float("nan")
    return float(np.mean([
        (p["model_home_wp"] - (1.0 if p["actual_home_win"] else 0.0)) ** 2
        for p in predictions
    ]))


def compute_accuracy(predictions: list[dict]) -> float:
    if not predictions:
        return float("nan")
    return float(np.mean([
        (p["model_home_wp"] > 0.5) == p["actual_home_win"]
        for p in predictions
    ]))


def simulate_bets(
    predictions: list[dict],
    market_weight: float,
    min_edge: float,
    min_market_price: float = 0.15,
    max_market_price: float = 0.85,
    kelly_fraction: float = 0.125,
    max_bet_pct: float = 0.03,
    starting_bankroll: float = 100.0,
) -> dict:
    """Replay the live ML edge filter on annotated predictions.

    Predictions must have `market_home_wp` (de-vigged DK closing line) and
    `dk_ml_home` / `dk_ml_away` set. Predictions without market data are
    skipped (we cannot evaluate them as bets).

    Returns: dict with bets placed, ROI, win rate, profit, CLV.
    """
    bankroll = starting_bankroll
    bets = []

    for p in predictions:
        market_h = p.get("market_home_wp")
        if market_h is None:
            continue
        if not (min_market_price <= market_h <= max_market_price):
            continue
        # The live system also filters out the inverse (away side) at extremes
        market_a = 1.0 - market_h

        model_h = p["model_home_wp"]
        # Blend model with market as Bayesian prior — same formula as
        # sim/ensemble.py:133 (without confidence discount, since
        # box-score-only doesn't have engine agreement to discount on).
        blended_h = (1 - market_weight) * model_h + market_weight * market_h
        blended_a = 1.0 - blended_h

        edge_h = blended_h - market_h  # positive: bet HOME
        edge_a = blended_a - market_a  # positive: bet AWAY

        side = None
        if edge_h >= min_edge and (min_market_price <= market_h <= max_market_price):
            side = "HOME"
            blend_p = blended_h
            mkt_p = market_h
            ml = p["dk_ml_home"]
            won = p["actual_home_win"]
        elif edge_a >= min_edge and (min_market_price <= market_a <= max_market_price):
            side = "AWAY"
            blend_p = blended_a
            mkt_p = market_a
            ml = p["dk_ml_away"]
            won = not p["actual_home_win"]
        else:
            continue

        # Eighth-Kelly sizing on the EDGE (not the blended prob — the edge is
        # what determines bet size in the real system).
        kelly_pct = min(max_bet_pct, max(0, blend_p - mkt_p) * kelly_fraction)
        if kelly_pct <= 0:
            continue
        stake = bankroll * kelly_pct
        if stake <= 0:
            continue

        dec_odds = american_to_decimal(ml)
        profit = stake * (dec_odds - 1) if won else -stake
        bankroll += profit

        # CLV proxy: model_p − market_p (positive = beat the market on this side)
        clv = blend_p - mkt_p

        bets.append({
            "date": p["date"],
            "game": f"{p['away_team']} @ {p['home_team']}",
            "side": side,
            "model_p": model_h if side == "HOME" else 1.0 - model_h,
            "blend_p": blend_p,
            "market_p": mkt_p,
            "ml": ml,
            "stake": stake,
            "dec_odds": dec_odds,
            "won": won,
            "profit": profit,
            "clv": clv,
        })

    total_staked = sum(b["stake"] for b in bets)
    total_profit = sum(b["profit"] for b in bets)
    wins = sum(1 for b in bets if b["won"])
    return {
        "n_bets": len(bets),
        "wins": wins,
        "losses": len(bets) - wins,
        "win_rate": wins / len(bets) if bets else float("nan"),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": total_profit / total_staked if total_staked > 0 else float("nan"),
        "ending_bankroll": bankroll,
        "mean_clv": float(np.mean([b["clv"] for b in bets])) if bets else float("nan"),
        "bets": bets,
    }


def simulate_total_bets(
    predictions: list[dict],
    min_edge: float,
    min_market_price: float = 0.20,
    max_market_price: float = 0.80,
    kelly_fraction: float = 0.125,
    max_bet_pct: float = 0.03,
    starting_bankroll: float = 100.0,
    over_juice: int = -110,
) -> dict:
    """Simulate over/under bets using DK closing total as the line.

    For each prediction with `dk_total_close`, look up the model's over_pct at
    the closest half-run line, then compute edge vs the de-vigged market prob
    (we approximate market O/U at -110 each side → 50/50 implied true prob).

    Unlike ML, we don't have separate over/under prices in the cached ESPN
    data — only the line. We assume standard -110/-110 vig on totals (true
    50/50 prob), so edge = model_prob - 0.50 in either direction. That's a
    reasonable proxy for closing-line totals priced near the consensus.
    """
    # Implied true prob each side at -110 / -110: 0.5238 raw → 0.5 de-vigged
    market_p = 0.50
    bankroll = starting_bankroll
    bets = []

    for p in predictions:
        line = p.get("dk_total_close")
        if line is None:
            continue
        # Find the over_pct at the line (or nearest half-run line)
        op = p.get("over_pct_by_line") or {}
        if not op:
            continue
        # over_pct_by_line keys are floats serialized; normalize
        keys_f = {float(k): float(v) for k, v in op.items()}
        if not keys_f:
            continue
        # nearest half-run line
        best_line = min(keys_f.keys(), key=lambda x: abs(x - line))
        if abs(best_line - line) > 0.5:
            continue  # too far from market; predictions don't cover this line
        over_p = keys_f[best_line]
        under_p = 1.0 - over_p

        # Edge vs 0.5 (post-vig market, both sides at -110)
        edge_over = over_p - market_p
        edge_under = under_p - market_p

        # Decide side
        if edge_over >= min_edge and (min_market_price <= market_p <= max_market_price):
            side = "OVER"
            model_p = over_p
            won = (p["actual_total"] > line)
        elif edge_under >= min_edge and (min_market_price <= market_p <= max_market_price):
            side = "UNDER"
            model_p = under_p
            won = (p["actual_total"] < line)
        else:
            continue
        # Push handling: if exactly the line, no bet outcome — treat as void
        if p["actual_total"] == line:
            continue

        # Kelly on edge
        kelly_pct = min(max_bet_pct, max(0, model_p - market_p) * kelly_fraction)
        if kelly_pct <= 0:
            continue
        stake = bankroll * kelly_pct
        if stake <= 0:
            continue
        dec_odds = american_to_decimal(over_juice)  # both sides -110
        profit = stake * (dec_odds - 1) if won else -stake
        bankroll += profit

        bets.append({
            "date": p["date"],
            "game": f"{p['away_team']} @ {p['home_team']}",
            "side": side,
            "line": line,
            "model_p": model_p,
            "market_p": market_p,
            "ml": over_juice,
            "stake": stake,
            "dec_odds": dec_odds,
            "won": won,
            "profit": profit,
            "clv": model_p - market_p,
        })

    total_staked = sum(b["stake"] for b in bets)
    total_profit = sum(b["profit"] for b in bets)
    wins = sum(1 for b in bets if b["won"])
    return {
        "n_bets": len(bets),
        "wins": wins,
        "losses": len(bets) - wins,
        "win_rate": wins / len(bets) if bets else float("nan"),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": total_profit / total_staked if total_staked > 0 else float("nan"),
        "ending_bankroll": bankroll,
        "mean_clv": float(np.mean([b["clv"] for b in bets])) if bets else float("nan"),
        "bets": bets,
    }


def compute_totals_accuracy(predictions: list[dict]) -> tuple[float, int]:
    """Accuracy of (over_pct > 0.5) vs (actual_total > line) using nearest-half-run line."""
    correct = 0
    n = 0
    for p in predictions:
        line = p.get("dk_total_close")
        if line is None:
            continue
        op = p.get("over_pct_by_line") or {}
        if not op:
            continue
        keys_f = {float(k): float(v) for k, v in op.items()}
        if not keys_f:
            continue
        best_line = min(keys_f.keys(), key=lambda x: abs(x - line))
        if abs(best_line - line) > 0.5:
            continue
        if p["actual_total"] == line:
            continue  # push
        over_p = keys_f[best_line]
        n += 1
        if (over_p > 0.5) == (p["actual_total"] > line):
            correct += 1
    return (correct / n if n else float("nan"), n)


def bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05,
                 rng_seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of `values`."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def bootstrap_roi_ci(bets: list[dict], n_boot: int = 5000, alpha: float = 0.05,
                     rng_seed: int = 7) -> tuple[float, float]:
    """Bootstrap CI on ROI by resampling bets (each bet's profit/stake)."""
    if not bets:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    n = len(bets)
    profits = np.array([b["profit"] for b in bets], dtype=float)
    stakes = np.array([b["stake"] for b in bets], dtype=float)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = stakes[idx].sum()
        p = profits[idx].sum()
        rois.append(p / s if s > 0 else 0.0)
    arr = np.array(rois)
    return (float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2)))


# -----------------------------------------------------------------------------
# Walk-forward driver
# -----------------------------------------------------------------------------

def tune_pitcher_params_on_train(
    inputs: dict,
    train_games: list[dict],
    n_sims: int = 1500,
) -> tuple[float, int, float]:
    """Grid-search (prior_weight, reg_n) minimizing in-fold-train Brier.

    Returns (best_prior_weight, best_reg_n, best_brier).
    """
    best = (None, None, float("inf"))
    for pw in PITCHER_PRIOR_WEIGHTS:
        for rn in PITCHER_REG_NS:
            preds = simulate_predictions_with_params(
                inputs, train_games,
                prior_weight=pw, reg_n=rn,
                n_sims=n_sims, seed=42,
            )
            br = compute_brier(preds)
            if br < best[2]:
                best = (pw, rn, br)
    return best


def run_fold(
    inputs: dict,
    odds_idx: dict,
    cutoff: str,
    n_sims_train: int,
    n_sims_eval: int,
    fixed_market_weight: float,
    fixed_min_edge: float,
) -> dict:
    all_games = inputs["all_games"]
    train = [g for g in all_games if g["game_date"] < cutoff]
    test = [g for g in all_games if g["game_date"] >= cutoff]

    print(f"\n{'='*70}")
    print(f"FOLD: cutoff={cutoff}  train={len(train)}  test={len(test)}")
    print(f"{'='*70}")

    if len(train) < 30 or len(test) < 10:
        print("  [skip] not enough games on one side of the cutoff")
        return {"cutoff": cutoff, "skipped": True}

    print("  Tuning pitcher (prior_weight, reg_n) on pre-cutoff data...")
    pw, rn, train_brier = tune_pitcher_params_on_train(
        inputs, train, n_sims=n_sims_train,
    )
    print(f"    best: prior_weight={pw}  reg_n={rn}  in-train Brier={train_brier:.4f}")

    print(f"  Evaluating on {len(test)} OOS games (n_sims={n_sims_eval})...")
    test_preds = simulate_predictions_with_params(
        inputs, test,
        prior_weight=pw, reg_n=rn,
        n_sims=n_sims_eval, seed=43,
    )
    test_preds = attach_espn_odds(test_preds, odds_idx)

    acc = compute_accuracy(test_preds)
    brier = compute_brier(test_preds)
    n_with_market = sum(1 for p in test_preds if p["market_home_wp"] is not None)

    print(f"    OOS accuracy: {acc:.1%} ({len(test_preds)} games)")
    print(f"    OOS Brier:    {brier:.4f}")
    print(f"    games with DK closing line: {n_with_market}")

    bet_sim = simulate_bets(
        test_preds,
        market_weight=fixed_market_weight,
        min_edge=fixed_min_edge,
    )
    if bet_sim["n_bets"] > 0:
        roi_lo, roi_hi = bootstrap_roi_ci(bet_sim["bets"])
        print(f"    ML live config (mw={fixed_market_weight} edge≥{fixed_min_edge:.0%}):")
        print(f"      bets: {bet_sim['n_bets']}  W-L: {bet_sim['wins']}-{bet_sim['losses']}")
        print(f"      ROI: {bet_sim['roi']:+.1%}  95% CI [{roi_lo:+.1%}, {roi_hi:+.1%}]")
        print(f"      mean CLV (blend_p − market_p): {bet_sim['mean_clv']:+.3f}")
    else:
        print(f"    ML live config (mw={fixed_market_weight} edge≥{fixed_min_edge:.0%}): no bets triggered")

    # Totals evaluation (Phase C of remediation plan).
    totals_acc, totals_n = compute_totals_accuracy(test_preds)
    print(f"    Totals OOS accuracy (model O/U vs DK closing line): "
          f"{totals_acc:.1%} on {totals_n} games")
    totals_sim = simulate_total_bets(test_preds, min_edge=fixed_min_edge)
    if totals_sim["n_bets"] > 0:
        roi_lo, roi_hi = bootstrap_roi_ci(totals_sim["bets"])
        print(f"    Totals live config (edge≥{fixed_min_edge:.0%}, -110 vig):")
        print(f"      bets: {totals_sim['n_bets']}  W-L: {totals_sim['wins']}-{totals_sim['losses']}")
        print(f"      ROI: {totals_sim['roi']:+.1%}  95% CI [{roi_lo:+.1%}, {roi_hi:+.1%}]")
        print(f"      mean edge (model − 0.5): {totals_sim['mean_clv']:+.3f}")
    else:
        print(f"    Totals live config (edge≥{fixed_min_edge:.0%}): no bets triggered")

    return {
        "cutoff": cutoff,
        "skipped": False,
        "tuned_prior_weight": pw,
        "tuned_reg_n": rn,
        "train_brier": train_brier,
        "test_predictions": test_preds,
        "test_accuracy": acc,
        "test_brier": brier,
        "n_test": len(test_preds),
        "n_with_market": n_with_market,
        "bet_sim": bet_sim,
        "totals_accuracy": totals_acc,
        "totals_n": totals_n,
        "totals_sim": totals_sim,
    }


def pool_and_report(fold_results: list[dict],
                    fixed_market_weight: float, fixed_min_edge: float):
    valid = [f for f in fold_results if not f.get("skipped")]
    if not valid:
        print("\nNo valid folds. Nothing to pool.")
        return

    all_preds = []
    all_bets = []
    for f in valid:
        all_preds.extend(f["test_predictions"])
        all_bets.extend(f["bet_sim"]["bets"])

    print()
    print("=" * 70)
    print("POOLED OUT-OF-SAMPLE METRICS")
    print("=" * 70)
    print(f"  Folds evaluated:        {len(valid)}")
    print(f"  Total OOS predictions:  {len(all_preds)}")
    print(f"  Predictions w/ market:  {sum(1 for p in all_preds if p['market_home_wp'] is not None)}")

    correct = [(p["model_home_wp"] > 0.5) == p["actual_home_win"] for p in all_preds]
    acc = float(np.mean(correct)) if correct else float("nan")
    acc_lo, acc_hi = bootstrap_ci([float(c) for c in correct])
    brier_per = [
        (p["model_home_wp"] - (1.0 if p["actual_home_win"] else 0.0)) ** 2
        for p in all_preds
    ]
    brier = float(np.mean(brier_per)) if brier_per else float("nan")
    brier_lo, brier_hi = bootstrap_ci(brier_per)

    print()
    print(f"  Accuracy: {acc:.1%}  95% CI [{acc_lo:.1%}, {acc_hi:.1%}]")
    print(f"  Brier:    {brier:.4f}  95% CI [{brier_lo:.4f}, {brier_hi:.4f}]")

    # Always-home baseline on the same pool
    always_home_acc = float(np.mean([p["actual_home_win"] for p in all_preds]))
    print(f"  Always-home baseline accuracy on pool: {always_home_acc:.1%}")

    print()
    print(f"  Live-config bet simulation (mw={fixed_market_weight}, edge≥{fixed_min_edge:.0%}):")
    if all_bets:
        total_staked = sum(b["stake"] for b in all_bets)
        total_profit = sum(b["profit"] for b in all_bets)
        wins = sum(1 for b in all_bets if b["won"])
        roi = total_profit / total_staked if total_staked > 0 else float("nan")
        roi_lo, roi_hi = bootstrap_roi_ci(all_bets)
        clv_mean = float(np.mean([b["clv"] for b in all_bets]))
        clv_lo, clv_hi = bootstrap_ci([b["clv"] for b in all_bets])
        print(f"    bets:        {len(all_bets)}  W-L: {wins}-{len(all_bets)-wins}  ({wins/len(all_bets):.1%})")
        print(f"    total stake: ${total_staked:.2f}   profit: ${total_profit:+.2f}")
        print(f"    ROI:  {roi:+.1%}   95% CI [{roi_lo:+.1%}, {roi_hi:+.1%}]")
        print(f"    CLV:  {clv_mean:+.3f}   95% CI [{clv_lo:+.3f}, {clv_hi:+.3f}]")
        print()
        # The gate requires both a positive lower CI AND a non-trivial sample.
        # With < 30 bets, bootstrap CIs systematically under-state uncertainty
        # (especially when the sample is all wins or all losses), so the gate
        # is "not enough data" rather than "passed."
        MIN_BETS_FOR_GATE = 30
        if len(all_bets) < MIN_BETS_FOR_GATE:
            print(f"  ⏸  GATE INCONCLUSIVE: only {len(all_bets)} bets across all folds.")
            print(f"     Need ≥ {MIN_BETS_FOR_GATE} for a meaningful 95% CI on ROI.")
            print(f"     Bootstrap CIs on small samples wildly under-state uncertainty,")
            print(f"     especially when wins or losses are 0. Do not interpret a")
            print(f"     'positive lower bound' here as evidence of edge.")
        elif roi_lo > 0:
            print(f"  ✅ GATE PASSED: 95% lower CI on OOS ROI is positive across {len(all_bets)} bets.")
        else:
            print(f"  ❌ GATE FAILED: 95% lower CI on OOS ROI is ≤ 0 across {len(all_bets)} bets.")
            print("     The model has not demonstrated edge under the live config out-of-sample.")
            print("     Do not move to real money. Either collect more data, redesign the")
            print("     edge filter (Phase 3), or pivot to totals-only (Phase 4).")
    else:
        print("    No bets triggered across any fold under the live config.")
        print("    Likely because few OOS games have DK closing-line data and the edge")
        print("    threshold is too high relative to model-vs-market disagreement.")

    # Pooled totals metrics
    print()
    print("-" * 70)
    all_totals_bets = []
    totals_correct = 0
    totals_n = 0
    for f in valid:
        all_totals_bets.extend(f["totals_sim"]["bets"])
    # Recompute pooled totals accuracy directly from pooled preds
    pooled_totals_acc, pooled_totals_n = compute_totals_accuracy(all_preds)
    print(f"  Pooled totals OOS accuracy: {pooled_totals_acc:.1%} on {pooled_totals_n} games")
    if all_totals_bets:
        ts = sum(b["stake"] for b in all_totals_bets)
        tp = sum(b["profit"] for b in all_totals_bets)
        wins = sum(1 for b in all_totals_bets if b["won"])
        roi = tp / ts if ts > 0 else float("nan")
        roi_lo, roi_hi = bootstrap_roi_ci(all_totals_bets)
        print(f"  Pooled totals bets: {len(all_totals_bets)}  W-L: {wins}-{len(all_totals_bets)-wins}")
        print(f"  Pooled totals ROI:  {roi:+.1%}   95% CI [{roi_lo:+.1%}, {roi_hi:+.1%}]")
        if len(all_totals_bets) >= 30 and roi_lo > 0:
            print("  ✅ Totals OOS shows positive lower-CI ROI — promising signal worth")
            print("     paper-trading. Re-run check_live_gate.py --metric totals to verify.")
        elif len(all_totals_bets) < 30:
            print(f"  ⏸  Only {len(all_totals_bets)} totals bets pooled — sample too small for")
            print(f"     a meaningful CI. Need ≥ 30 to interpret either direction.")
        else:
            print(f"  ❌ Totals lower-CI ≤ 0 — no edge demonstrated even on the strongest")
            print(f"     supposed signal. Project should be reframed as research, not trading.")
    else:
        print("  No totals bets triggered. Likely missing dk_total_close on most OOS games.")

    print()
    print("  Sweep over (market_weight, min_edge) on pooled ML predictions:")
    print(f"  {'mw':>6} {'min_edge':>10} {'bets':>6} {'wins':>5} {'ROI':>10} {'CLV':>8}")
    print("  " + "-" * 50)
    for mw in MARKET_WEIGHT_SWEEP:
        for me in MIN_EDGE_SWEEP:
            sim = simulate_bets(all_preds, market_weight=mw, min_edge=me)
            if sim["n_bets"] == 0:
                continue
            print(f"  {mw:>6.2f} {me:>10.0%} {sim['n_bets']:>6} "
                  f"{sim['wins']:>5} {sim['roi']:>+9.1%} {sim['mean_clv']:>+8.3f}")
    print()
    print("  NOTE: This sweep is post-hoc; selecting (mw, min_edge) by best ROI here")
    print("  is its own form of overfitting. Use the sweep only to sanity-check that")
    print("  the locked live config (mw=0.65, edge=0.04) isn't dominated by a small")
    print("  perturbation. If it is, the config is fragile and not real edge.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", type=str, default=",".join(DEFAULT_CUTOFFS),
                        help="Comma-separated cutoff dates (YYYY-MM-DD)")
    parser.add_argument("--sims-train", type=int, default=1200,
                        help="n_sims used during in-fold tuning (lower = faster)")
    parser.add_argument("--sims-eval", type=int, default=2500,
                        help="n_sims used for OOS evaluation")
    parser.add_argument("--market-weight", type=float, default=config.TRAINED_MARKET_WEIGHT)
    parser.add_argument("--min-edge", type=float, default=config.MIN_EDGE_THRESHOLD)
    parser.add_argument("--season-file", type=str, default=".context/season_2026.json")
    parser.add_argument("--prior-season-file", type=str, default=".context/season_2025.json")
    parser.add_argument("--odds-file", type=str, default=".context/espn_odds_2026.json")
    args = parser.parse_args()

    cutoffs = [c.strip() for c in args.cutoffs.split(",") if c.strip()]

    print("Loading season data...")
    inputs = load_backtest_inputs(args.season_file, args.prior_season_file)
    print(f"  Loaded {len(inputs['all_games'])} games, "
          f"{len(inputs['pitcher_index'])} pitchers")
    if inputs['prior_pitcher_index'] is None:
        print("  WARNING: no prior-season pitcher data available — prior_weight has limited effect")

    odds_idx = load_espn_odds(args.odds_file)
    print(f"  Loaded {len(odds_idx)} games with DK closing odds")

    fold_results = []
    for cutoff in cutoffs:
        result = run_fold(
            inputs, odds_idx, cutoff,
            n_sims_train=args.sims_train,
            n_sims_eval=args.sims_eval,
            fixed_market_weight=args.market_weight,
            fixed_min_edge=args.min_edge,
        )
        fold_results.append(result)

    pool_and_report(fold_results, args.market_weight, args.min_edge)


if __name__ == "__main__":
    main()
