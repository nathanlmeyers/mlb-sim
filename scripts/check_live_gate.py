"""Hard gate that must pass before any live betting begins.

Runs the walk-forward backtest and refuses to print PASSED unless BOTH:
  1. ≥ MIN_OOS_BETS pooled across folds
  2. 95% bootstrap lower-CI on ROI > MIN_LOWER_CI_ROI

Exit codes:
  0  gate passed     — live trading is permitted
  1  gate failed     — exited with explicit FAIL
  2  gate inconclusive — not enough OOS bets to evaluate

This script is the load-bearing precondition for `paper_trade.py live`.
There is no env-var override; if you want to bypass it, change the code
in source control where the change is reviewable.

Usage:
    python scripts/check_live_gate.py
    python scripts/check_live_gate.py --cutoffs 2026-04-08,2026-04-12,2026-04-16
    python scripts/check_live_gate.py --metric totals   # gate on totals not ML
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from backtest.json_backtest import load_backtest_inputs
from scripts.walk_forward_backtest import (
    DEFAULT_CUTOFFS,
    load_espn_odds,
    run_fold,
    bootstrap_roi_ci,
)
import config


# Gate thresholds — locked here, not in config.py, so they are not silently
# mutated. Change requires a code review.
MIN_OOS_BETS = 50
MIN_LOWER_CI_ROI = 0.01     # +1% lower-CI ROI is the bar; positive but conservative
MIN_FOLDS = 3               # need at least 3 train/test splits


def evaluate_gate(metric: str, cutoffs: list[str], n_sims_train: int,
                  n_sims_eval: int, mw: float, me: float) -> int:
    inputs = load_backtest_inputs()
    odds_idx = load_espn_odds()
    print(f"Loaded {len(inputs['all_games'])} games, "
          f"{len(odds_idx)} games with DK closing odds")

    fold_results = []
    for cutoff in cutoffs:
        result = run_fold(
            inputs, odds_idx, cutoff,
            n_sims_train=n_sims_train, n_sims_eval=n_sims_eval,
            fixed_market_weight=mw, fixed_min_edge=me,
        )
        fold_results.append(result)

    valid = [f for f in fold_results if not f.get("skipped")]
    if len(valid) < MIN_FOLDS:
        print()
        print("=" * 70)
        print(f"❌ GATE FAILED: only {len(valid)} valid folds (need ≥ {MIN_FOLDS})")
        print("=" * 70)
        return 1

    all_bets = []
    for f in valid:
        if metric == "totals":
            all_bets.extend(f.get("totals_sim", {}).get("bets", []))
        else:
            all_bets.extend(f["bet_sim"]["bets"])

    n = len(all_bets)
    if n < MIN_OOS_BETS:
        print()
        print("=" * 70)
        print(f"⏸  GATE INCONCLUSIVE: only {n} OOS bets pooled (need ≥ {MIN_OOS_BETS})")
        print(f"   Bootstrap CI on n < {MIN_OOS_BETS} systematically under-states")
        print(f"   uncertainty. Cannot trust 'positive lower bound' on this sample.")
        print(f"   Likely cause: too few OOS games with DK closing-line data.")
        print(f"   → Run B2 (backfill ESPN odds) and re-run this gate.")
        print("=" * 70)
        return 2

    total_staked = sum(b["stake"] for b in all_bets)
    total_profit = sum(b["profit"] for b in all_bets)
    roi = total_profit / total_staked if total_staked > 0 else float("nan")
    lo, hi = bootstrap_roi_ci(all_bets)

    print()
    print("=" * 70)
    print("LIVE-TRADING GATE")
    print("=" * 70)
    print(f"  metric:                {metric}")
    print(f"  folds:                 {len(valid)}")
    print(f"  pooled OOS bets:       {n}")
    print(f"  pooled OOS ROI:        {roi:+.1%}")
    print(f"  95% bootstrap CI:      [{lo:+.1%}, {hi:+.1%}]")
    print(f"  required lower-CI:     ≥ {MIN_LOWER_CI_ROI:+.1%}")
    print()
    if lo >= MIN_LOWER_CI_ROI:
        print(f"  ✅ GATE PASSED — live trading permitted.")
        print(f"     Even so, cap per-bet stake at $2 for the first 50 live bets.")
        print(f"     Auto-halt rules in betting/risk.py still apply.")
        return 0
    else:
        print(f"  ❌ GATE FAILED — lower-CI {lo:+.1%} < required {MIN_LOWER_CI_ROI:+.1%}")
        print(f"     Do not place real money. Either:")
        print(f"       - collect more closing-line data (Phase B)")
        print(f"       - re-tune via grid search (Phase D)")
        print(f"       - pivot to totals (Phase C) and re-run this gate")
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["ml", "totals"], default="ml",
                        help="Which bet type to evaluate the gate on")
    parser.add_argument("--cutoffs", type=str, default=",".join(DEFAULT_CUTOFFS))
    parser.add_argument("--sims-train", type=int, default=800)
    parser.add_argument("--sims-eval", type=int, default=1500)
    parser.add_argument("--market-weight", type=float, default=config.TRAINED_MARKET_WEIGHT)
    parser.add_argument("--min-edge", type=float, default=config.MIN_EDGE_THRESHOLD)
    args = parser.parse_args()

    cutoffs = [c.strip() for c in args.cutoffs.split(",") if c.strip()]
    code = evaluate_gate(args.metric, cutoffs,
                         args.sims_train, args.sims_eval,
                         args.market_weight, args.min_edge)
    sys.exit(code)


if __name__ == "__main__":
    main()
