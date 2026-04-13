"""Consistent model evaluation framework.

Evaluates any model configuration on a fixed eval set and produces
a standardized scorecard. Use this to compare model versions and
ensure changes move us in the right direction.

The framework measures 6 metrics across 3 bet types:
  - ML: accuracy + Brier score (calibration)
  - RL: -1.5 cover accuracy
  - O/U: total prediction accuracy + MAE
  - Margin: spread MAE
  - CLV proxy: how often we'd beat a closing line

Usage:
    python backtest/model_eval.py              # run all models
    python backtest/model_eval.py --detailed   # include slow detailed model
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")

from backtest.json_backtest import (
    build_team_model_from_json, build_pitcher_from_json,
    _build_pitcher_index, load_park_factors_from_json, TEAM_ABBR,
)
from models.batter_model import build_league_average_batter
from models.pitcher_model import build_league_average_pitcher
from models.team_model import ParkFactor
from sim.game import simulate_n_box_score_games
from sim.detailed_game import simulate_n_detailed_games
import config


@dataclass
class ModelScorecard:
    """Standardized evaluation results for one model."""
    name: str
    games: int
    # Moneyline
    ml_accuracy: float      # % of winners correctly picked
    ml_brier: float         # Brier score (lower = better calibration)
    # Run line (-1.5)
    rl_accuracy: float      # % of run line calls correct
    # Over/Under
    ou_accuracy: float      # % of O/U calls correct
    # Spread
    margin_mae: float       # mean absolute error of predicted margin
    # Totals
    total_mae: float        # mean absolute error of predicted total
    # Composite
    composite_score: float = 0.0  # weighted composite (higher = better)

    def compute_composite(self):
        """Weighted composite score. Higher is better.

        Weights reflect where we have the most edge potential:
        - Brier: 30% (calibration is king for betting)
        - ML accuracy: 20%
        - RL accuracy: 15%
        - O/U accuracy: 15%
        - Margin MAE: 10%
        - Total MAE: 10%
        """
        # Normalize each metric to 0-1 scale (higher = better)
        brier_score = max(0, 1.0 - self.ml_brier / 0.25)  # 0.25 = baseline, 0 = perfect
        ml_score = self.ml_accuracy
        rl_score = self.rl_accuracy
        ou_score = self.ou_accuracy
        margin_score = max(0, 1.0 - self.margin_mae / 5.0)  # 5.0 = bad, 0 = perfect
        total_score = max(0, 1.0 - self.total_mae / 5.0)

        self.composite_score = (
            0.30 * brier_score +
            0.20 * ml_score +
            0.15 * rl_score +
            0.15 * ou_score +
            0.10 * margin_score +
            0.10 * total_score
        )

    def __str__(self):
        return (
            f"{self.name:<25} ML:{self.ml_accuracy:.1%} Brier:{self.ml_brier:.4f} "
            f"RL:{self.rl_accuracy:.1%} O/U:{self.ou_accuracy:.1%} "
            f"MrgMAE:{self.margin_mae:.2f} TotMAE:{self.total_mae:.2f} "
            f"Composite:{self.composite_score:.3f}"
        )


def load_eval_data():
    """Load all data needed for evaluation."""
    with open(".context/season_2026.json") as f:
        current_games = json.load(f)

    prior_games = None
    prior_pitcher_idx = None
    if Path(".context/season_2025.json").exists():
        with open(".context/season_2025.json") as f:
            prior_games = json.load(f)
        prior_pitcher_idx = _build_pitcher_index(prior_games)

    with open(".context/season_2026_summary.json") as f:
        summaries = json.load(f)

    starter_lookup = {
        s["game_id"]: (s.get("home_starter", ""), s.get("away_starter", ""))
        for s in summaries
    }

    pitcher_idx = _build_pitcher_index(current_games)
    park_factors = load_park_factors_from_json()

    return current_games, prior_games, pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors


def evaluate_on_games(
    eval_games: list[dict],
    all_games: list[dict],
    model_name: str,
    sim_fn,
    pitcher_idx: dict,
    prior_pitcher_idx: dict | None,
    starter_lookup: dict,
    park_factors: dict,
    use_pitchers: bool = False,
    n_sims: int = 1000,
    min_team_games: int = 3,
) -> ModelScorecard:
    """Run evaluation for a single model config."""
    neutral = ParkFactor(venue="N", overall_factor=1.0, hr_factor=1.0, h_factor=1.0, bb_factor=1.0)
    league_avg = build_league_average_pitcher(is_starter=True)
    rng = np.random.default_rng(42)

    correct = 0; total = 0; brier = 0.0
    rl_correct = 0; ou_correct = 0
    margin_errs = []; total_errs = []

    for g in eval_games:
        ht = build_team_model_from_json(g["home_id"], g["home_team"], all_games, g["game_date"], None)
        at = build_team_model_from_json(g["away_id"], g["away_team"], all_games, g["game_date"], None)
        if ht.wins + ht.losses < min_team_games or at.wins + at.losses < min_team_games:
            continue

        hs = league_avg; as_ = league_avg
        if use_pitchers:
            hs_name, as_name = starter_lookup.get(g["game_id"], ("", ""))
            if hs_name:
                sp = build_pitcher_from_json(hs_name, pitcher_idx, g["game_date"], prior_pitcher_idx)
                if sp: hs = sp
            if as_name:
                sp = build_pitcher_from_json(as_name, pitcher_idx, g["game_date"], prior_pitcher_idx)
                if sp: as_ = sp

        park = park_factors.get(TEAM_ABBR.get(g["home_team"], ""), neutral)
        results = sim_fn(ht, at, hs, as_, park, n_sims, rng)

        actual_hw = g["home_score"] > g["away_score"]
        actual_margin = g["home_score"] - g["away_score"]
        actual_total = g["home_score"] + g["away_score"]

        if (results["home_win_pct"] > 0.5) == actual_hw: correct += 1
        brier += (results["home_win_pct"] - (1.0 if actual_hw else 0.0)) ** 2
        margin_errs.append(abs(results["spread_mean"] - actual_margin))
        total_errs.append(abs(results["total_mean"] - actual_total))

        if (results["home_cover_pct"] > 0.5) == (actual_margin >= 2): rl_correct += 1

        line = round(results["total_mean"] * 2) / 2
        over_p = results["over_pct_by_line"].get(line, 0.5)
        if (over_p > 0.5) == (actual_total > line): ou_correct += 1

        total += 1

    card = ModelScorecard(
        name=model_name,
        games=total,
        ml_accuracy=correct / total if total else 0,
        ml_brier=brier / total if total else 1,
        rl_accuracy=rl_correct / total if total else 0,
        ou_accuracy=ou_correct / total if total else 0,
        margin_mae=float(np.mean(margin_errs)) if margin_errs else 99,
        total_mae=float(np.mean(total_errs)) if total_errs else 99,
    )
    card.compute_composite()
    return card


def run_full_evaluation(include_detailed: bool = False):
    """Run all model variants and print comparison."""
    print("Loading data...")
    current_games, prior_games, pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors = load_eval_data()

    eval_games = [
        g for g in current_games
        if "2026-03-28" <= g["game_date"] <= "2026-04-09" and g.get("home_id")
    ]
    print(f"Eval set: {len(eval_games)} games (Mar 28 - Apr 9)")

    def box_sim(ht, at, hs, as_, park, n, rng):
        return simulate_n_box_score_games(ht, at, hs, as_, park, n, rng)

    def detailed_sim(ht, at, hs, as_, park, n, rng):
        lineup = [build_league_average_batter() for _ in range(9)]
        for i, b in enumerate(lineup): b.player_id = i + 1
        bp = [build_league_average_pitcher(is_starter=False) for _ in range(3)]
        for i, p in enumerate(bp): p.player_id = 100 + i
        return simulate_n_detailed_games(lineup, lineup, hs, as_, bp, bp, park, n, rng)

    scorecards = []

    # Always-home baseline
    home_wins = sum(1 for g in eval_games if g["home_score"] > g["away_score"])
    n = len(eval_games)
    baseline_brier = np.mean([(0.54 - (1.0 if g["home_score"] > g["away_score"] else 0.0))**2 for g in eval_games])
    baseline = ModelScorecard(
        name="Always Home (54%)", games=n,
        ml_accuracy=home_wins/n, ml_brier=baseline_brier,
        rl_accuracy=0, ou_accuracy=0, margin_mae=99, total_mae=99,
    )
    baseline.compute_composite()
    scorecards.append(baseline)

    # Box score models
    print("  Running: Box (no SP)...")
    scorecards.append(evaluate_on_games(
        eval_games, current_games, "Box (no SP)", box_sim,
        pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors,
        use_pitchers=False, n_sims=2000,
    ))

    print("  Running: Box + SP...")
    scorecards.append(evaluate_on_games(
        eval_games, current_games, "Box + SP (2025 priors)", box_sim,
        pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors,
        use_pitchers=True, n_sims=2000,
    ))

    if include_detailed:
        print("  Running: Detailed + TTO (no SP)...")
        scorecards.append(evaluate_on_games(
            eval_games, current_games, "Detailed+TTO (no SP)", detailed_sim,
            pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors,
            use_pitchers=False, n_sims=500,
        ))

        print("  Running: Detailed + TTO + SP...")
        scorecards.append(evaluate_on_games(
            eval_games, current_games, "Detailed+TTO + SP", detailed_sim,
            pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors,
            use_pitchers=True, n_sims=500,
        ))

    # Print results
    print()
    print("=" * 95)
    print(f"  MODEL EVALUATION SCORECARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Eval set: {len(eval_games)} games | Mar 28 – Apr 9, 2026")
    print("=" * 95)
    print()
    print(f"  {'Model':<25} {'ML':>6} {'Brier':>7} {'RL':>6} {'O/U':>6} {'MrgE':>6} {'TotE':>6} {'Score':>7}")
    print("  " + "-" * 88)

    for sc in sorted(scorecards, key=lambda s: -s.composite_score):
        ml_str = f"{sc.ml_accuracy:.1%}" if sc.ml_accuracy > 0 else "—"
        rl_str = f"{sc.rl_accuracy:.1%}" if sc.rl_accuracy > 0 else "—"
        ou_str = f"{sc.ou_accuracy:.1%}" if sc.ou_accuracy > 0 else "—"
        mrg_str = f"{sc.margin_mae:.2f}" if sc.margin_mae < 50 else "—"
        tot_str = f"{sc.total_mae:.2f}" if sc.total_mae < 50 else "—"
        print(f"  {sc.name:<25} {ml_str:>6} {sc.ml_brier:>7.4f} {rl_str:>6} {ou_str:>6} {mrg_str:>6} {tot_str:>6} {sc.composite_score:>7.3f}")

    print()
    print("  Legend: ML=moneyline accuracy, Brier=lower is better, RL=run line -1.5,")
    print("         O/U=over/under, MrgE/TotE=margin/total MAE (lower better),")
    print("         Score=weighted composite (higher better)")
    print("=" * 95)

    # Save results
    results = [{
        "name": sc.name, "games": sc.games,
        "ml_acc": sc.ml_accuracy, "brier": sc.ml_brier,
        "rl_acc": sc.rl_accuracy, "ou_acc": sc.ou_accuracy,
        "margin_mae": sc.margin_mae, "total_mae": sc.total_mae,
        "composite": sc.composite_score,
    } for sc in scorecards]

    out_path = Path(".context") / f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    include_detailed = "--detailed" in sys.argv
    run_full_evaluation(include_detailed)
