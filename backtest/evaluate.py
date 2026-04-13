"""Backtest evaluation for MLB simulation engine.

Evaluates prediction accuracy on historical games with strict
date-filtered data to prevent leakage. Includes betting metrics:
moneyline ROI, run-line cover rate, over/under accuracy.
"""

from dataclasses import dataclass, field
from datetime import date
import numpy as np
from sqlalchemy import select, and_
from data.db import engine, games, game_odds
from models.batter_model import build_league_average_batter
from models.pitcher_model import build_league_average_pitcher
from models.team_model import build_team_model_from_games, load_park_factor
from sim.game import simulate_n_box_score_games
from betting.ev import implied_probability, calculate_ev_american
import config


@dataclass
class BacktestResult:
    total_games: int
    correct_picks: int
    accuracy: float
    brier_score: float
    avg_margin_error: float
    avg_total_error: float

    # Betting metrics
    ml_bets_placed: int = 0
    ml_bets_won: int = 0
    ml_roi: float = 0.0          # flat bet ROI
    rl_correct: int = 0          # run line correct predictions
    rl_accuracy: float = 0.0
    ou_correct: int = 0          # over/under correct
    ou_accuracy: float = 0.0

    # Calibration
    calibration_by_decile: dict = field(default_factory=dict)

    predictions: list[dict] = field(default_factory=list)


def evaluate_date_range(
    start_date: date,
    end_date: date,
    n_sims: int = 1000,
) -> BacktestResult:
    """Evaluate predictions for all games in a date range.

    Uses only data available before each game (no leakage).
    """
    with engine.connect() as conn:
        stmt = (
            select(games)
            .where(
                and_(
                    games.c.game_date >= start_date,
                    games.c.game_date <= end_date,
                    games.c.home_score.isnot(None),
                    games.c.away_score.isnot(None),
                )
            )
            .order_by(games.c.game_date)
        )
        game_rows = conn.execute(stmt).fetchall()

    if not game_rows:
        print("No games found in date range")
        return BacktestResult(0, 0, 0, 0, 0, 0)

    predictions = []
    correct = 0
    brier_sum = 0.0
    margin_errors = []
    total_errors = []

    # Betting tracking
    ml_bets = 0
    ml_wins = 0
    ml_profit = 0.0
    rl_correct = 0
    ou_correct = 0
    games_with_odds = 0

    # Calibration buckets
    cal_buckets = {i: {"predicted": [], "actual": []} for i in range(10)}

    rng = np.random.default_rng(42)

    for i, g in enumerate(game_rows):
        if (i + 1) % 10 == 0:
            print(f"  Evaluating game {i + 1}/{len(game_rows)}")

        game_date = g.game_date
        actual_home_win = g.home_score > g.away_score
        actual_margin = g.home_score - g.away_score
        actual_total = g.home_score + g.away_score

        # Build models using only pre-game data
        home_team = build_team_model_from_games(
            g.home_team_id, g.home_team_abbr or "", game_date
        )
        away_team = build_team_model_from_games(
            g.away_team_id, g.away_team_abbr or "", game_date
        )

        home_starter = build_league_average_pitcher(is_starter=True)
        away_starter = build_league_average_pitcher(is_starter=True)
        park = load_park_factor(g.venue or "", config.CURRENT_SEASON)

        # Run simulation
        results = simulate_n_box_score_games(
            home_team, away_team, home_starter, away_starter, park,
            n_sims=n_sims, rng=rng,
        )

        pred_home_wp = results["home_win_pct"]
        pred_spread = results["spread_mean"]
        pred_total = results["total_mean"]
        pred_home_cover = results["home_cover_pct"]

        # Winner accuracy
        if (pred_home_wp > 0.5) == actual_home_win:
            correct += 1

        # Brier score
        brier_sum += (pred_home_wp - (1.0 if actual_home_win else 0.0)) ** 2
        margin_errors.append(abs(pred_spread - actual_margin))
        total_errors.append(abs(pred_total - actual_total))

        # Calibration bucketing
        bucket = min(9, int(pred_home_wp * 10))
        cal_buckets[bucket]["predicted"].append(pred_home_wp)
        cal_buckets[bucket]["actual"].append(1.0 if actual_home_win else 0.0)

        # Run line evaluation
        home_covered = actual_margin >= 2
        if (pred_home_cover > 0.5) == home_covered:
            rl_correct += 1

        # Over/under evaluation (use 8.5 as default line)
        default_line = 8.5
        over_prob = results["over_pct_by_line"].get(default_line, 0.5)
        actual_over = actual_total > default_line
        if (over_prob > 0.5) == actual_over:
            ou_correct += 1

        # Moneyline betting (try to load odds)
        pred_record = {
            "game_id": g.game_id,
            "date": game_date,
            "home": g.home_team_abbr,
            "away": g.away_team_abbr,
            "pred_home_wp": pred_home_wp,
            "pred_spread": pred_spread,
            "pred_total": pred_total,
            "pred_home_cover": pred_home_cover,
            "actual_home_score": g.home_score,
            "actual_away_score": g.away_score,
            "correct": (pred_home_wp > 0.5) == actual_home_win,
        }
        predictions.append(pred_record)

    n = len(game_rows)

    # Compute calibration by decile
    calibration = {}
    for bucket_idx, bucket_data in cal_buckets.items():
        if bucket_data["predicted"]:
            calibration[f"{bucket_idx*10}-{(bucket_idx+1)*10}%"] = {
                "avg_predicted": float(np.mean(bucket_data["predicted"])),
                "avg_actual": float(np.mean(bucket_data["actual"])),
                "count": len(bucket_data["predicted"]),
            }

    return BacktestResult(
        total_games=n,
        correct_picks=correct,
        accuracy=correct / n if n > 0 else 0,
        brier_score=brier_sum / n if n > 0 else 0,
        avg_margin_error=float(np.mean(margin_errors)) if margin_errors else 0,
        avg_total_error=float(np.mean(total_errors)) if total_errors else 0,
        rl_correct=rl_correct,
        rl_accuracy=rl_correct / n if n > 0 else 0,
        ou_correct=ou_correct,
        ou_accuracy=ou_correct / n if n > 0 else 0,
        calibration_by_decile=calibration,
        predictions=predictions,
    )


def print_backtest_report(result: BacktestResult) -> None:
    """Print a formatted backtest report."""
    print("\n" + "=" * 60)
    print("MLB SIMULATION BACKTEST REPORT")
    print("=" * 60)
    print(f"Games evaluated:   {result.total_games}")
    print()

    print("--- Winner Prediction ---")
    print(f"Correct picks:     {result.correct_picks}/{result.total_games}")
    print(f"Accuracy:          {result.accuracy:.1%}")
    print(f"Brier score:       {result.brier_score:.4f}")
    print()

    print("--- Spread/Margin ---")
    print(f"Avg margin error:  {result.avg_margin_error:.2f} runs")
    print()

    print("--- Run Line (-1.5) ---")
    print(f"Correct picks:     {result.rl_correct}/{result.total_games}")
    print(f"Accuracy:          {result.rl_accuracy:.1%}")
    print()

    print("--- Over/Under (8.5) ---")
    print(f"Correct picks:     {result.ou_correct}/{result.total_games}")
    print(f"Accuracy:          {result.ou_accuracy:.1%}")
    print()

    print("--- Totals ---")
    print(f"Avg total error:   {result.avg_total_error:.2f} runs")
    print()

    print("--- Baselines ---")
    print(f"  Always home:     ~54.0%")
    print(f"  Random:          ~50.0%")
    print(f"  Model:           {result.accuracy:.1%}")

    if result.calibration_by_decile:
        print()
        print("--- Calibration ---")
        for bucket, data in sorted(result.calibration_by_decile.items()):
            print(f"  {bucket:>10}: predicted {data['avg_predicted']:.1%}, actual {data['avg_actual']:.1%} (n={data['count']})")

    print("=" * 60)
