"""Command-line interface for MLB simulator."""

import click
from datetime import date


@click.group()
def cli():
    """MLB Simulation Engine - predict game outcomes and find betting edges."""
    pass


@cli.command()
def init_db():
    """Initialize database tables."""
    from data.db import init_db
    init_db()


@cli.command()
@click.argument("season", type=int, default=2024)
def fetch(season):
    """Fetch and load a season of MLB data."""
    from data.load import load_season, load_default_park_factors
    load_default_park_factors()
    load_season(season)


@cli.command()
@click.option("--sims", default=3000, help="Number of simulations")
def smoke_test(sims):
    """Run a smoke test with league-average teams."""
    import numpy as np
    from models.batter_model import build_league_average_batter
    from models.pitcher_model import build_league_average_pitcher
    from models.team_model import ParkFactor

    lineup = [build_league_average_batter() for _ in range(9)]
    for i, b in enumerate(lineup):
        b.player_id = i + 1

    starter = build_league_average_pitcher(is_starter=True)
    bullpen = [build_league_average_pitcher(is_starter=False) for _ in range(5)]
    for i, p in enumerate(bullpen):
        p.player_id = 100 + i

    park = ParkFactor(venue="Neutral Park", overall_factor=1.0,
                      hr_factor=1.0, h_factor=1.0, bb_factor=1.0)

    rng = np.random.default_rng(42)

    # Detailed simulation
    from sim.detailed_game import simulate_n_detailed_games
    print(f"Running {sims} detailed game simulations...")
    results = simulate_n_detailed_games(
        home_lineup=lineup, away_lineup=lineup,
        home_starter=starter, away_starter=starter,
        home_bullpen=bullpen, away_bullpen=bullpen,
        park=park, n_sims=sims, rng=rng,
    )

    print(f"\n{'='*55}")
    print("SMOKE TEST - Detailed Model (League Avg vs League Avg)")
    print(f"{'='*55}")
    print(f"Simulations:     {sims}")
    print(f"Home win %:      {results['home_win_pct']:.1%}")
    print(f"Home score avg:  {results['home_score_mean']:.2f}")
    print(f"Away score avg:  {results['away_score_mean']:.2f}")
    print(f"Total avg:       {results['total_mean']:.2f}")
    print(f"Spread mean:     {results['spread_mean']:+.2f}")
    print(f"Home cover -1.5: {results['home_cover_pct']:.1%}")

    # O/U at common lines
    print(f"\nOver/Under probabilities:")
    for line in [7.5, 8.0, 8.5, 9.0, 9.5]:
        if line in results["over_pct_by_line"]:
            print(f"  O/U {line}:  {results['over_pct_by_line'][line]:.1%} over")

    # Box score model
    from sim.game import simulate_n_box_score_games
    import config

    class AvgTeam:
        team_id = 1; team_abbr = "AVG"; season = 2024; wins = 81; losses = 81
        runs_per_game = config.LEAGUE_AVG_RUNS_PER_GAME
        runs_allowed_per_game = config.LEAGUE_AVG_RUNS_PER_GAME
        wrc_plus = 100.0; fip_minus = 100.0
        team_k_pct = config.LEAGUE_AVG_K_PCT; team_bb_pct = config.LEAGUE_AVG_BB_PCT
        team_hr_per_fb = config.LEAGUE_AVG_HR_PER_FB; team_babip = config.LEAGUE_AVG_BABIP
        win_pct = 0.500

    rng2 = np.random.default_rng(42)
    box_results = simulate_n_box_score_games(
        AvgTeam(), AvgTeam(), starter, starter, park, n_sims=sims, rng=rng2,
    )

    print(f"\n{'='*55}")
    print("SMOKE TEST - Box Score Model (League Avg vs League Avg)")
    print(f"{'='*55}")
    print(f"Home win %:      {box_results['home_win_pct']:.1%}")
    print(f"Home score avg:  {box_results['home_score_mean']:.2f}")
    print(f"Away score avg:  {box_results['away_score_mean']:.2f}")
    print(f"Total avg:       {box_results['total_mean']:.2f}")
    print(f"Spread mean:     {box_results['spread_mean']:+.2f}")
    print(f"Home cover -1.5: {box_results['home_cover_pct']:.1%}")

    # Betting module test
    from betting.predictions import BettingPrediction
    pred = BettingPrediction(
        home_win_prob=results["home_win_pct"],
        away_win_prob=1.0 - results["home_win_pct"],
        home_cover_prob=results["home_cover_pct"],
        away_cover_prob=1.0 - results["home_cover_pct"],
        total_mean=results["total_mean"],
        total_std=results.get("total_std", 3.5),
        over_prob=results["over_pct_by_line"],
        home_score_mean=results["home_score_mean"],
        away_score_mean=results["away_score_mean"],
    )

    print(f"\n{'='*55}")
    print("BETTING PREDICTION")
    print(f"{'='*55}")
    print(pred.summary())

    # Evaluate against hypothetical odds
    ml_eval = pred.evaluate_moneyline(-130, +110)
    print(f"\nMoneyline EV vs -130/+110:")
    print(f"  Home EV: {ml_eval['home_ev']:+.3f}  Away EV: {ml_eval['away_ev']:+.3f}")
    print(f"  Best bet: {ml_eval['best_bet']} (edge: {ml_eval['has_edge']})")

    rl_eval = pred.evaluate_run_line()
    print(f"\nRun Line EV vs -110/-110:")
    print(f"  Home -1.5 EV: {rl_eval['home_cover_ev']:+.3f}  Away +1.5 EV: {rl_eval['away_cover_ev']:+.3f}")

    ou_eval = pred.evaluate_total(8.5)
    print(f"\nO/U 8.5 EV vs -110/-110:")
    print(f"  Over EV: {ou_eval['over_ev']:+.3f}  Under EV: {ou_eval['under_ev']:+.3f}")


@cli.command()
@click.argument("start_date")
@click.argument("end_date")
@click.option("--sims", default=1000, help="Simulations per game")
def backtest(start_date, end_date, sims):
    """Run backtest evaluation on historical games."""
    from backtest.evaluate import evaluate_date_range, print_backtest_report
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    result = evaluate_date_range(start, end, n_sims=sims)
    print_backtest_report(result)


@cli.command()
@click.argument("season", type=int, default=2026)
def fetch_season(season):
    """Fetch all games for a season to .context/ (no database needed)."""
    from scripts.fetch_season import fetch_season_to_date
    import json
    from pathlib import Path

    full_games, summaries = fetch_season_to_date(season)
    context_dir = Path(".context")
    context_dir.mkdir(exist_ok=True)
    with open(context_dir / f"season_{season}.json", "w") as f:
        json.dump(full_games, f, indent=2)
    with open(context_dir / f"season_{season}_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(full_games)} games")


@cli.command()
@click.argument("start_date", required=False)
@click.argument("end_date", required=False)
@click.option("--sims", default=1000, help="Simulations per game")
@click.option("--season", default=2026, help="Season year")
def backtest_json(start_date, end_date, sims, season):
    """Run backtest from JSON data (no database needed)."""
    from backtest.json_backtest import evaluate_from_json, print_backtest_report
    result = evaluate_from_json(
        season_file=f".context/season_{season}.json",
        start_date=start_date,
        end_date=end_date,
        n_sims=sims,
    )
    print_backtest_report(result)


@cli.command()
def train():
    """Train ensemble weights via grid search."""
    from scripts.train_ensemble import train_ensemble
    train_ensemble()


if __name__ == "__main__":
    cli()
