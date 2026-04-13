"""Daily MLB prediction scanner — finds high-edge betting opportunities.

Fetches today's schedule, runs simulations for each game, and identifies
the strongest edges on moneyline, run line, and totals.

Usage:
    python scripts/daily_picks.py              # today's games
    python scripts/daily_picks.py 2026-04-10   # specific date
"""

import sys
import json
import time
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")

import numpy as np
import statsapi

from backtest.json_backtest import (
    build_team_model_from_json, build_pitcher_from_json,
    _build_pitcher_index, load_park_factors_from_json, TEAM_ABBR,
)
from models.pitcher_model import build_league_average_pitcher
from models.team_model import ParkFactor
from sim.game import simulate_n_box_score_games, _starter_quality_factor
from betting.ev import (
    win_probability_to_moneyline, format_odds,
    calculate_ev_american, fractional_kelly, american_to_decimal,
    remove_vig, implied_probability,
)
import config


def load_season_data():
    """Load 2025 + 2026 season data and build indexes."""
    prior_pitcher_index = None
    if Path(".context/season_2025.json").exists():
        with open(".context/season_2025.json") as f:
            prior_games = json.load(f)
        prior_pitcher_index = _build_pitcher_index(prior_games)
        print(f"  2025 priors: {len(prior_games)} games, {len(prior_pitcher_index)} pitchers")

    with open(".context/season_2026.json") as f:
        current_games = json.load(f)
    pitcher_index = _build_pitcher_index(current_games)
    print(f"  2026 season: {len(current_games)} games, {len(pitcher_index)} pitchers")

    park_factors = load_park_factors_from_json()
    return current_games, pitcher_index, prior_pitcher_index, park_factors


def fetch_todays_schedule(target_date: str) -> list[dict]:
    """Fetch today's MLB schedule from the Stats API."""
    sched = statsapi.schedule(start_date=target_date, end_date=target_date)
    games = []
    for g in sched:
        if g.get("game_type") != "R":
            continue
        games.append({
            "game_id": str(g["game_id"]),
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_id": g["home_id"],
            "away_id": g["away_id"],
            "venue": g.get("venue_name", ""),
            "home_starter": g.get("home_probable_pitcher", ""),
            "away_starter": g.get("away_probable_pitcher", ""),
            "status": g.get("status", ""),
            "game_time": g.get("game_datetime", ""),
        })
    return games


def predict_game(
    game: dict,
    all_games: list[dict],
    pitcher_index: dict,
    prior_pitcher_index: dict | None,
    park_factors: dict[str, ParkFactor],
    target_date: str,
    n_sims: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run full prediction for a single game."""
    if rng is None:
        rng = np.random.default_rng()

    neutral = ParkFactor(venue="Neutral", overall_factor=1.0,
                         hr_factor=1.0, h_factor=1.0, bb_factor=1.0)
    league_avg = build_league_average_pitcher(is_starter=True)

    # Build team models
    home_team = build_team_model_from_json(
        game["home_id"], game["home_team"], all_games, target_date, None
    )
    away_team = build_team_model_from_json(
        game["away_id"], game["away_team"], all_games, target_date, None
    )

    # Build pitcher models
    home_sp = league_avg
    away_sp = league_avg
    home_sp_name = game.get("home_starter", "")
    away_sp_name = game.get("away_starter", "")

    if home_sp_name:
        sp = build_pitcher_from_json(home_sp_name, pitcher_index, target_date, prior_pitcher_index)
        if sp:
            home_sp = sp

    if away_sp_name:
        sp = build_pitcher_from_json(away_sp_name, pitcher_index, target_date, prior_pitcher_index)
        if sp:
            away_sp = sp

    # Park factor
    home_abbr = TEAM_ABBR.get(game["home_team"], "")
    park = park_factors.get(home_abbr, neutral)

    # Run simulation
    results = simulate_n_box_score_games(
        home_team, away_team, home_sp, away_sp, park,
        n_sims=n_sims, rng=rng,
    )

    # Compute quality factors for display
    home_qf = _starter_quality_factor(home_sp)
    away_qf = _starter_quality_factor(away_sp)

    return {
        "game": game,
        "home_team_model": home_team,
        "away_team_model": away_team,
        "home_sp_name": home_sp_name or "TBD",
        "away_sp_name": away_sp_name or "TBD",
        "home_sp_qf": home_qf,
        "away_sp_qf": away_qf,
        "park_factor": park.overall_factor,
        "park_hr_factor": park.hr_factor,
        "results": results,
    }


def classify_edge(pred: dict) -> list[dict]:
    """Identify specific betting edges from a prediction."""
    edges = []
    r = pred["results"]
    park_f = pred["park_factor"]
    park_hr = pred["park_hr_factor"]
    home_qf = pred["home_sp_qf"]
    away_qf = pred["away_sp_qf"]
    home_rpg = pred["home_team_model"].runs_per_game
    away_rpg = pred["away_team_model"].runs_per_game

    # Edge 1: TOTALS in pitcher's parks with elite starters
    if park_f < 0.99 and (home_qf < 0.85 or away_qf < 0.85):
        pred_total = r["total_mean"]
        # Check common O/U lines
        for line in [7.5, 8.0, 8.5, 9.0]:
            over_pct = r["over_pct_by_line"].get(line, 0.5)
            under_pct = 1.0 - over_pct
            if under_pct >= 0.55:
                ev = calculate_ev_american(under_pct, -110)
                if ev > 0.01:
                    edges.append({
                        "type": "UNDER",
                        "line": line,
                        "prob": under_pct,
                        "ev_vs_110": ev,
                        "reason": f"Pitcher's park (PF={park_f:.2f}) + elite SP (QF={min(home_qf,away_qf):.2f})",
                        "confidence": "HIGH" if under_pct >= 0.58 else "MEDIUM",
                    })

    # Edge 2: TOTALS in hitter's parks
    if park_f > 1.03 and park_hr > 1.05:
        pred_total = r["total_mean"]
        for line in [8.5, 9.0, 9.5, 10.0]:
            over_pct = r["over_pct_by_line"].get(line, 0.5)
            if over_pct >= 0.55:
                ev = calculate_ev_american(over_pct, -110)
                if ev > 0.01:
                    edges.append({
                        "type": "OVER",
                        "line": line,
                        "prob": over_pct,
                        "ev_vs_110": ev,
                        "reason": f"Hitter's park (PF={park_f:.2f}, HR={park_hr:.2f})",
                        "confidence": "HIGH" if over_pct >= 0.58 else "MEDIUM",
                    })

    # Edge 3: RUN LINE when pitcher mismatch is large
    qf_diff = away_qf - home_qf  # positive = home has better pitcher
    if abs(qf_diff) > 0.15:
        home_cover = r["home_cover_pct"]
        away_cover = 1.0 - home_cover
        if qf_diff > 0.15 and home_cover > 0.42:
            ev = calculate_ev_american(home_cover, -110)
            if ev > 0.01:
                edges.append({
                    "type": "HOME -1.5",
                    "prob": home_cover,
                    "ev_vs_110": ev,
                    "reason": f"Pitcher mismatch: Home QF={home_qf:.2f} vs Away QF={away_qf:.2f}",
                    "confidence": "MEDIUM",
                })
        elif qf_diff < -0.15 and away_cover > 0.58:
            ev = calculate_ev_american(away_cover, -110)
            if ev > 0.01:
                edges.append({
                    "type": "AWAY +1.5",
                    "prob": away_cover,
                    "ev_vs_110": ev,
                    "reason": f"Pitcher mismatch: Away QF={away_qf:.2f} vs Home QF={home_qf:.2f}",
                    "confidence": "MEDIUM",
                })

    # Edge 4: MONEYLINE when team + pitcher + park all align
    home_wp = r["home_win_pct"]
    away_wp = 1.0 - home_wp
    if home_wp > 0.56 and home_qf < 0.85 and home_rpg > away_rpg:
        edges.append({
            "type": "HOME ML",
            "prob": home_wp,
            "ev_vs_110": calculate_ev_american(home_wp, -110),
            "fair_ml": win_probability_to_moneyline(home_wp),
            "reason": f"Strong home team (RPG={home_rpg:.1f}) + elite SP (QF={home_qf:.2f})",
            "confidence": "MEDIUM",
        })
    elif away_wp > 0.54 and away_qf < 0.85:
        edges.append({
            "type": "AWAY ML",
            "prob": away_wp,
            "ev_vs_110": calculate_ev_american(away_wp, -110),
            "fair_ml": win_probability_to_moneyline(away_wp),
            "reason": f"Elite away SP (QF={away_qf:.2f}) + away team edge",
            "confidence": "MEDIUM",
        })

    return edges


def print_predictions(predictions: list[dict], target_date: str):
    """Print formatted daily predictions."""
    print()
    print("=" * 75)
    print(f"  MLB DAILY PREDICTIONS — {target_date}")
    print("=" * 75)
    print()

    all_edges = []

    for pred in predictions:
        g = pred["game"]
        r = pred["results"]
        ht = pred["home_team_model"]
        at = pred["away_team_model"]

        home_wp = r["home_win_pct"]
        away_wp = 1.0 - home_wp
        fair_home = win_probability_to_moneyline(home_wp)
        fair_away = win_probability_to_moneyline(away_wp)

        print(f"  {g['away_team']:>25} @ {g['home_team']}")
        print(f"  {'':>25}   Venue: {g['venue']}")
        print(f"  SP: {pred['away_sp_name']:>24} vs {pred['home_sp_name']}")
        print(f"  QF: {'':>19}{pred['away_sp_qf']:>5.2f} vs {pred['home_sp_qf']:.2f}")
        print(f"  Team RPG: {'':>14}{at.runs_per_game:>5.2f} vs {ht.runs_per_game:.2f}")
        print(f"  Park: PF={pred['park_factor']:.2f} HR={pred['park_hr_factor']:.2f}")
        print()
        print(f"  ML:    Home {home_wp:.1%} ({format_odds(fair_home)})  |  Away {away_wp:.1%} ({format_odds(fair_away)})")
        print(f"  RL:    Home -1.5 {r['home_cover_pct']:.1%}  |  Away +1.5 {1-r['home_cover_pct']:.1%}")
        print(f"  Total: {r['total_mean']:.1f} runs (σ={r['total_std']:.1f})")

        # O/U at common lines
        ou_lines = []
        for line in [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]:
            op = r["over_pct_by_line"].get(line, 0.5)
            if 0.35 < op < 0.65:
                ou_lines.append(f"O{line}:{op:.0%}")
        if ou_lines:
            print(f"  O/U:   {' | '.join(ou_lines)}")

        # Find edges
        edges = classify_edge(pred)
        if edges:
            print()
            for e in edges:
                star = "***" if e["confidence"] == "HIGH" else " * "
                ml_str = f" (fair: {format_odds(e.get('fair_ml', 0))})" if "fair_ml" in e else ""
                print(f"  {star} EDGE: {e['type']:<12} P={e['prob']:.1%}  EV={e['ev_vs_110']:+.3f}{ml_str}")
                print(f"       {e['reason']}")
            all_edges.extend([(pred, e) for e in edges])

        print()
        print("  " + "-" * 71)
        print()

    # Summary of all edges
    if all_edges:
        print()
        print("=" * 75)
        print("  TOP EDGES SUMMARY")
        print("=" * 75)
        print()

        # Sort by EV
        all_edges.sort(key=lambda x: -x[1]["ev_vs_110"])

        for pred, edge in all_edges:
            g = pred["game"]
            conf_marker = "!!!" if edge["confidence"] == "HIGH" else " ! "
            print(f"  {conf_marker} {edge['type']:<12} {g['away_team'][:15]:>15} @ {g['home_team'][:15]:<15}  "
                  f"P={edge['prob']:.1%}  EV={edge['ev_vs_110']:+.3f}  [{edge['confidence']}]")

        high_conf = [e for _, e in all_edges if e["confidence"] == "HIGH"]
        med_conf = [e for _, e in all_edges if e["confidence"] == "MEDIUM"]
        print()
        print(f"  Total edges: {len(all_edges)} ({len(high_conf)} HIGH, {len(med_conf)} MEDIUM)")
    else:
        print("  No strong edges found today.")

    print()
    print("=" * 75)


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    print(f"MLB Daily Picks — {target_date}")
    print(f"Loading data...")

    current_games, pitcher_index, prior_pitcher_index, park_factors = load_season_data()

    print(f"Fetching schedule for {target_date}...")
    schedule = fetch_todays_schedule(target_date)
    print(f"Found {len(schedule)} games")

    if not schedule:
        print("No games scheduled.")
        return

    rng = np.random.default_rng()
    predictions = []

    for i, game in enumerate(schedule):
        print(f"  Simulating {game['away_team'][:12]} @ {game['home_team'][:12]}... ({i+1}/{len(schedule)})")
        pred = predict_game(
            game, current_games, pitcher_index, prior_pitcher_index,
            park_factors, target_date, n_sims=n_sims, rng=rng,
        )
        predictions.append(pred)

    print_predictions(predictions, target_date)

    # Save predictions to JSON
    output = []
    for pred in predictions:
        g = pred["game"]
        r = pred["results"]
        output.append({
            "game_id": g["game_id"],
            "away": g["away_team"],
            "home": g["home_team"],
            "venue": g["venue"],
            "away_sp": pred["away_sp_name"],
            "home_sp": pred["home_sp_name"],
            "home_wp": r["home_win_pct"],
            "home_cover": r["home_cover_pct"],
            "total_mean": r["total_mean"],
            "total_std": r["total_std"],
            "over_pct_by_line": r["over_pct_by_line"],
            "edges": classify_edge(pred),
        })

    out_path = Path(".context") / f"picks_{target_date.replace('-','_')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
