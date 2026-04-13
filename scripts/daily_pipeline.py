"""Full daily MLB prediction pipeline.

Runs the complete workflow:
1. Fetch yesterday's results → update season data
2. Fetch Kalshi market prices (ML, totals, spread, HR)
3. Fetch confirmed lineups from MLB Stats API
4. Run simulations for today's games
5. Compare model predictions to Kalshi prices → find edges
6. Output actionable report with sizing recommendations

Usage:
    python scripts/daily_pipeline.py              # today's games
    python scripts/daily_pipeline.py 2026-04-14   # specific date
"""

import sys
import json
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import statsapi

from scripts.fetch_daily import fetch_games_for_date
from data.kalshi import fetch_mlb_markets, format_kalshi_report
from data.lineup_fetcher import fetch_lineups_from_schedule, build_game_models
from backtest.json_backtest import (
    build_team_model_from_json, build_pitcher_from_json,
    _build_pitcher_index, load_park_factors_from_json, TEAM_ABBR,
)
from models.pitcher_model import build_league_average_pitcher
from models.team_model import ParkFactor
from sim.game import simulate_n_box_score_games, _starter_quality_factor
from sim.detailed_game import simulate_n_detailed_games
from betting.ev import calculate_ev, fractional_kelly, win_probability_to_moneyline, format_odds
import config


def load_all_data():
    """Load season data, prior season, and indexes."""
    print("  Loading season data...")
    with open(".context/season_2026.json") as f:
        current_games = json.load(f)

    prior_games = None
    prior_pitcher_idx = None
    if Path(".context/season_2025.json").exists():
        with open(".context/season_2025.json") as f:
            prior_games = json.load(f)
        prior_pitcher_idx = _build_pitcher_index(prior_games)

    pitcher_idx = _build_pitcher_index(current_games)
    park_factors = load_park_factors_from_json()

    with open(".context/season_2026_summary.json") as f:
        summaries = json.load(f)
    starter_lookup = {s["game_id"]: (s.get("home_starter", ""), s.get("away_starter", ""))
                      for s in summaries}

    print(f"  Loaded {len(current_games)} games, {len(pitcher_idx)} pitchers")
    return current_games, prior_games, pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors


def step1_fetch_yesterday(target_date: str, current_games: list):
    """Fetch yesterday's results and update season file."""
    yesterday = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    print(f"\n[1/5] Fetching yesterday's results ({yesterday})...")

    existing_ids = {g["game_id"] for g in current_games}
    new_games = fetch_games_for_date(yesterday)
    added = 0
    for g in new_games:
        if g["game_id"] not in existing_ids:
            current_games.append(g)
            existing_ids.add(g["game_id"])
            added += 1

    if added:
        with open(".context/season_2026.json", "w") as f:
            json.dump(current_games, f)
        print(f"  Added {added} new games (total: {len(current_games)})")
    else:
        print(f"  No new games to add (already had {len(current_games)})")

    return current_games


def step2_fetch_kalshi(target_date: str) -> dict:
    """Fetch Kalshi market prices."""
    print(f"\n[2/5] Fetching Kalshi markets for {target_date}...")
    markets = fetch_mlb_markets(target_date)

    # Also check tomorrow for totals/spread that post early
    tomorrow = (date.fromisoformat(target_date) + timedelta(days=1)).isoformat()
    print(f"  Also checking tomorrow ({tomorrow}) for totals/spread...")
    tomorrow_markets = fetch_mlb_markets(tomorrow)

    return {"today": markets, "tomorrow": tomorrow_markets}


def step3_fetch_lineups(target_date: str) -> dict:
    """Fetch confirmed lineups from MLB Stats API."""
    print(f"\n[3/5] Fetching lineups for {target_date}...")
    lineups = fetch_lineups_from_schedule(target_date)
    print(f"  Found lineups for {len(lineups)} games")
    return lineups


def step4_run_simulations(
    target_date: str,
    current_games: list,
    pitcher_idx: dict,
    prior_pitcher_idx: dict | None,
    starter_lookup: dict,
    park_factors: dict,
    prior_games: list | None,
) -> list:
    """Run simulations for all today's games."""
    print(f"\n[4/5] Running simulations...")
    neutral = ParkFactor(venue="N", overall_factor=1.0, hr_factor=1.0, h_factor=1.0, bb_factor=1.0)
    league_avg = build_league_average_pitcher(is_starter=True)
    rng = np.random.default_rng()

    # Get schedule
    sched = statsapi.schedule(start_date=target_date, end_date=target_date)
    games = [g for g in sched if g.get("game_type") == "R"]

    predictions = []
    for i, g in enumerate(games):
        away = g["away_name"]
        home = g["home_name"]
        print(f"  [{i+1}/{len(games)}] {away[:15]} @ {home[:15]}...", end="", flush=True)

        # Build team models
        ht = build_team_model_from_json(g["home_id"], home, current_games, target_date, None)
        at = build_team_model_from_json(g["away_id"], away, current_games, target_date, None)

        # Build pitcher models
        hs_name = g.get("home_probable_pitcher", "")
        as_name = g.get("away_probable_pitcher", "")
        hs = league_avg
        as_ = league_avg

        if hs_name:
            sp = build_pitcher_from_json(hs_name, pitcher_idx, target_date, prior_pitcher_idx)
            if sp: hs = sp
        if as_name:
            sp = build_pitcher_from_json(as_name, pitcher_idx, target_date, prior_pitcher_idx)
            if sp: as_ = sp

        park = park_factors.get(TEAM_ABBR.get(home, ""), neutral)

        # Run box score sim
        results = simulate_n_box_score_games(ht, at, hs, as_, park, n_sims=5000, rng=rng)

        home_qf = _starter_quality_factor(hs)
        away_qf = _starter_quality_factor(as_)

        pred = {
            "game_id": str(g["game_id"]),
            "away": away,
            "home": home,
            "venue": g.get("venue_name", ""),
            "away_sp": as_name or "TBD",
            "home_sp": hs_name or "TBD",
            "away_sp_qf": away_qf,
            "home_sp_qf": home_qf,
            "home_wp": results["home_win_pct"],
            "away_wp": 1 - results["home_win_pct"],
            "home_cover": results["home_cover_pct"],
            "total_mean": results["total_mean"],
            "total_std": results["total_std"],
            "spread_mean": results["spread_mean"],
            "over_pct_by_line": results["over_pct_by_line"],
            "park_factor": park.overall_factor,
            "home_rpg": ht.runs_per_game,
            "away_rpg": at.runs_per_game,
        }
        predictions.append(pred)
        print(f" HW:{results['home_win_pct']:.0%} Total:{results['total_mean']:.1f}")

    return predictions


def step5_compare_and_report(predictions: list, kalshi_markets: dict, target_date: str):
    """Compare model predictions to Kalshi prices and output report."""
    print(f"\n[5/5] Generating report...")

    today_markets = kalshi_markets.get("today", {})
    tomorrow_markets = kalshi_markets.get("tomorrow", {})

    edges = []

    print()
    print("=" * 85)
    print(f"  MLB DAILY PIPELINE REPORT — {target_date}")
    print("=" * 85)

    # Kalshi market summary
    if today_markets:
        print(f"\n  KALSHI MARKETS (today):")
        print(format_kalshi_report(today_markets))

    if tomorrow_markets:
        print(f"\n  KALSHI MARKETS (tomorrow — totals/spreads):")
        print(format_kalshi_report(tomorrow_markets))

    # Model predictions vs Kalshi
    print(f"\n  {'='*80}")
    print(f"  MODEL vs KALSHI")
    print(f"  {'='*80}")
    print()
    print(f"  {'Game':<30} {'Model':>8} {'Kalshi':>8} {'Edge':>8} {'Type':>8} {'EV':>8}")
    print(f"  {'-'*80}")

    for pred in predictions:
        away_short = pred["away"][:12]
        home_short = pred["home"][:12]
        game_label = f"{away_short} @ {home_short}"

        # Try to match to Kalshi market
        matched = None
        for game_title, mkt in today_markets.items():
            # Fuzzy match on team names
            t = game_title.lower()
            if (pred["away"][:6].lower() in t or pred["home"][:6].lower() in t):
                matched = mkt
                break

        if matched and matched.get("ml"):
            # Find home/away sides
            for side, info in matched["ml"].items():
                mid = info.get("mid")
                if mid is None:
                    continue

                # Determine if this is home or away
                is_home_side = any(pred["home"][:4].upper().startswith(s[:3]) for s in [side])

                if is_home_side:
                    model_prob = pred["home_wp"]
                    kalshi_price = mid
                else:
                    model_prob = pred["away_wp"]
                    kalshi_price = mid

                edge = model_prob - kalshi_price
                if abs(edge) > 0.02:
                    ev = model_prob * (1.0 / kalshi_price - 1) - (1 - model_prob)
                    direction = "HOME ML" if (is_home_side and edge > 0) or (not is_home_side and edge < 0) else "AWAY ML"
                    if edge > 0:
                        edges.append({
                            "game": game_label,
                            "type": direction,
                            "model": model_prob,
                            "kalshi": kalshi_price,
                            "edge": edge,
                            "ev": ev,
                        })

        # Check totals (may be in tomorrow's markets)
        for markets_set in [today_markets, tomorrow_markets]:
            for game_title, mkt in markets_set.items():
                t = game_title.lower()
                if not (pred["away"][:6].lower() in t or pred["home"][:6].lower() in t):
                    continue
                for tot in mkt.get("total", []):
                    line = tot.get("line")
                    over_mid = None
                    if tot.get("over_bid") and tot.get("over_ask"):
                        over_mid = (tot["over_bid"] + tot["over_ask"]) / 2
                    elif tot.get("over_bid"):
                        over_mid = tot["over_bid"]

                    if line and over_mid:
                        model_over = pred["over_pct_by_line"].get(line, 0.5)
                        model_under = 1 - model_over
                        kalshi_under = 1 - over_mid

                        over_edge = model_over - over_mid
                        under_edge = model_under - kalshi_under

                        if under_edge > 0.05:
                            ev = model_under * (1.0 / kalshi_under - 1) - (1 - model_under)
                            edges.append({
                                "game": game_label,
                                "type": f"UNDER {line}",
                                "model": model_under,
                                "kalshi": kalshi_under,
                                "edge": under_edge,
                                "ev": ev,
                            })
                        elif over_edge > 0.05:
                            ev = model_over * (1.0 / over_mid - 1) - (1 - model_over)
                            edges.append({
                                "game": game_label,
                                "type": f"OVER {line}",
                                "model": model_over,
                                "kalshi": over_mid,
                                "edge": over_edge,
                                "ev": ev,
                            })

    # Print all predictions
    for pred in predictions:
        fair_ml = win_probability_to_moneyline(pred["home_wp"])
        print(f"  {pred['away'][:12]+' @ '+pred['home'][:12]:<30} "
              f"HW:{pred['home_wp']:>5.1%} "
              f"Tot:{pred['total_mean']:>5.1f} "
              f"SP:{pred['away_sp'][:10]}({pred['away_sp_qf']:.2f}) v {pred['home_sp'][:10]}({pred['home_sp_qf']:.2f})")

    # Print edges
    if edges:
        edges.sort(key=lambda x: -x["ev"])
        print()
        print(f"  {'='*80}")
        print(f"  TOP EDGES vs KALSHI")
        print(f"  {'='*80}")
        print()
        print(f"  {'Game':<25} {'Type':<15} {'Model':>8} {'Kalshi':>8} {'Edge':>8} {'EV':>8}")
        print(f"  {'-'*80}")
        for e in edges:
            conf = "!!!" if e["edge"] > 0.08 else " ! " if e["edge"] > 0.04 else "   "
            print(f"  {conf}{e['game']:<22} {e['type']:<15} {e['model']:>7.1%} {e['kalshi']:>7.1%} "
                  f"{e['edge']:>+7.1%} {e['ev']:>+7.1%}")

        # Sizing recommendations
        print()
        print(f"  SIZING (eighth-Kelly, max 3% bankroll):")
        for e in edges[:5]:
            if e["edge"] > 0.03 and e["ev"] > 0:
                dec_odds = 1.0 / e["kalshi"] if e["kalshi"] > 0 else 2.0
                kelly = fractional_kelly(e["model"], dec_odds)
                print(f"    {e['type']:<15} {e['game']:<25} Kelly: {kelly:.2%} of bankroll")
    else:
        print("\n  No significant edges found vs Kalshi prices.")

    print()
    print("=" * 85)

    return edges


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    print(f"MLB Daily Pipeline — {target_date}")
    print("=" * 50)

    # Load data
    print("\n[0/5] Loading data...")
    current_games, prior_games, pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors = load_all_data()

    # Step 1: Fetch yesterday
    current_games = step1_fetch_yesterday(target_date, current_games)
    # Rebuild indexes after update
    pitcher_idx = _build_pitcher_index(current_games)

    # Step 2: Fetch Kalshi
    kalshi_markets = step2_fetch_kalshi(target_date)

    # Step 3: Fetch lineups
    lineups = step3_fetch_lineups(target_date)

    # Step 4: Run simulations
    predictions = step4_run_simulations(
        target_date, current_games, pitcher_idx, prior_pitcher_idx,
        starter_lookup, park_factors, prior_games,
    )

    # Step 5: Compare and report
    edges = step5_compare_and_report(predictions, kalshi_markets, target_date)

    # Save results
    output = {
        "date": target_date,
        "predictions": predictions,
        "edges": edges,
        "kalshi_games": len(kalshi_markets.get("today", {})),
    }
    out_path = Path(".context") / f"pipeline_{target_date.replace('-', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Lineup status
    print(f"\nLINEUP STATUS:")
    print(f"  Lineups confirmed for {len(lineups)} games")
    print(f"  MLB lineups typically released 2-4 hours before first pitch")
    print(f"  Re-run pipeline after lineups are confirmed for updated predictions")


if __name__ == "__main__":
    main()
