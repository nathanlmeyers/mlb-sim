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


def _cache_kalshi_prices(kalshi_markets: dict, target_date: str):
    """Cache today's Kalshi prices for future backtesting."""
    history_path = Path(".context/kalshi_history.json")
    history = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    today_prices = {}
    for game_title, mkt in kalshi_markets.get("today", {}).items():
        ml = mkt.get("ml", {})
        prices = {}
        for side, info in ml.items():
            mid = info.get("mid")
            if mid:
                prices[side] = mid
        if prices:
            today_prices[game_title] = prices

        # Also cache totals
        totals = {}
        for t in mkt.get("total", []):
            if t.get("line") and t.get("over_bid"):
                over_mid = (t["over_bid"] + (t.get("over_ask") or t["over_bid"])) / 2
                totals[str(t["line"])] = over_mid
        if totals:
            today_prices.setdefault(game_title, {})["totals"] = totals

    if today_prices:
        history[target_date] = today_prices
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  Cached Kalshi prices for {len(today_prices)} games")


def _parse_market_odds_for_game(game_title: str, kalshi_markets: dict) -> dict | None:
    """Extract market odds for a specific game from Kalshi data.

    Returns {"home_win_prob": float, "away_win_prob": float, "total": float|None}
    or None if no data.
    """
    for markets_set in [kalshi_markets.get("today", {}), kalshi_markets.get("tomorrow", {})]:
        for mkt_title, mkt in markets_set.items():
            # Fuzzy match on game title
            if not any(word in mkt_title.lower() for word in game_title.lower().split()[:2]):
                continue

            ml = mkt.get("ml", {})
            if len(ml) < 2:
                continue

            # Get the two sides' midpoints
            sides = list(ml.values())
            if not all(s.get("mid") for s in sides):
                continue

            # First side listed is typically away on Kalshi
            probs = [s["mid"] for s in sides]
            # Normalize to sum to 1
            total_p = sum(probs)
            if total_p > 0:
                probs = [p / total_p for p in probs]

            # Parse total from totals markets
            game_total = None
            for t in mkt.get("total", []):
                if t.get("line") and t.get("over_bid"):
                    over_mid = (t["over_bid"] + (t.get("over_ask") or t["over_bid"])) / 2
                    if abs(over_mid - 0.5) < 0.15:  # closest to 50/50 is the market line
                        game_total = t["line"]

            return {
                "home_win_prob": probs[1] if len(probs) > 1 else probs[0],
                "away_win_prob": probs[0] if len(probs) > 1 else 1 - probs[0],
                "total": game_total,
                "raw_ml": ml,
            }

    return None


def step4_run_simulations(
    target_date: str,
    current_games: list,
    pitcher_idx: dict,
    prior_pitcher_idx: dict | None,
    starter_lookup: dict,
    park_factors: dict,
    prior_games: list | None,
    kalshi_markets: dict | None = None,
    only_game_id: str | None = None,
) -> list:
    """Run simulations for today's games.

    If only_game_id is set, run for just that one game.
    """
    print(f"\n[4/5] Running simulations...")
    neutral = ParkFactor(venue="N", overall_factor=1.0, hr_factor=1.0, h_factor=1.0, bb_factor=1.0)
    league_avg = build_league_average_pitcher(is_starter=True)
    rng = np.random.default_rng()

    # Get schedule
    sched = statsapi.schedule(start_date=target_date, end_date=target_date)
    games = [g for g in sched if g.get("game_type") == "R"]
    if only_game_id:
        games = [g for g in games if str(g.get("game_id")) == str(only_game_id)]
        if not games:
            print(f"  Game ID {only_game_id} not found in today's schedule")
            return []

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

        # Look up Kalshi market odds for this game
        game_market = None
        if kalshi_markets:
            game_market = _parse_market_odds_for_game(
                f"{away} {home}", kalshi_markets
            )

        # Run box score sim
        results = simulate_n_box_score_games(ht, at, hs, as_, park, n_sims=5000, rng=rng)

        home_qf = _starter_quality_factor(hs)
        away_qf = _starter_quality_factor(as_)

        # Apply market blending to win probability
        model_wp = results["home_win_pct"]
        blended_wp = model_wp
        market_weight_used = 0.0
        if game_market and game_market.get("home_win_prob"):
            from betting.confidence import compute_confidence
            confidence = 0.5  # default when we don't have dual engines
            adj_wt = config.TRAINED_MARKET_WEIGHT * (1.0 - config.MARKET_CONFIDENCE_DISCOUNT * confidence)
            blended_wp = (1 - adj_wt) * model_wp + adj_wt * game_market["home_win_prob"]
            blended_wp = max(0.02, min(0.98, blended_wp))
            market_weight_used = adj_wt

        pred = {
            "game_id": str(g["game_id"]),
            "away": away,
            "home": home,
            "venue": g.get("venue_name", ""),
            "away_sp": as_name or "TBD",
            "home_sp": hs_name or "TBD",
            "away_sp_qf": away_qf,
            "home_sp_qf": home_qf,
            "model_home_wp": results["home_win_pct"],
            "home_wp": blended_wp,
            "away_wp": 1 - blended_wp,
            "market_home_wp": game_market.get("home_win_prob") if game_market else None,
            "market_weight": market_weight_used,
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

    filtered_extreme = 0
    filtered_small_edge = 0

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

                # IMPORTANT: filter on the RAW-model edge, not the post-blend edge.
                # `pred["home_wp"]` is the ensemble *after* blending with kalshi at
                # weight 0.65, so |home_wp − kalshi| ≥ 0.04 algebraically requires
                # |raw_model − kalshi| ≥ 0.04 / (1 − 0.65) = 0.114. Filtering on the
                # blended price under-counts that — it makes the threshold appear to
                # be 4% but the real model disagreement is 11%+. We now use the raw
                # model probability for the filter so the threshold has its
                # intended meaning.
                raw_model_home = pred.get("model_home_wp", pred["home_wp"])
                raw_model_prob = raw_model_home if is_home_side else (1 - raw_model_home)
                blended_prob = pred["home_wp"] if is_home_side else pred["away_wp"]
                kalshi_price = mid

                raw_edge = raw_model_prob - kalshi_price

                # Skip extreme markets — don't bet against strong market conviction
                if kalshi_price < config.MIN_MARKET_PRICE or kalshi_price > config.MAX_MARKET_PRICE:
                    if raw_edge > 0:
                        filtered_extreme += 1
                    continue
                if raw_edge > 0:
                    if raw_edge < config.MIN_EDGE_THRESHOLD:
                        filtered_small_edge += 1
                        continue
                    # EV uses the BLENDED probability (best estimate of true win prob
                    # given both model + market). The filter uses the raw edge so we
                    # only bet when the model is genuinely confident enough to
                    # override the market prior, not just nudge it.
                    ev = blended_prob * (1.0 / kalshi_price - 1) - (1 - blended_prob)
                    direction = "HOME ML" if (is_home_side and raw_edge > 0) or (not is_home_side and raw_edge < 0) else "AWAY ML"
                    edges.append({
                        "game": game_label,
                        "type": direction,
                        "model": blended_prob,        # what we use for sizing
                        "raw_model": raw_model_prob,  # what we filtered on
                        "kalshi": kalshi_price,
                        "edge": raw_edge,
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

                        # Skip extreme markets on totals too
                        if over_mid < config.MIN_MARKET_PRICE or over_mid > config.MAX_MARKET_PRICE:
                            filtered_extreme += 1
                            continue

                        if under_edge > config.MIN_EDGE_THRESHOLD:
                            ev = model_under * (1.0 / kalshi_under - 1) - (1 - model_under)
                            edges.append({
                                "game": game_label,
                                "type": f"UNDER {line}",
                                "model": model_under,
                                "kalshi": kalshi_under,
                                "edge": under_edge,
                                "ev": ev,
                            })
                        elif over_edge > config.MIN_EDGE_THRESHOLD:
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
        model_wp = pred.get("model_home_wp", pred["home_wp"])
        mkt_wp = pred.get("market_home_wp")
        blended = pred["home_wp"]
        mkt_str = f"Mkt:{mkt_wp:.0%}" if mkt_wp else "Mkt:—"
        print(f"  {pred['away'][:12]+' @ '+pred['home'][:12]:<30} "
              f"Mdl:{model_wp:.0%} {mkt_str} →{blended:.0%}  "
              f"Tot:{pred['total_mean']:.1f}  "
              f"SP:{pred['away_sp'][:10]}({pred['away_sp_qf']:.2f})v{pred['home_sp'][:10]}({pred['home_sp_qf']:.2f})")

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

    # Show filter counts
    if filtered_extreme or filtered_small_edge:
        print()
        print(f"  FILTERS APPLIED:")
        if filtered_extreme:
            print(f"    Skipped {filtered_extreme} bets vs extreme markets (< {config.MIN_MARKET_PRICE:.0%} or > {config.MAX_MARKET_PRICE:.0%})")
        if filtered_small_edge:
            print(f"    Skipped {filtered_small_edge} bets with edge < {config.MIN_EDGE_THRESHOLD:.0%} (noise)")

    print()
    print("=" * 85)

    return edges


def _merge_pipeline_output(target_date: str, predictions: list, edges: list, kalshi_count: int):
    """Merge new predictions/edges into today's pipeline file, deduping by game_id.

    This lets per-game runs accumulate results across multiple invocations.
    """
    out_path = Path(".context") / f"pipeline_{target_date.replace('-', '_')}.json"
    existing = {"date": target_date, "predictions": [], "edges": [], "kalshi_games": 0}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    # Dedup predictions by game_id (new entries replace old)
    new_ids = {p["game_id"] for p in predictions}
    merged_preds = [p for p in existing.get("predictions", []) if p["game_id"] not in new_ids]
    merged_preds.extend(predictions)

    # Dedup edges by (game, type)
    new_edge_keys = {(e["game"], e["type"]) for e in edges}
    merged_edges = [e for e in existing.get("edges", []) if (e["game"], e["type"]) not in new_edge_keys]
    merged_edges.extend(edges)

    output = {
        "date": target_date,
        "predictions": merged_preds,
        "edges": merged_edges,
        "kalshi_games": max(kalshi_count, existing.get("kalshi_games", 0)),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path} ({len(merged_preds)} predictions, {len(merged_edges)} edges)")


def cmd_full(target_date: str):
    """Full daily pipeline: fetch yesterday, all games, settle, log.

    Used for manual runs or legacy full-pipeline cron.
    """
    print(f"MLB Daily Pipeline — {target_date} (FULL)")
    print("=" * 50)

    print("\n[0/5] Loading data...")
    current_games, prior_games, pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors = load_all_data()

    current_games = step1_fetch_yesterday(target_date, current_games)
    pitcher_idx = _build_pitcher_index(current_games)

    kalshi_markets = step2_fetch_kalshi(target_date)
    _cache_kalshi_prices(kalshi_markets, target_date)

    lineups = step3_fetch_lineups(target_date)

    predictions = step4_run_simulations(
        target_date, current_games, pitcher_idx, prior_pitcher_idx,
        starter_lookup, park_factors, prior_games,
        kalshi_markets=kalshi_markets,
    )

    edges = step5_compare_and_report(predictions, kalshi_markets, target_date)
    _merge_pipeline_output(target_date, predictions, edges, len(kalshi_markets.get("today", {})))

    print(f"\nLINEUP STATUS: {len(lineups)} games with confirmed lineups")


def cmd_single_game(game_id: str, target_date: str):
    """Per-game pipeline: fetch just this game's Kalshi + simulate + log edge.

    Skips yesterday fetch and settle (those happen in the morning job).
    """
    print(f"MLB Per-Game Pipeline — game {game_id} on {target_date}")
    print("=" * 50)

    print("\n[0/3] Loading data...")
    current_games, prior_games, pitcher_idx, prior_pitcher_idx, starter_lookup, park_factors = load_all_data()

    # Fetch Kalshi markets (filtered by parse logic to just matching games)
    kalshi_markets = step2_fetch_kalshi(target_date)
    _cache_kalshi_prices(kalshi_markets, target_date)

    # Run simulation for just this game
    predictions = step4_run_simulations(
        target_date, current_games, pitcher_idx, prior_pitcher_idx,
        starter_lookup, park_factors, prior_games,
        kalshi_markets=kalshi_markets,
        only_game_id=game_id,
    )

    if not predictions:
        print(f"\nNo prediction produced for game {game_id} (possibly not in today's schedule)")
        return

    # Compare + log edges for this game only
    edges = step5_compare_and_report(predictions, kalshi_markets, target_date)

    # Merge into today's pipeline file
    _merge_pipeline_output(target_date, predictions, edges, len(kalshi_markets.get("today", {})))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MLB daily prediction pipeline")
    parser.add_argument("date", nargs="?", default=None, help="Target date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--game-id", dest="game_id", default=None,
                        help="Run pipeline for a single game only (skips yesterday fetch + settle)")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

    if args.game_id:
        cmd_single_game(args.game_id, target_date)
    else:
        cmd_full(target_date)


if __name__ == "__main__":
    main()
