"""Fetch historical MLB closing odds from ESPN's public API.

Adapted from the NBA sim's ESPN odds fetcher. Gets moneyline,
run line (spread), and totals for every completed game.

Usage:
    python scripts/fetch_espn_odds.py                    # 2026 season to date
    python scripts/fetch_espn_odds.py 2025                # full 2025 season
    python scripts/fetch_espn_odds.py 2026 20260401 20260414  # specific range
"""

import json
import ssl
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

SCOREBOARD_URL = "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY_URL = "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"


def _get(url):
    """Fetch JSON from URL."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


def fetch_game_ids_for_date(dt: str) -> list[dict]:
    """Fetch game IDs and basic info from ESPN scoreboard for a date.

    Args:
        dt: date string in YYYYMMDD format

    Returns list of {event_id, home, away, home_score, away_score, status}
    """
    data = _get(f"{SCOREBOARD_URL}?dates={dt}")
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        home = away = None
        for c in competitors:
            team = c.get("team", {})
            info = {
                "abbr": team.get("abbreviation", ""),
                "name": team.get("displayName", ""),
                "score": int(c.get("score", 0)),
            }
            if c.get("homeAway") == "home":
                home = info
            else:
                away = info

        if home and away:
            games.append({
                "event_id": event.get("id", ""),
                "home": home["abbr"],
                "away": away["abbr"],
                "home_name": home["name"],
                "away_name": away["name"],
                "home_score": home["score"],
                "away_score": away["score"],
                "date": dt[:4] + "-" + dt[4:6] + "-" + dt[6:8],
            })

    return games


def fetch_odds_for_event(event_id: str) -> dict | None:
    """Fetch odds data from ESPN summary endpoint.

    Returns {
        "spread_open": float, "spread_close": float,
        "total_open": float, "total_close": float,
        "ml_home": int, "ml_away": int,
        "provider": str
    } or None if no odds available.
    """
    data = _get(f"{SUMMARY_URL}?event={event_id}")
    if not data:
        return None

    pickcenter = data.get("pickcenter", [])
    if not pickcenter:
        return None

    # Use the first provider (usually DraftKings or consensus)
    pick = pickcenter[0]
    provider = pick.get("provider", {}).get("name", "unknown")

    odds = {
        "provider": provider,
        "spread_open": None,
        "spread_close": None,
        "total_open": None,
        "total_close": None,
        "ml_home": None,
        "ml_away": None,
    }

    # Parse spread
    spread = pick.get("spread")
    if spread is not None:
        odds["spread_close"] = float(spread)

    # Parse total (overUnder)
    total = pick.get("overUnder")
    if total is not None:
        odds["total_close"] = float(total)

    # Parse moneyline from homeTeamOdds/awayTeamOdds
    home_odds = pick.get("homeTeamOdds", {})
    away_odds = pick.get("awayTeamOdds", {})

    if home_odds.get("moneyLine"):
        odds["ml_home"] = int(home_odds["moneyLine"])
    if away_odds.get("moneyLine"):
        odds["ml_away"] = int(away_odds["moneyLine"])

    # Check for multiple providers
    for pick in pickcenter[1:]:
        p_name = pick.get("provider", {}).get("name", "")
        if "consensus" in p_name.lower() or "dk" in p_name.lower():
            s = pick.get("spread")
            if s is not None:
                odds["spread_close"] = float(s)
            t = pick.get("overUnder")
            if t is not None:
                odds["total_close"] = float(t)
            ho = pick.get("homeTeamOdds", {})
            ao = pick.get("awayTeamOdds", {})
            if ho.get("moneyLine"):
                odds["ml_home"] = int(ho["moneyLine"])
            if ao.get("moneyLine"):
                odds["ml_away"] = int(ao["moneyLine"])

    return odds


def fetch_season_odds(season: int, start_dt: str = None, end_dt: str = None) -> list[dict]:
    """Fetch odds for all games in a season or date range."""
    if start_dt is None:
        start_dt = f"{season}0320"  # Opening Day window
    if end_dt is None:
        today = date.today()
        end = min(date(season, 11, 5), today - timedelta(days=1))
        end_dt = end.strftime("%Y%m%d")

    all_games = []
    current = date(int(start_dt[:4]), int(start_dt[4:6]), int(start_dt[6:8]))
    end_date = date(int(end_dt[:4]), int(end_dt[4:6]), int(end_dt[6:8]))

    while current <= end_date:
        dt_str = current.strftime("%Y%m%d")
        games = fetch_game_ids_for_date(dt_str)

        if games:
            print(f"  {current}: {len(games)} games", end="", flush=True)
            for g in games:
                odds = fetch_odds_for_event(g["event_id"])
                if odds:
                    g["odds"] = odds
                    all_games.append(g)
                time.sleep(0.5)  # rate limit
            odds_count = sum(1 for g in games if "odds" in g)
            print(f" ({odds_count} with odds)")
        else:
            pass  # no games this day

        current += timedelta(days=1)

    return all_games


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    start_dt = sys.argv[2] if len(sys.argv) > 2 else None
    end_dt = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Fetching ESPN MLB odds for {season} season...")
    games = fetch_season_odds(season, start_dt, end_dt)

    # Save
    out_path = Path(f".context/espn_odds_{season}.json")
    with open(out_path, "w") as f:
        json.dump(games, f, indent=2)

    # Summary
    games_with_odds = [g for g in games if g.get("odds")]
    games_with_ml = [g for g in games_with_odds if g["odds"].get("ml_home")]
    games_with_total = [g for g in games_with_odds if g["odds"].get("total_close")]
    games_with_spread = [g for g in games_with_odds if g["odds"].get("spread_close")]

    print(f"\n{'='*60}")
    print(f"ESPN ODDS — {season} MLB Season")
    print(f"{'='*60}")
    print(f"  Total games:     {len(games)}")
    print(f"  With odds:       {len(games_with_odds)}")
    print(f"  With moneyline:  {len(games_with_ml)}")
    print(f"  With total:      {len(games_with_total)}")
    print(f"  With spread:     {len(games_with_spread)}")

    if games:
        dates = sorted(set(g["date"] for g in games))
        print(f"  Date range:      {dates[0]} to {dates[-1]}")

    print(f"\n  Saved to {out_path}")

    # Show sample
    if games_with_odds:
        print(f"\n  Sample game:")
        g = games_with_odds[-1]
        o = g["odds"]
        print(f"    {g['away']} @ {g['home']} ({g['date']})")
        print(f"    Score: {g['away_score']}-{g['home_score']}")
        print(f"    ML: {o.get('ml_home')}/{o.get('ml_away')}  Spread: {o.get('spread_close')}  Total: {o.get('total_close')}")
        print(f"    Provider: {o.get('provider')}")


if __name__ == "__main__":
    main()
