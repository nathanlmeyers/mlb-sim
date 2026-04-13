"""Data fetching from MLB data sources.

Uses pybaseball for Statcast/FanGraphs data and MLB-StatsAPI for
game schedules, box scores, and lineup data.
"""

import time
from datetime import date, timedelta
import pandas as pd
import statsapi
import config


def fetch_schedule(start_date: date, end_date: date) -> list[dict]:
    """Fetch MLB game schedule for a date range.

    Returns list of game dicts with game_id, date, teams, scores, venue.
    """
    games = []
    sched = statsapi.schedule(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    for g in sched:
        if g.get("status") != "Final":
            continue
        games.append({
            "game_id": str(g["game_id"]),
            "game_date": date.fromisoformat(g["game_date"]),
            "home_team_id": g.get("home_id"),
            "away_team_id": g.get("away_id"),
            "home_team_abbr": g.get("home_name", "")[:5],
            "away_team_abbr": g.get("away_name", "")[:5],
            "home_score": g.get("home_score"),
            "away_score": g.get("away_score"),
            "venue": g.get("venue_name", ""),
            "season": start_date.year,
        })
        time.sleep(config.API_DELAY_SECONDS * 0.1)

    return games


def fetch_box_score(game_id: str) -> dict:
    """Fetch detailed box score for a game.

    Returns dict with 'batting' and 'pitching' lists.
    """
    try:
        box = statsapi.boxscore_data(game_id)
    except Exception as e:
        print(f"Error fetching box score for game {game_id}: {e}")
        return {"batting": [], "pitching": []}

    batting = []
    pitching = []

    for side in ["home", "away"]:
        team_info = box.get(side, {})
        team_id = team_info.get("teamStats", {}).get("batting", {}).get("teamId")
        team_abbr = team_info.get("team", {}).get("abbreviation", "")

        # Batting stats
        for player_id, player_data in team_info.get("players", {}).items():
            if not player_data.get("battingOrder"):
                continue
            stats = player_data.get("stats", {}).get("batting", {})
            if not stats:
                continue
            batting.append({
                "game_id": game_id,
                "player_id": int(player_id.replace("ID", "")),
                "player_name": player_data.get("person", {}).get("fullName", ""),
                "team_id": team_id,
                "team_abbr": team_abbr,
                "batting_order": int(player_data.get("battingOrder", "0")[:1]),
                "pa": stats.get("plateAppearances", 0),
                "ab": stats.get("atBats", 0),
                "hits": stats.get("hits", 0),
                "doubles": stats.get("doubles", 0),
                "triples": stats.get("triples", 0),
                "hr": stats.get("homeRuns", 0),
                "rbi": stats.get("rbi", 0),
                "bb": stats.get("baseOnBalls", 0),
                "hbp": stats.get("hitByPitch", 0),
                "k": stats.get("strikeOuts", 0),
                "sb": stats.get("stolenBases", 0),
                "cs": stats.get("caughtStealing", 0),
            })

        # Pitching stats
        for player_id, player_data in team_info.get("players", {}).items():
            stats = player_data.get("stats", {}).get("pitching", {})
            if not stats or stats.get("inningsPitched") is None:
                continue
            ip_str = stats.get("inningsPitched", "0")
            pitching.append({
                "game_id": game_id,
                "player_id": int(player_id.replace("ID", "")),
                "player_name": player_data.get("person", {}).get("fullName", ""),
                "team_id": team_id,
                "team_abbr": team_abbr,
                "is_starter": player_data.get("gameStatus", {}).get("isCurrentPitcher") is not None,
                "innings_pitched": float(ip_str) if ip_str else 0,
                "hits_allowed": stats.get("hits", 0),
                "runs": stats.get("runs", 0),
                "earned_runs": stats.get("earnedRuns", 0),
                "bb": stats.get("baseOnBalls", 0),
                "k": stats.get("strikeOuts", 0),
                "hr_allowed": stats.get("homeRuns", 0),
                "pitches_thrown": stats.get("numberOfPitches", 0),
                "batters_faced": stats.get("battersFaced", 0),
                "hbp": stats.get("hitBatsmen", 0),
            })

    time.sleep(config.API_DELAY_SECONDS)
    return {"batting": batting, "pitching": pitching}


def fetch_season_games(season: int) -> list[dict]:
    """Fetch all completed games for an entire season."""
    start = date(season, 3, 20)  # spring training / opening day window
    end = date(season, 11, 5)    # through World Series
    today = date.today()
    if end > today:
        end = today

    all_games = []
    current = start
    # Fetch in 30-day chunks to avoid API limits
    while current < end:
        chunk_end = min(current + timedelta(days=30), end)
        print(f"Fetching games {current} to {chunk_end}...")
        chunk = fetch_schedule(current, chunk_end)
        all_games.extend(chunk)
        current = chunk_end + timedelta(days=1)
        time.sleep(config.API_DELAY_SECONDS)

    print(f"Fetched {len(all_games)} games for {season} season")
    return all_games
