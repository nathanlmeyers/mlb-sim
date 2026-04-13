"""Load fetched data into PostgreSQL database."""

from datetime import date
from sqlalchemy import insert, select
from data.db import (
    engine, metadata, games, player_batting_stats,
    player_pitching_stats, park_factors,
)
from data.fetch import fetch_schedule, fetch_box_score, fetch_season_games
import config


def load_games(game_list: list[dict]) -> int:
    """Insert games into the database. Skips duplicates."""
    loaded = 0
    with engine.begin() as conn:
        for g in game_list:
            # Check if already exists
            exists = conn.execute(
                select(games.c.game_id).where(games.c.game_id == g["game_id"])
            ).fetchone()
            if exists:
                continue

            conn.execute(insert(games).values(**g))
            loaded += 1

    print(f"Loaded {loaded} new games ({len(game_list) - loaded} already existed)")
    return loaded


def load_batting_stats(batting_list: list[dict], game_date: date) -> int:
    """Insert batting stats for a game."""
    loaded = 0
    with engine.begin() as conn:
        for b in batting_list:
            b["game_date"] = game_date
            try:
                conn.execute(insert(player_batting_stats).values(**b))
                loaded += 1
            except Exception:
                pass  # duplicate
    return loaded


def load_pitching_stats(pitching_list: list[dict], game_date: date) -> int:
    """Insert pitching stats for a game."""
    loaded = 0
    with engine.begin() as conn:
        for p in pitching_list:
            p["game_date"] = game_date
            try:
                conn.execute(insert(player_pitching_stats).values(**p))
                loaded += 1
            except Exception:
                pass  # duplicate
    return loaded


def load_season(season: int) -> None:
    """Fetch and load an entire season of data."""
    print(f"=== Loading {season} MLB season ===")

    # 1. Fetch and load game schedule
    game_list = fetch_season_games(season)
    load_games(game_list)

    # 2. Fetch and load box scores for each game
    for i, g in enumerate(game_list):
        if (i + 1) % 50 == 0:
            print(f"  Loading box scores: {i + 1}/{len(game_list)}")

        box = fetch_box_score(g["game_id"])
        load_batting_stats(box["batting"], g["game_date"])
        load_pitching_stats(box["pitching"], g["game_date"])

    print(f"=== Finished loading {season} season ===")


def load_default_park_factors() -> None:
    """Load default park factors for all MLB venues."""
    # 2024 park factors (source: FanGraphs)
    factors = [
        ("Coors Field", "COL", 1.38, 1.40, 1.20, 1.05),
        ("Fenway Park", "BOS", 1.10, 1.05, 1.15, 1.00),
        ("Great American Ball Park", "CIN", 1.12, 1.25, 1.05, 1.00),
        ("Globe Life Field", "TEX", 1.05, 1.10, 1.02, 0.98),
        ("Yankee Stadium", "NYY", 1.05, 1.15, 0.98, 1.00),
        ("Citizens Bank Park", "PHI", 1.04, 1.08, 1.02, 1.00),
        ("Wrigley Field", "CHC", 1.03, 1.05, 1.02, 1.00),
        ("Guaranteed Rate Field", "CWS", 1.02, 1.05, 1.00, 1.00),
        ("Dodger Stadium", "LAD", 0.98, 0.95, 1.00, 1.00),
        ("Target Field", "MIN", 0.98, 0.95, 1.00, 1.02),
        ("Tropicana Field", "TB", 0.96, 0.90, 1.00, 1.00),
        ("T-Mobile Park", "SEA", 0.95, 0.88, 0.98, 1.00),
        ("Petco Park", "SD", 0.93, 0.85, 0.98, 1.00),
        ("Oracle Park", "SF", 0.90, 0.82, 0.95, 1.02),
        ("Oakland Coliseum", "OAK", 0.92, 0.85, 0.95, 1.00),
        ("Kauffman Stadium", "KC", 0.97, 0.92, 1.00, 1.00),
        ("Busch Stadium", "STL", 0.98, 0.95, 1.00, 1.00),
        ("Minute Maid Park", "HOU", 1.02, 1.05, 1.00, 1.00),
        ("Truist Park", "ATL", 1.01, 1.03, 1.00, 1.00),
        ("loanDepot park", "MIA", 0.94, 0.88, 0.98, 1.00),
        ("Citi Field", "NYM", 0.96, 0.92, 0.98, 1.00),
        ("PNC Park", "PIT", 0.97, 0.93, 1.00, 1.00),
        ("Progressive Field", "CLE", 0.98, 0.95, 1.00, 1.00),
        ("Camden Yards", "BAL", 1.02, 1.05, 1.00, 1.00),
        ("Nationals Park", "WSH", 1.00, 1.00, 1.00, 1.00),
        ("Rogers Centre", "TOR", 1.01, 1.03, 1.00, 0.98),
        ("Angel Stadium", "LAA", 0.98, 0.95, 1.00, 1.00),
        ("Chase Field", "ARI", 1.06, 1.10, 1.03, 1.00),
        ("American Family Field", "MIL", 1.02, 1.05, 1.00, 1.00),
        ("Comerica Park", "DET", 0.97, 0.93, 1.00, 1.00),
    ]

    with engine.begin() as conn:
        for venue, abbr, overall, hr, h, bb in factors:
            try:
                conn.execute(insert(park_factors).values(
                    venue=venue,
                    team_abbr=abbr,
                    season=config.CURRENT_SEASON,
                    overall_factor=overall,
                    hr_factor=hr,
                    h_factor=h,
                    bb_factor=bb,
                ))
            except Exception:
                pass  # already exists

    print(f"Loaded {len(factors)} park factors")
