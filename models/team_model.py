"""Team-level statistical model.

Aggregates team offensive and defensive performance for the box score model.
"""

from dataclasses import dataclass
import numpy as np
from sqlalchemy import select, and_
from data.db import engine, games, team_stats, park_factors as park_factors_table
import config


@dataclass
class TeamModel:
    team_id: int
    team_abbr: str
    season: int
    wins: int
    losses: int
    runs_per_game: float
    runs_allowed_per_game: float
    wrc_plus: float       # 100 = league average
    fip_minus: float      # 100 = league average
    team_k_pct: float
    team_bb_pct: float
    team_hr_per_fb: float
    team_babip: float
    win_pct: float


@dataclass
class ParkFactor:
    venue: str
    overall_factor: float
    hr_factor: float
    h_factor: float
    bb_factor: float


def build_team_model_from_games(
    team_id: int,
    team_abbr: str,
    as_of_date,
    season: int | None = None,
) -> TeamModel:
    """Build a team model by aggregating game results before as_of_date.

    Uses exponential decay weighting for recent performance.
    """
    if season is None:
        season = config.CURRENT_SEASON

    with engine.connect() as conn:
        # Get games as home team
        home_stmt = (
            select(games)
            .where(
                and_(
                    games.c.home_team_id == team_id,
                    games.c.game_date < as_of_date,
                    games.c.season == season,
                )
            )
            .order_by(games.c.game_date.desc())
        )
        home_rows = conn.execute(home_stmt).fetchall()

        # Get games as away team
        away_stmt = (
            select(games)
            .where(
                and_(
                    games.c.away_team_id == team_id,
                    games.c.game_date < as_of_date,
                    games.c.season == season,
                )
            )
            .order_by(games.c.game_date.desc())
        )
        away_rows = conn.execute(away_stmt).fetchall()

    # Combine and sort by date descending
    all_games = []
    for r in home_rows:
        all_games.append({
            "date": r.game_date,
            "runs_for": r.home_score or 0,
            "runs_against": r.away_score or 0,
            "won": (r.home_score or 0) > (r.away_score or 0),
        })
    for r in away_rows:
        all_games.append({
            "date": r.game_date,
            "runs_for": r.away_score or 0,
            "runs_against": r.home_score or 0,
            "won": (r.away_score or 0) > (r.home_score or 0),
        })

    all_games.sort(key=lambda g: g["date"], reverse=True)

    if not all_games:
        return _league_average_team(team_id, team_abbr, season)

    n = len(all_games)
    weights = np.exp(-config.TEAM_DECAY_LAMBDA * np.arange(n))
    w_sum = weights.sum()

    runs_for = np.array([g["runs_for"] for g in all_games], dtype=float)
    runs_against = np.array([g["runs_against"] for g in all_games], dtype=float)
    won = np.array([g["won"] for g in all_games], dtype=float)

    wins = int(won.sum())
    losses = n - wins
    win_pct = wins / n if n > 0 else 0.5

    rpg = np.dot(weights, runs_for) / w_sum
    rapg = np.dot(weights, runs_against) / w_sum

    # wRC+ and FIP- approximations from run scoring
    wrc_plus = (rpg / config.LEAGUE_AVG_RUNS_PER_GAME) * 100
    fip_minus = (rapg / config.LEAGUE_AVG_RUNS_PER_GAME) * 100

    return TeamModel(
        team_id=team_id,
        team_abbr=team_abbr,
        season=season,
        wins=wins,
        losses=losses,
        runs_per_game=rpg,
        runs_allowed_per_game=rapg,
        wrc_plus=wrc_plus,
        fip_minus=fip_minus,
        team_k_pct=config.LEAGUE_AVG_K_PCT,
        team_bb_pct=config.LEAGUE_AVG_BB_PCT,
        team_hr_per_fb=config.LEAGUE_AVG_HR_PER_FB,
        team_babip=config.LEAGUE_AVG_BABIP,
        win_pct=win_pct,
    )


def load_park_factor(venue: str, season: int | None = None) -> ParkFactor:
    """Load park factor for a venue. Returns neutral if not found."""
    if season is None:
        season = config.CURRENT_SEASON

    with engine.connect() as conn:
        stmt = (
            select(park_factors_table)
            .where(
                and_(
                    park_factors_table.c.venue == venue,
                    park_factors_table.c.season == season,
                )
            )
        )
        row = conn.execute(stmt).fetchone()

    if row:
        return ParkFactor(
            venue=venue,
            overall_factor=row.overall_factor or 1.0,
            hr_factor=row.hr_factor or 1.0,
            h_factor=row.h_factor or 1.0,
            bb_factor=row.bb_factor or 1.0,
        )

    return ParkFactor(
        venue=venue,
        overall_factor=config.DEFAULT_PARK_FACTOR,
        hr_factor=config.DEFAULT_PARK_FACTOR,
        h_factor=config.DEFAULT_PARK_FACTOR,
        bb_factor=config.DEFAULT_PARK_FACTOR,
    )


def _league_average_team(team_id: int, team_abbr: str, season: int) -> TeamModel:
    """Return a league-average team model as fallback."""
    return TeamModel(
        team_id=team_id,
        team_abbr=team_abbr,
        season=season,
        wins=81,
        losses=81,
        runs_per_game=config.LEAGUE_AVG_RUNS_PER_GAME,
        runs_allowed_per_game=config.LEAGUE_AVG_RUNS_PER_GAME,
        wrc_plus=100.0,
        fip_minus=100.0,
        team_k_pct=config.LEAGUE_AVG_K_PCT,
        team_bb_pct=config.LEAGUE_AVG_BB_PCT,
        team_hr_per_fb=config.LEAGUE_AVG_HR_PER_FB,
        team_babip=config.LEAGUE_AVG_BABIP,
        win_pct=0.500,
    )
