"""Database setup and schema for MLB simulator."""

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, Float, String, DateTime, Date, Boolean, Text,
    UniqueConstraint, Index,
)
from sqlalchemy.sql import func
import config

engine = create_engine(config.DATABASE_URL)
metadata = MetaData()

# Games table
games = Table(
    "games", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("game_id", String(20), unique=True, nullable=False),  # MLB game_pk
    Column("game_date", Date, nullable=False),
    Column("home_team_id", Integer, nullable=False),
    Column("away_team_id", Integer, nullable=False),
    Column("home_team_abbr", String(5)),
    Column("away_team_abbr", String(5)),
    Column("home_score", Integer),
    Column("away_score", Integer),
    Column("venue", String(100)),
    Column("season", Integer),
    Column("innings_played", Integer, default=9),
    Column("fetched_at", DateTime, server_default=func.now()),
    Index("idx_games_date", "game_date"),
    Index("idx_games_home_team", "home_team_id"),
    Index("idx_games_away_team", "away_team_id"),
)

# Player batting stats (per-game box score lines)
player_batting_stats = Table(
    "player_batting_stats", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("game_id", String(20), nullable=False),
    Column("game_date", Date, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("player_name", String(100)),
    Column("team_id", Integer),
    Column("team_abbr", String(5)),
    Column("bats", String(1)),  # L, R, S (switch)
    Column("batting_order", Integer),  # 1-9
    Column("pa", Integer),      # plate appearances
    Column("ab", Integer),      # at-bats
    Column("hits", Integer),
    Column("doubles", Integer),
    Column("triples", Integer),
    Column("hr", Integer),
    Column("rbi", Integer),
    Column("bb", Integer),
    Column("hbp", Integer),
    Column("k", Integer),       # strikeouts
    Column("sb", Integer),      # stolen bases
    Column("cs", Integer),      # caught stealing
    Column("sf", Integer),      # sacrifice flies
    Column("sh", Integer),      # sacrifice bunts
    Column("gidp", Integer),    # grounded into double play
    UniqueConstraint("game_id", "player_id", name="uq_batting_game_player"),
    Index("idx_batting_player", "player_id"),
    Index("idx_batting_game_date", "game_date"),
    Index("idx_batting_team", "team_id"),
)

# Player pitching stats (per-game)
player_pitching_stats = Table(
    "player_pitching_stats", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("game_id", String(20), nullable=False),
    Column("game_date", Date, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("player_name", String(100)),
    Column("team_id", Integer),
    Column("team_abbr", String(5)),
    Column("throws", String(1)),  # L, R
    Column("is_starter", Boolean),
    Column("innings_pitched", Float),  # 6.1 = 6 and 1/3 innings
    Column("hits_allowed", Integer),
    Column("runs", Integer),
    Column("earned_runs", Integer),
    Column("bb", Integer),
    Column("k", Integer),
    Column("hr_allowed", Integer),
    Column("pitches_thrown", Integer),
    Column("batters_faced", Integer),
    Column("gb", Integer),       # ground balls
    Column("fb", Integer),       # fly balls
    Column("ld", Integer),       # line drives
    Column("wp", Integer),       # wild pitches
    Column("hbp", Integer),      # hit batters
    UniqueConstraint("game_id", "player_id", name="uq_pitching_game_player"),
    Index("idx_pitching_player", "player_id"),
    Index("idx_pitching_game_date", "game_date"),
    Index("idx_pitching_team", "team_id"),
)

# Plate appearance events (for detailed model training/validation)
plate_appearance_events = Table(
    "plate_appearance_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("game_id", String(20), nullable=False),
    Column("inning", Integer),
    Column("half", String(6)),  # "top" or "bottom"
    Column("batter_id", Integer),
    Column("batter_name", String(100)),
    Column("batter_hand", String(1)),  # L, R
    Column("pitcher_id", Integer),
    Column("pitcher_name", String(100)),
    Column("pitcher_hand", String(1)),  # L, R
    Column("pa_result", String(20)),  # K, BB, HBP, 1B, 2B, 3B, HR, GO, FO, LO, PO, FC, SAC, etc.
    Column("batted_ball_type", String(5)),  # GB, FB, LD, PU (popup)
    Column("pitch_count_balls", Integer),
    Column("pitch_count_strikes", Integer),
    Column("total_pitches", Integer),
    Column("runners_before", String(10)),  # encoded: "---", "1--", "-2-", "12-", etc.
    Column("outs_before", Integer),
    Column("runs_on_play", Integer, default=0),
    Index("idx_pa_game", "game_id"),
    Index("idx_pa_batter", "batter_id"),
    Index("idx_pa_pitcher", "pitcher_id"),
)

# Team season stats
team_stats = Table(
    "team_stats", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("team_id", Integer, nullable=False),
    Column("team_abbr", String(5)),
    Column("team_name", String(50)),
    Column("season", Integer),
    Column("wins", Integer),
    Column("losses", Integer),
    Column("runs_per_game", Float),
    Column("runs_allowed_per_game", Float),
    Column("wrc_plus", Float),         # weighted runs created plus (100 = league avg)
    Column("fip_minus", Float),        # FIP minus (100 = league avg)
    Column("team_ops", Float),
    Column("team_era", Float),
    Column("team_fip", Float),
    Column("team_k_pct", Float),
    Column("team_bb_pct", Float),
    Column("team_hr_per_fb", Float),
    Column("team_babip", Float),
    Column("fetched_at", DateTime, server_default=func.now()),
    UniqueConstraint("team_id", "season", name="uq_mlb_team_season"),
)

# Park factors
park_factors = Table(
    "park_factors", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("venue", String(100), nullable=False),
    Column("team_abbr", String(5)),
    Column("season", Integer),
    Column("overall_factor", Float, default=1.0),  # 1.0 = neutral
    Column("hr_factor", Float, default=1.0),
    Column("h_factor", Float, default=1.0),
    Column("bb_factor", Float, default=1.0),
    UniqueConstraint("venue", "season", name="uq_park_season"),
)

# Game odds (moneyline, run line, totals)
game_odds = Table(
    "game_odds", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("game_id", String(20), nullable=False),
    Column("source", String(50), default="consensus"),
    Column("home_ml", Integer),         # American odds, e.g., -150
    Column("away_ml", Integer),         # American odds, e.g., +130
    Column("run_line", Float),          # e.g., -1.5 (home perspective)
    Column("run_line_home_odds", Integer, default=-110),
    Column("total_line", Float),        # e.g., 8.5
    Column("over_odds", Integer, default=-110),
    UniqueConstraint("game_id", "source", name="uq_mlb_game_source_odds"),
)


def init_db():
    """Create all tables."""
    metadata.create_all(engine)
    print("MLB database tables created successfully.")


if __name__ == "__main__":
    init_db()
