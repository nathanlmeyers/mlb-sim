"""Batter statistical model with exponential decay and Bayesian regression.

Each batter gets a model built from their recent game logs, with:
- Exponential decay weighting (recent games matter more)
- Bayesian regression to league means (handles small samples)
- Platoon splits (vs LHP / vs RHP)
"""

from dataclasses import dataclass
import numpy as np
from sqlalchemy import select, and_
from data.db import engine, player_batting_stats
import config


@dataclass
class BatterModel:
    player_id: int
    player_name: str
    team_id: int
    bats: str  # L, R, S (switch)

    # Core rate stats (per PA)
    k_pct: float
    bb_pct: float
    hbp_pct: float

    # Batted ball profile (conditional on ball in play)
    gb_pct: float
    fb_pct: float
    ld_pct: float

    # Hit quality
    hr_per_fb: float
    babip: float

    # Platoon splits (vs LHP)
    k_pct_vs_lhp: float
    bb_pct_vs_lhp: float
    hr_per_fb_vs_lhp: float
    babip_vs_lhp: float

    # Platoon splits (vs RHP)
    k_pct_vs_rhp: float
    bb_pct_vs_rhp: float
    hr_per_fb_vs_rhp: float
    babip_vs_rhp: float

    # Baserunning
    speed_score: float  # 0-10 scale

    # Sample sizes
    games_played: int
    pa_total: int


def _regress(observed: float, sample_size: float, league_mean: float, regression_n: float) -> float:
    """Bayesian regression toward league mean.

    Same formula as NBA engine: weighted average of observed and prior,
    where weight = sample_size / (sample_size + regression_n).
    """
    if sample_size + regression_n == 0:
        return league_mean
    weight = sample_size / (sample_size + regression_n)
    return weight * observed + (1 - weight) * league_mean


def _exponential_decay_weights(n_games: int, lam: float) -> np.ndarray:
    """Generate exponential decay weights for n games.

    Game 0 (most recent) gets weight 1.0.
    Game i gets weight exp(-lambda * i).
    """
    indices = np.arange(n_games)
    return np.exp(-lam * indices)


def build_batter_model(
    player_id: int,
    as_of_date,
    max_games: int = 60,
) -> BatterModel | None:
    """Build a batter model from game logs prior to as_of_date.

    Uses exponential decay weighting and Bayesian regression.
    Returns None if the player has no games in the window.
    """
    with engine.connect() as conn:
        stmt = (
            select(player_batting_stats)
            .where(
                and_(
                    player_batting_stats.c.player_id == player_id,
                    player_batting_stats.c.game_date < as_of_date,
                )
            )
            .order_by(player_batting_stats.c.game_date.desc())
            .limit(max_games)
        )
        rows = conn.execute(stmt).fetchall()

    if not rows:
        return None

    n = len(rows)
    weights = _exponential_decay_weights(n, config.BATTER_DECAY_LAMBDA)
    w_sum = weights.sum()

    # Extract arrays
    pa = np.array([r.pa or 0 for r in rows], dtype=float)
    ab = np.array([r.ab or 0 for r in rows], dtype=float)
    k = np.array([r.k or 0 for r in rows], dtype=float)
    bb = np.array([r.bb or 0 for r in rows], dtype=float)
    hbp = np.array([r.hbp or 0 for r in rows], dtype=float)
    hits = np.array([r.hits or 0 for r in rows], dtype=float)
    doubles = np.array([r.doubles or 0 for r in rows], dtype=float)
    triples = np.array([r.triples or 0 for r in rows], dtype=float)
    hr = np.array([r.hr or 0 for r in rows], dtype=float)
    sb = np.array([r.sb or 0 for r in rows], dtype=float)

    total_pa = pa.sum()
    if total_pa == 0:
        return None

    # Weighted rate stats
    weighted_pa = np.dot(weights, pa)
    raw_k_pct = np.dot(weights, k) / weighted_pa if weighted_pa > 0 else config.LEAGUE_AVG_K_PCT
    raw_bb_pct = np.dot(weights, bb) / weighted_pa if weighted_pa > 0 else config.LEAGUE_AVG_BB_PCT
    raw_hbp_pct = np.dot(weights, hbp) / weighted_pa if weighted_pa > 0 else config.LEAGUE_AVG_HBP_PCT

    # Batted ball profile: balls in play = AB - K - HR + SF (approx: AB - K)
    bip = ab - k
    weighted_bip = np.dot(weights, np.maximum(bip, 0))
    singles = hits - doubles - triples - hr

    # BABIP = (H - HR) / (AB - K - HR + SF) ≈ (H - HR) / (AB - K - HR)
    h_minus_hr = hits - hr
    bip_for_babip = ab - k - hr
    weighted_h_minus_hr = np.dot(weights, np.maximum(h_minus_hr, 0))
    weighted_bip_babip = np.dot(weights, np.maximum(bip_for_babip, 0))
    raw_babip = weighted_h_minus_hr / weighted_bip_babip if weighted_bip_babip > 0 else config.LEAGUE_AVG_BABIP

    # HR/FB rate: we don't have FB data in batting stats directly,
    # so estimate FB from (HR / estimated_fb_rate)
    # For now, use HR/AB as a proxy for power, regressed
    raw_hr_rate = np.dot(weights, hr) / weighted_pa if weighted_pa > 0 else config.LEAGUE_AVG_HR_PER_FB * config.LEAGUE_AVG_FB_PCT
    # Convert to HR/FB using league average FB%
    raw_hr_per_fb = raw_hr_rate / config.LEAGUE_AVG_FB_PCT if config.LEAGUE_AVG_FB_PCT > 0 else config.LEAGUE_AVG_HR_PER_FB

    # Speed score (simplified: SB rate as proxy)
    total_sb = sb.sum()
    total_games = n
    raw_speed = min(10.0, (total_sb / max(total_games, 1)) * 5.0)

    # Apply Bayesian regression
    k_pct = _regress(raw_k_pct, total_pa, config.LEAGUE_AVG_K_PCT, config.REGRESSION_SAMPLES["k_pct"])
    bb_pct = _regress(raw_bb_pct, total_pa, config.LEAGUE_AVG_BB_PCT, config.REGRESSION_SAMPLES["bb_pct"])
    hbp_pct = _regress(raw_hbp_pct, total_pa, config.LEAGUE_AVG_HBP_PCT, config.REGRESSION_SAMPLES["hbp_pct"])
    babip = _regress(raw_babip, weighted_bip_babip, config.LEAGUE_AVG_BABIP, config.REGRESSION_SAMPLES["babip"])
    hr_per_fb = _regress(raw_hr_per_fb, total_pa * config.LEAGUE_AVG_FB_PCT, config.LEAGUE_AVG_HR_PER_FB, config.REGRESSION_SAMPLES["hr_per_fb"])

    # Batted ball profile: default to league average (need pitch-level data for accurate split)
    gb_pct = config.LEAGUE_AVG_GB_PCT
    fb_pct = config.LEAGUE_AVG_FB_PCT
    ld_pct = config.LEAGUE_AVG_LD_PCT

    # Platoon splits: apply simple additive adjustment from overall
    # In v2, these would be built from actual split data
    k_pct_vs_lhp = k_pct + config.PLATOON_ADJ_BATTER_VS_OPPOSITE * 0.5
    k_pct_vs_rhp = k_pct + config.PLATOON_ADJ_BATTER_VS_SAME * 0.5
    bb_pct_vs_lhp = bb_pct + abs(config.PLATOON_ADJ_BATTER_VS_OPPOSITE) * 0.3
    bb_pct_vs_rhp = bb_pct - abs(config.PLATOON_ADJ_BATTER_VS_SAME) * 0.3
    hr_per_fb_vs_lhp = hr_per_fb * 1.05
    hr_per_fb_vs_rhp = hr_per_fb * 0.95
    babip_vs_lhp = babip + 0.010
    babip_vs_rhp = babip - 0.010

    bats = rows[0].bats or "R"

    return BatterModel(
        player_id=player_id,
        player_name=rows[0].player_name,
        team_id=rows[0].team_id,
        bats=bats,
        k_pct=k_pct,
        bb_pct=bb_pct,
        hbp_pct=hbp_pct,
        gb_pct=gb_pct,
        fb_pct=fb_pct,
        ld_pct=ld_pct,
        hr_per_fb=hr_per_fb,
        babip=babip,
        k_pct_vs_lhp=k_pct_vs_lhp,
        bb_pct_vs_lhp=bb_pct_vs_lhp,
        hr_per_fb_vs_lhp=hr_per_fb_vs_lhp,
        babip_vs_lhp=babip_vs_lhp,
        k_pct_vs_rhp=k_pct_vs_rhp,
        bb_pct_vs_rhp=bb_pct_vs_rhp,
        hr_per_fb_vs_rhp=hr_per_fb_vs_rhp,
        babip_vs_rhp=babip_vs_rhp,
        speed_score=raw_speed,
        games_played=n,
        pa_total=int(total_pa),
    )


def build_league_average_batter() -> BatterModel:
    """Build a league-average batter model (for fallback/testing)."""
    return BatterModel(
        player_id=0,
        player_name="League Average",
        team_id=0,
        bats="R",
        k_pct=config.LEAGUE_AVG_K_PCT,
        bb_pct=config.LEAGUE_AVG_BB_PCT,
        hbp_pct=config.LEAGUE_AVG_HBP_PCT,
        gb_pct=config.LEAGUE_AVG_GB_PCT,
        fb_pct=config.LEAGUE_AVG_FB_PCT,
        ld_pct=config.LEAGUE_AVG_LD_PCT,
        hr_per_fb=config.LEAGUE_AVG_HR_PER_FB,
        babip=config.LEAGUE_AVG_BABIP,
        k_pct_vs_lhp=config.LEAGUE_AVG_K_PCT,
        bb_pct_vs_lhp=config.LEAGUE_AVG_BB_PCT,
        hr_per_fb_vs_lhp=config.LEAGUE_AVG_HR_PER_FB,
        babip_vs_lhp=config.LEAGUE_AVG_BABIP,
        k_pct_vs_rhp=config.LEAGUE_AVG_K_PCT,
        bb_pct_vs_rhp=config.LEAGUE_AVG_BB_PCT,
        hr_per_fb_vs_rhp=config.LEAGUE_AVG_HR_PER_FB,
        babip_vs_rhp=config.LEAGUE_AVG_BABIP,
        speed_score=5.0,
        games_played=0,
        pa_total=0,
    )
