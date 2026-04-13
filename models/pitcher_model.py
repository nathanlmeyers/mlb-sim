"""Pitcher statistical model with exponential decay and Bayesian regression.

Each pitcher gets a model built from their recent game logs, with:
- Exponential decay weighting (recent starts matter more)
- Bayesian regression to league means (handles small samples)
- Platoon splits (vs LHB / vs RHB)
- Stamina tracking (avg pitches per start, fatigue modeling)
"""

from dataclasses import dataclass
import numpy as np
from sqlalchemy import select, and_
from data.db import engine, player_pitching_stats
import config


@dataclass
class PitcherModel:
    player_id: int
    player_name: str
    team_id: int
    throws: str  # L or R

    # Core rate stats (per batter faced)
    k_pct: float
    bb_pct: float
    hbp_pct: float

    # Batted ball profile allowed
    gb_pct: float
    fb_pct: float
    ld_pct: float

    # Quality of contact allowed
    hr_per_fb: float
    babip_allowed: float

    # Platoon splits (vs LHB)
    k_pct_vs_lhb: float
    bb_pct_vs_lhb: float
    hr_per_fb_vs_lhb: float
    babip_vs_lhb: float

    # Platoon splits (vs RHB)
    k_pct_vs_rhb: float
    bb_pct_vs_rhb: float
    hr_per_fb_vs_rhb: float
    babip_vs_rhb: float

    # Stamina
    avg_pitches_per_start: float
    avg_innings_per_start: float

    # Role
    is_starter: bool
    is_closer: bool

    # Sample sizes
    games_played: int
    batters_faced_total: int

    # Pitch command (strike% from API — most important command metric)
    strike_pct: float = 0.62  # % of pitches that are strikes


def _regress(observed: float, sample_size: float, league_mean: float, regression_n: float) -> float:
    """Bayesian regression toward league mean."""
    if sample_size + regression_n == 0:
        return league_mean
    weight = sample_size / (sample_size + regression_n)
    return weight * observed + (1 - weight) * league_mean


def _exponential_decay_weights(n_games: int, lam: float) -> np.ndarray:
    """Generate exponential decay weights for n games."""
    indices = np.arange(n_games)
    return np.exp(-lam * indices)


def get_fatigue_multiplier(pitches_thrown: int) -> float:
    """Returns a fatigue multiplier (1.0 = fresh, >1.0 = degraded).

    Applied to BB% and BABIP allowed (increase with fatigue).
    Applied inversely to K% (decreases with fatigue).
    """
    if pitches_thrown <= config.STARTER_FATIGUE_ONSET:
        return 1.0
    excess = pitches_thrown - config.STARTER_FATIGUE_ONSET
    return 1.0 + excess * config.FATIGUE_DEGRADATION_PER_PITCH


def build_pitcher_model(
    player_id: int,
    as_of_date,
    max_games: int = 30,
) -> PitcherModel | None:
    """Build a pitcher model from game logs prior to as_of_date.

    Uses exponential decay weighting and Bayesian regression.
    Returns None if the pitcher has no games in the window.
    """
    with engine.connect() as conn:
        stmt = (
            select(player_pitching_stats)
            .where(
                and_(
                    player_pitching_stats.c.player_id == player_id,
                    player_pitching_stats.c.game_date < as_of_date,
                )
            )
            .order_by(player_pitching_stats.c.game_date.desc())
            .limit(max_games)
        )
        rows = conn.execute(stmt).fetchall()

    if not rows:
        return None

    n = len(rows)
    weights = _exponential_decay_weights(n, config.PITCHER_DECAY_LAMBDA)

    # Extract arrays
    bf = np.array([r.batters_faced or 0 for r in rows], dtype=float)
    k = np.array([r.k or 0 for r in rows], dtype=float)
    bb = np.array([r.bb or 0 for r in rows], dtype=float)
    hbp = np.array([r.hbp or 0 for r in rows], dtype=float)
    hr = np.array([r.hr_allowed or 0 for r in rows], dtype=float)
    hits = np.array([r.hits_allowed or 0 for r in rows], dtype=float)
    ip = np.array([r.innings_pitched or 0 for r in rows], dtype=float)
    pitches = np.array([r.pitches_thrown or 0 for r in rows], dtype=float)
    gb = np.array([r.gb or 0 for r in rows], dtype=float)
    fb = np.array([r.fb or 0 for r in rows], dtype=float)
    ld = np.array([r.ld or 0 for r in rows], dtype=float)

    total_bf = bf.sum()
    if total_bf == 0:
        return None

    weighted_bf = np.dot(weights, bf)

    # Core rate stats
    raw_k_pct = np.dot(weights, k) / weighted_bf if weighted_bf > 0 else config.LEAGUE_AVG_K_PCT
    raw_bb_pct = np.dot(weights, bb) / weighted_bf if weighted_bf > 0 else config.LEAGUE_AVG_BB_PCT
    raw_hbp_pct = np.dot(weights, hbp) / weighted_bf if weighted_bf > 0 else config.LEAGUE_AVG_HBP_PCT

    # Batted ball profile
    total_batted = gb + fb + ld
    weighted_batted = np.dot(weights, total_batted)
    if weighted_batted > 0:
        raw_gb_pct = np.dot(weights, gb) / weighted_batted
        raw_fb_pct = np.dot(weights, fb) / weighted_batted
        raw_ld_pct = np.dot(weights, ld) / weighted_batted
    else:
        raw_gb_pct = config.LEAGUE_AVG_GB_PCT
        raw_fb_pct = config.LEAGUE_AVG_FB_PCT
        raw_ld_pct = config.LEAGUE_AVG_LD_PCT

    # HR/FB rate
    weighted_fb = np.dot(weights, fb)
    raw_hr_per_fb = np.dot(weights, hr) / weighted_fb if weighted_fb > 0 else config.LEAGUE_AVG_HR_PER_FB

    # BABIP allowed
    h_minus_hr = hits - hr
    bip_for_babip = bf - k - bb - hbp - hr
    weighted_h = np.dot(weights, np.maximum(h_minus_hr, 0))
    weighted_bip = np.dot(weights, np.maximum(bip_for_babip, 0))
    raw_babip = weighted_h / weighted_bip if weighted_bip > 0 else config.LEAGUE_AVG_BABIP

    # Apply Bayesian regression
    k_pct = _regress(raw_k_pct, total_bf, config.LEAGUE_AVG_K_PCT, config.REGRESSION_SAMPLES["p_k_pct"])
    bb_pct = _regress(raw_bb_pct, total_bf, config.LEAGUE_AVG_BB_PCT, config.REGRESSION_SAMPLES["p_bb_pct"])
    hbp_pct = _regress(raw_hbp_pct, total_bf, config.LEAGUE_AVG_HBP_PCT, config.REGRESSION_SAMPLES["p_hbp_pct"])
    gb_pct = _regress(raw_gb_pct, weighted_batted, config.LEAGUE_AVG_GB_PCT, config.REGRESSION_SAMPLES["p_gb_pct"])
    fb_pct = _regress(raw_fb_pct, weighted_batted, config.LEAGUE_AVG_FB_PCT, config.REGRESSION_SAMPLES["p_fb_pct"])
    ld_pct = _regress(raw_ld_pct, weighted_batted, config.LEAGUE_AVG_LD_PCT, config.REGRESSION_SAMPLES["p_ld_pct"])
    hr_per_fb = _regress(raw_hr_per_fb, weighted_fb, config.LEAGUE_AVG_HR_PER_FB, config.REGRESSION_SAMPLES["p_hr_per_fb"])
    babip_allowed = _regress(raw_babip, weighted_bip, config.LEAGUE_AVG_BABIP, config.REGRESSION_SAMPLES["p_babip"])

    # Normalize batted ball profile
    bb_total = gb_pct + fb_pct + ld_pct
    if bb_total > 0:
        gb_pct /= bb_total
        fb_pct /= bb_total
        ld_pct /= bb_total

    # Stamina
    starter_rows = [r for r in rows if r.is_starter]
    if starter_rows:
        avg_pitches = np.mean([r.pitches_thrown or 0 for r in starter_rows])
        avg_ip = np.mean([r.innings_pitched or 0 for r in starter_rows])
    else:
        avg_pitches = 20.0  # reliever default
        avg_ip = 1.0

    # Role detection
    is_starter = sum(1 for r in rows if r.is_starter) > len(rows) * 0.5
    is_closer = not is_starter and n >= 5

    throws = rows[0].throws or "R"

    # Platoon splits: LHP gets advantage vs RHB, disadvantage vs LHB (and vice versa)
    if throws == "L":
        # LHP: better vs LHB (same side), worse vs RHB (opposite)
        k_pct_vs_lhb = k_pct * 1.08
        k_pct_vs_rhb = k_pct * 0.92
        bb_pct_vs_lhb = bb_pct * 0.90
        bb_pct_vs_rhb = bb_pct * 1.10
        hr_per_fb_vs_lhb = hr_per_fb * 0.85
        hr_per_fb_vs_rhb = hr_per_fb * 1.15
        babip_vs_lhb = babip_allowed - 0.015
        babip_vs_rhb = babip_allowed + 0.015
    else:
        # RHP: better vs RHB (same side), worse vs LHB (opposite)
        k_pct_vs_rhb = k_pct * 1.08
        k_pct_vs_lhb = k_pct * 0.92
        bb_pct_vs_rhb = bb_pct * 0.90
        bb_pct_vs_lhb = bb_pct * 1.10
        hr_per_fb_vs_rhb = hr_per_fb * 0.85
        hr_per_fb_vs_lhb = hr_per_fb * 1.15
        babip_vs_rhb = babip_allowed - 0.015
        babip_vs_lhb = babip_allowed + 0.015

    return PitcherModel(
        player_id=player_id,
        player_name=rows[0].player_name,
        team_id=rows[0].team_id,
        throws=throws,
        k_pct=k_pct,
        bb_pct=bb_pct,
        hbp_pct=hbp_pct,
        gb_pct=gb_pct,
        fb_pct=fb_pct,
        ld_pct=ld_pct,
        hr_per_fb=hr_per_fb,
        babip_allowed=babip_allowed,
        k_pct_vs_lhb=k_pct_vs_lhb,
        bb_pct_vs_lhb=bb_pct_vs_lhb,
        hr_per_fb_vs_lhb=hr_per_fb_vs_lhb,
        babip_vs_lhb=babip_vs_lhb,
        k_pct_vs_rhb=k_pct_vs_rhb,
        bb_pct_vs_rhb=bb_pct_vs_rhb,
        hr_per_fb_vs_rhb=hr_per_fb_vs_rhb,
        babip_vs_rhb=babip_vs_rhb,
        avg_pitches_per_start=avg_pitches,
        avg_innings_per_start=avg_ip,
        is_starter=is_starter,
        is_closer=is_closer,
        games_played=n,
        batters_faced_total=int(total_bf),
    )


def build_league_average_pitcher(is_starter: bool = True) -> PitcherModel:
    """Build a league-average pitcher model (for fallback/testing)."""
    return PitcherModel(
        player_id=0,
        player_name="League Average",
        team_id=0,
        throws="R",
        k_pct=config.LEAGUE_AVG_K_PCT,
        bb_pct=config.LEAGUE_AVG_BB_PCT,
        hbp_pct=config.LEAGUE_AVG_HBP_PCT,
        gb_pct=config.LEAGUE_AVG_GB_PCT,
        fb_pct=config.LEAGUE_AVG_FB_PCT,
        ld_pct=config.LEAGUE_AVG_LD_PCT,
        hr_per_fb=config.LEAGUE_AVG_HR_PER_FB,
        babip_allowed=config.LEAGUE_AVG_BABIP,
        k_pct_vs_lhb=config.LEAGUE_AVG_K_PCT,
        bb_pct_vs_lhb=config.LEAGUE_AVG_BB_PCT,
        hr_per_fb_vs_lhb=config.LEAGUE_AVG_HR_PER_FB,
        babip_vs_lhb=config.LEAGUE_AVG_BABIP,
        k_pct_vs_rhb=config.LEAGUE_AVG_K_PCT,
        bb_pct_vs_rhb=config.LEAGUE_AVG_BB_PCT,
        hr_per_fb_vs_rhb=config.LEAGUE_AVG_HR_PER_FB,
        babip_vs_rhb=config.LEAGUE_AVG_BABIP,
        avg_pitches_per_start=90.0 if is_starter else 20.0,
        avg_innings_per_start=5.5 if is_starter else 1.0,
        is_starter=is_starter,
        is_closer=False,
        games_played=0,
        batters_faced_total=0,
    )
