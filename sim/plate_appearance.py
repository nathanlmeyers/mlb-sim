"""Core plate appearance simulation with count-progression Markov chain.

Simulates each PA pitch-by-pitch through a count state machine:
  (0,0) → each pitch is ball/called-strike/swinging-strike/foul/in-play/HBP
  Walk at 4 balls, strikeout at 3 strikes, fouls protect with 2 strikes.

TTO (Times Through the Order) penalty degrades pitcher effectiveness
each time the lineup cycles. Count-dependent swing/contact rates
capture pitcher command and batter plate discipline.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from models.batter_model import BatterModel
from models.pitcher_model import PitcherModel
from models.team_model import ParkFactor
import config


class PAOutcome(Enum):
    STRIKEOUT = "K"
    WALK = "BB"
    HIT_BY_PITCH = "HBP"
    SINGLE = "1B"
    DOUBLE = "2B"
    TRIPLE = "3B"
    HOME_RUN = "HR"
    GROUNDOUT = "GO"
    FLYOUT = "FO"
    LINEOUT = "LO"


@dataclass
class PAResult:
    outcome: PAOutcome
    batted_ball_type: str | None = None  # GB, FB, LD
    is_out: bool = False
    bases: int = 0  # 0 for outs, 1-4 for hits/walks
    is_double_play_eligible: bool = False  # GB out
    pitch_count: int = 0  # actual pitches thrown in this PA

    @property
    def is_home_run(self) -> bool:
        return self.outcome == PAOutcome.HOME_RUN


def _odds_ratio(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
    """Merge batter and pitcher rates using the odds-ratio (log5) method."""
    if league_rate <= 0:
        return batter_rate
    result = (batter_rate * pitcher_rate) / league_rate
    return np.clip(result, 0.001, 0.999)


def _get_platoon_rates(
    batter: BatterModel, pitcher: PitcherModel
) -> tuple[float, float, float, float, float, float, float, float]:
    """Get platoon-adjusted rates based on batter/pitcher handedness."""
    batter_hand = batter.bats
    if batter_hand == "S":
        batter_hand = "L" if pitcher.throws == "R" else "R"

    if pitcher.throws == "L":
        b_k = batter.k_pct_vs_lhp
        b_bb = batter.bb_pct_vs_lhp
        b_hr_fb = batter.hr_per_fb_vs_lhp
        b_babip = batter.babip_vs_lhp
    else:
        b_k = batter.k_pct_vs_rhp
        b_bb = batter.bb_pct_vs_rhp
        b_hr_fb = batter.hr_per_fb_vs_rhp
        b_babip = batter.babip_vs_rhp

    if batter_hand == "L":
        p_k = pitcher.k_pct_vs_lhb
        p_bb = pitcher.bb_pct_vs_lhb
        p_hr_fb = pitcher.hr_per_fb_vs_lhb
        p_babip = pitcher.babip_vs_lhb
    else:
        p_k = pitcher.k_pct_vs_rhb
        p_bb = pitcher.bb_pct_vs_rhb
        p_hr_fb = pitcher.hr_per_fb_vs_rhb
        p_babip = pitcher.babip_vs_rhb

    return b_k, b_bb, b_hr_fb, b_babip, p_k, p_bb, p_hr_fb, p_babip


def simulate_plate_appearance(
    batter: BatterModel,
    pitcher: PitcherModel,
    park: ParkFactor,
    fatigue_mult: float = 1.0,
    times_through_order: int = 1,
    is_home: bool = False,
    rng: np.random.Generator | None = None,
) -> PAResult:
    """Simulate one plate appearance with count-progression Markov chain.

    Each pitch progresses through the ball-strike count. The PA ends on:
    - 4 balls (walk)
    - 3 strikes (strikeout)
    - Ball in play (contact → GB/FB/LD → hit/out)
    - HBP (rare, ~0.4% per pitch)

    Args:
        batter: Batter statistical model
        pitcher: Pitcher statistical model
        park: Park factor for the venue
        fatigue_mult: Pitcher fatigue multiplier (1.0 = fresh)
        times_through_order: 1-4, how many times this batter's lineup spot
                             has faced this pitcher (for TTO penalty)
        is_home: Whether the batter is on the home team (for HFA)
        rng: numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get platoon-adjusted rates
    b_k, b_bb, b_hr_fb, b_babip, p_k, p_bb, p_hr_fb, p_babip = _get_platoon_rates(batter, pitcher)

    # Apply home field advantage
    if is_home:
        b_bb += config.HOME_BB_BOOST
        b_babip += config.HOME_BABIP_BOOST

    # Merge batter + pitcher via odds-ratio
    base_k = _odds_ratio(b_k, p_k, config.LEAGUE_AVG_K_PCT)
    base_bb = _odds_ratio(b_bb, p_bb, config.LEAGUE_AVG_BB_PCT)

    # Apply TTO penalty (pitcher gets worse each time through)
    tto = config.TTO_MULTIPLIERS.get(min(times_through_order, 4), config.TTO_MULTIPLIERS[4])
    tto_k = tto["k_pct"]
    tto_bb = tto["bb_pct"]
    tto_hr = tto["hr_per_fb"]
    tto_babip = tto["babip"]

    # Apply fatigue with TTO overlap scaling
    fatigue_effect = 1.0 + (fatigue_mult - 1.0) * config.FATIGUE_TTO_OVERLAP_FACTOR

    # Combined effective rates (TTO + fatigue)
    eff_k_pct = np.clip(base_k * tto_k / fatigue_effect, 0.05, 0.50)
    eff_bb_pct = np.clip(base_bb * tto_bb * fatigue_effect, 0.02, 0.25)

    # Derive per-pitch adjustment factors from PA-level rates.
    # These scale the count-level base rates so that players with
    # higher K% actually strike out more, and players with higher BB% walk more.
    # At league average, all factors = 1.0 (no adjustment).

    # batter_discipline: >1 = patient (swings less, takes more) → more BB
    batter_discipline = np.clip(eff_bb_pct / config.LEAGUE_AVG_BB_PCT, 0.7, 1.4)

    # pitcher_command: derived from actual strike% when available
    # Higher strike% = better command = fewer balls when batter takes
    if pitcher.strike_pct > 0 and pitcher.strike_pct != config.LEAGUE_AVG_STRIKE_PCT:
        pitcher_command = np.clip(
            config.LEAGUE_AVG_STRIKE_PCT / pitcher.strike_pct, 0.7, 1.4
        )
    else:
        pitcher_command = np.clip(eff_bb_pct / config.LEAGUE_AVG_BB_PCT, 0.7, 1.4)

    # Command degrades with fatigue — pitcher throws more balls as they tire
    if fatigue_effect > 1.0:
        command_decay = (fatigue_effect - 1.0) * 0.5
        pitcher_command *= (1.0 + command_decay)
        pitcher_command = min(pitcher_command, 1.5)

    # batter_contact: >1 = good contact (fewer whiffs) → fewer K
    batter_contact = np.clip(config.LEAGUE_AVG_K_PCT / max(eff_k_pct, 0.05), 0.7, 1.3)

    # Count-progression loop
    balls = 0
    strikes = 0
    pitches = 0
    max_pitches = 20  # safety valve

    while pitches < max_pitches:
        pitches += 1
        count = (balls, strikes)

        # HBP check (~0.4% per pitch)
        if rng.random() < config.HBP_PER_PITCH:
            return PAResult(outcome=PAOutcome.HIT_BY_PITCH, bases=1, pitch_count=pitches)

        # Get count-dependent base rates
        base_swing = config.COUNT_SWING_RATE.get(count, 0.46)
        base_contact = config.COUNT_CONTACT_RATE.get(count, 0.78)
        base_ball_prob = config.COUNT_BALL_PROB.get(count, 0.38)
        base_foul = config.COUNT_FOUL_RATE.get(count, 0.42)

        # Adjust for batter/pitcher quality
        # Patient batter (high discipline) swings LESS → divide by discipline
        swing_prob = base_swing / batter_discipline
        # Good contact hitter makes more contact
        contact_prob = base_contact * batter_contact
        # Wild pitcher (high command value) throws more balls
        ball_prob = base_ball_prob * pitcher_command

        swing_prob = np.clip(swing_prob, 0.15, 0.75)
        contact_prob = np.clip(contact_prob, 0.45, 0.95)
        ball_prob = np.clip(ball_prob, 0.20, 0.65)

        if rng.random() < swing_prob:
            # Batter swings
            if rng.random() < contact_prob:
                # Contact made
                if strikes < 2 and rng.random() < base_foul:
                    # Foul ball — adds a strike only if < 2 strikes
                    strikes += 1
                    continue
                elif strikes == 2 and rng.random() < base_foul * 0.6:
                    # Foul with 2 strikes — doesn't add a strike, extends AB
                    continue
                else:
                    # Ball in play — resolve outcome
                    return _resolve_ball_in_play(
                        batter, pitcher, park, b_hr_fb, p_hr_fb,
                        b_babip, p_babip, tto_hr, tto_babip,
                        fatigue_effect, is_home, pitches, rng,
                    )
            else:
                # Swing and miss
                strikes += 1
                if strikes >= 3:
                    return PAResult(outcome=PAOutcome.STRIKEOUT, is_out=True, pitch_count=pitches)
        else:
            # Batter takes the pitch
            if rng.random() < ball_prob:
                # Ball
                balls += 1
                if balls >= 4:
                    return PAResult(outcome=PAOutcome.WALK, bases=1, pitch_count=pitches)
            else:
                # Called strike
                strikes += 1
                if strikes >= 3:
                    return PAResult(outcome=PAOutcome.STRIKEOUT, is_out=True, pitch_count=pitches)

    # Safety: if we somehow reach max pitches, resolve as ball in play
    return _resolve_ball_in_play(
        batter, pitcher, park, b_hr_fb, p_hr_fb,
        b_babip, p_babip, 1.0, 1.0,
        fatigue_effect, is_home, pitches, rng,
    )


def _resolve_ball_in_play(
    batter: BatterModel,
    pitcher: PitcherModel,
    park: ParkFactor,
    b_hr_fb: float,
    p_hr_fb: float,
    b_babip: float,
    p_babip: float,
    tto_hr_mult: float,
    tto_babip_mult: float,
    fatigue_effect: float,
    is_home: bool,
    pitch_count: int,
    rng: np.random.Generator,
) -> PAResult:
    """Resolve a ball in play into GB/FB/LD → hit/out."""
    # Determine batted ball type
    eff_gb = (batter.gb_pct + pitcher.gb_pct) / 2
    eff_fb = (batter.fb_pct + pitcher.fb_pct) / 2
    eff_ld = (batter.ld_pct + pitcher.ld_pct) / 2

    bb_total = eff_gb + eff_fb + eff_ld
    if bb_total > 0:
        eff_gb /= bb_total
        eff_fb /= bb_total
        eff_ld /= bb_total

    roll = rng.random()

    if roll < eff_gb:
        return _resolve_ground_ball(batter, pitcher, park, tto_babip_mult, fatigue_effect, is_home, pitch_count, rng)
    elif roll < eff_gb + eff_fb:
        return _resolve_fly_ball(batter, pitcher, park, b_hr_fb, p_hr_fb, tto_hr_mult, fatigue_effect, pitch_count, rng)
    else:
        return _resolve_line_drive(batter, pitcher, park, b_babip, p_babip, tto_babip_mult, tto_hr_mult, fatigue_effect, is_home, pitch_count, rng)


def _resolve_ground_ball(
    batter: BatterModel,
    pitcher: PitcherModel,
    park: ParkFactor,
    tto_babip: float,
    fatigue_effect: float,
    is_home: bool,
    pitch_count: int,
    rng: np.random.Generator,
) -> PAResult:
    """Resolve a ground ball outcome with normalized probabilities."""
    babip_adj = _odds_ratio(batter.babip, pitcher.babip_allowed, config.LEAGUE_AVG_BABIP)
    if is_home:
        babip_adj += config.HOME_BABIP_BOOST
    babip_adj *= park.h_factor * tto_babip * fatigue_effect
    hit_mult = babip_adj / config.LEAGUE_AVG_BABIP

    p_infield_hit = config.P_INFIELD_HIT_ON_GB * (1.0 + batter.speed_score * 0.03)
    p_single = config.P_SINGLE_ON_GB * hit_mult
    p_double = config.P_DOUBLE_ON_GB * hit_mult

    total_hit = p_infield_hit + p_single + p_double
    p_out = max(config.P_OUT_ON_GB * 0.85, 1.0 - total_hit)
    total = total_hit + p_out
    p_infield_hit /= total
    p_single /= total
    p_double /= total

    roll = rng.random()
    if roll < p_infield_hit:
        return PAResult(outcome=PAOutcome.SINGLE, batted_ball_type="GB", bases=1, pitch_count=pitch_count)
    elif roll < p_infield_hit + p_single:
        return PAResult(outcome=PAOutcome.SINGLE, batted_ball_type="GB", bases=1, pitch_count=pitch_count)
    elif roll < p_infield_hit + p_single + p_double:
        return PAResult(outcome=PAOutcome.DOUBLE, batted_ball_type="GB", bases=2, pitch_count=pitch_count)
    else:
        return PAResult(outcome=PAOutcome.GROUNDOUT, batted_ball_type="GB",
                        is_out=True, is_double_play_eligible=True, pitch_count=pitch_count)


def _resolve_fly_ball(
    batter: BatterModel,
    pitcher: PitcherModel,
    park: ParkFactor,
    b_hr_fb: float,
    p_hr_fb: float,
    tto_hr: float,
    fatigue_effect: float,
    pitch_count: int,
    rng: np.random.Generator,
) -> PAResult:
    """Resolve a fly ball outcome."""
    eff_hr_fb = _odds_ratio(b_hr_fb, p_hr_fb, config.LEAGUE_AVG_HR_PER_FB)
    eff_hr_fb *= park.hr_factor * tto_hr * fatigue_effect
    eff_hr_fb = np.clip(eff_hr_fb, 0.03, 0.35)

    if rng.random() < eff_hr_fb:
        return PAResult(outcome=PAOutcome.HOME_RUN, batted_ball_type="FB", bases=4, pitch_count=pitch_count)

    non_hr_total = config.P_SINGLE_ON_FB + config.P_DOUBLE_ON_FB + config.P_TRIPLE_ON_FB + config.P_OUT_ON_FB
    p_single = config.P_SINGLE_ON_FB / non_hr_total
    p_double = config.P_DOUBLE_ON_FB / non_hr_total
    p_triple = config.P_TRIPLE_ON_FB / non_hr_total

    roll = rng.random()
    if roll < p_single:
        return PAResult(outcome=PAOutcome.SINGLE, batted_ball_type="FB", bases=1, pitch_count=pitch_count)
    elif roll < p_single + p_double:
        return PAResult(outcome=PAOutcome.DOUBLE, batted_ball_type="FB", bases=2, pitch_count=pitch_count)
    elif roll < p_single + p_double + p_triple:
        return PAResult(outcome=PAOutcome.TRIPLE, batted_ball_type="FB", bases=3, pitch_count=pitch_count)
    else:
        return PAResult(outcome=PAOutcome.FLYOUT, batted_ball_type="FB", is_out=True, pitch_count=pitch_count)


def _resolve_line_drive(
    batter: BatterModel,
    pitcher: PitcherModel,
    park: ParkFactor,
    b_babip: float,
    p_babip: float,
    tto_babip: float,
    tto_hr: float,
    fatigue_effect: float,
    is_home: bool,
    pitch_count: int,
    rng: np.random.Generator,
) -> PAResult:
    """Resolve a line drive outcome with normalized probabilities."""
    babip_adj = _odds_ratio(b_babip, p_babip, config.LEAGUE_AVG_BABIP)
    if is_home:
        babip_adj += config.HOME_BABIP_BOOST
    babip_adj *= park.h_factor * tto_babip * fatigue_effect
    hit_mult = babip_adj / config.LEAGUE_AVG_BABIP

    p_hr = config.P_HR_ON_LD * park.hr_factor * tto_hr
    p_single = config.P_SINGLE_ON_LD * hit_mult
    p_double = config.P_DOUBLE_ON_LD * hit_mult
    p_triple = config.P_TRIPLE_ON_LD * hit_mult

    total_hit = p_hr + p_single + p_double + p_triple
    p_out = max(config.P_OUT_ON_LD * 0.80, 1.0 - total_hit)
    total = total_hit + p_out
    p_hr /= total
    p_single /= total
    p_double /= total
    p_triple /= total

    roll = rng.random()
    if roll < p_hr:
        return PAResult(outcome=PAOutcome.HOME_RUN, batted_ball_type="LD", bases=4, pitch_count=pitch_count)
    elif roll < p_hr + p_triple:
        return PAResult(outcome=PAOutcome.TRIPLE, batted_ball_type="LD", bases=3, pitch_count=pitch_count)
    elif roll < p_hr + p_triple + p_double:
        return PAResult(outcome=PAOutcome.DOUBLE, batted_ball_type="LD", bases=2, pitch_count=pitch_count)
    elif roll < p_hr + p_triple + p_double + p_single:
        return PAResult(outcome=PAOutcome.SINGLE, batted_ball_type="LD", bases=1, pitch_count=pitch_count)
    else:
        return PAResult(outcome=PAOutcome.LINEOUT, batted_ball_type="LD", is_out=True, pitch_count=pitch_count)
