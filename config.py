"""Configuration constants for MLB simulator."""

import os

DATABASE_URL = os.environ.get("MLB_SIM_DB_URL", "postgresql://localhost/mlb_sim")

# MLB Stats API rate limiting
API_DELAY_SECONDS = 1.0

# Current season
CURRENT_SEASON = 2025

# Simulation defaults
DEFAULT_NUM_SIMS = 10_000
INNINGS_PER_GAME = 9

# Home field advantage
# MLB historical home win rate: ~54% (2010-2024 average)
# Translates to roughly +0.24 runs/game for home team
HOME_FIELD_ADVANTAGE_RUNS = 0.24

# Exponential decay parameters
# Lambda controls how fast old games lose weight: weight = exp(-lambda * i)
# where i = 0 is most recent game
BATTER_DECAY_LAMBDA = 0.08   # half-life ~9 games
PITCHER_DECAY_LAMBDA = 0.05  # half-life ~14 games (more stable start-to-start)
TEAM_DECAY_LAMBDA = 0.03     # half-life ~23 games (most stable)

# Bayesian regression samples (higher = more regression to league mean)
# Key: volatile stats need MORE regression samples
REGRESSION_SAMPLES = {
    # Batter stats
    "k_pct": 80,          # very stable, stabilizes ~60 PA
    "bb_pct": 120,        # moderately stable
    "hbp_pct": 300,       # rare event, needs heavy regression
    "hr_per_fb": 300,     # very volatile (small sample)
    "babip": 400,         # extremely noisy, strong mean-reversion
    "gb_pct": 60,         # stable batted ball profile
    "fb_pct": 60,
    "ld_pct": 100,
    # Pitcher stats
    "p_k_pct": 60,        # very stable for pitchers
    "p_bb_pct": 100,      # moderately stable
    "p_hbp_pct": 300,
    "p_hr_per_fb": 350,   # very volatile
    "p_babip": 500,       # pitchers have almost no control over BABIP
    "p_gb_pct": 50,       # very stable (pitch mix driven)
    "p_fb_pct": 50,
    "p_ld_pct": 120,
}

# League averages (2024 MLB season — sourced from Statcast/FanGraphs)
LEAGUE_AVG_K_PCT = 0.2258      # Statcast: 41197 SO / 182449 PA
LEAGUE_AVG_BB_PCT = 0.0818     # Statcast: 14929 BB / 182449 PA
LEAGUE_AVG_HBP_PCT = 0.0111    # Statcast: 2020 HBP / 182449 PA
LEAGUE_AVG_HR_PER_FB = 0.116   # FanGraphs 2024
LEAGUE_AVG_BABIP = 0.291       # FanGraphs 2024
LEAGUE_AVG_GB_PCT = 0.422      # FanGraphs 2024
LEAGUE_AVG_FB_PCT = 0.381      # FanGraphs 2024 (includes popups)
LEAGUE_AVG_LD_PCT = 0.196      # FanGraphs 2024
LEAGUE_AVG_RUNS_PER_GAME = 4.39  # Baseball Reference 2024
LEAGUE_AVG_FIP = 4.10          # FanGraphs 2024 (approx)

# Pitcher fatigue thresholds
STARTER_FATIGUE_ONSET = 75             # pitches: effectiveness starts declining
STARTER_PITCH_COUNT_THRESHOLD = 95     # pitches: avg starter pulled around here
FATIGUE_DEGRADATION_PER_PITCH = 0.003  # 0.3% worse per pitch after onset
# After 100 pitches: 0.003 * 25 = 7.5% degradation

# Bullpen management
RELIEVER_MAX_OUTS = 6      # 2 innings max for non-closers
CLOSER_ENTRY_INNING = 9    # closers typically enter 9th
CLOSER_MAX_OUTS = 3        # 1 inning

# Estimated pitches per plate appearance (for pitch count tracking)
AVG_PITCHES_PER_PA = 3.9

# Platoon split adjustments (additive wOBA-scale adjustments)
PLATOON_ADJ_BATTER_VS_SAME = -0.015     # RHB vs RHP or LHB vs LHP: disadvantage
PLATOON_ADJ_BATTER_VS_OPPOSITE = 0.015  # RHB vs LHP or LHB vs RHP: advantage

# Park factor neutral value
DEFAULT_PARK_FACTOR = 1.00

# Batted ball outcome probabilities (2024 Statcast + FanGraphs derived)
# Ground balls (GB BABIP ~0.239, IFH% = 6.7%)
P_OUT_ON_GB = 0.761
P_SINGLE_ON_GB = 0.172
P_DOUBLE_ON_GB = 0.008
P_INFIELD_HIT_ON_GB = 0.067   # FanGraphs IFH% on GB
# Fly balls — after removing HR (HR/FB handled separately)
P_OUT_ON_FB = 0.903            # includes popups (IFFB 10.3%)
P_SINGLE_ON_FB = 0.029
P_DOUBLE_ON_FB = 0.044
P_TRIPLE_ON_FB = 0.010
# Line drives (LD BABIP ~0.670, LD HR ~2.8%)
P_OUT_ON_LD = 0.330
P_SINGLE_ON_LD = 0.449
P_DOUBLE_ON_LD = 0.161
P_TRIPLE_ON_LD = 0.019
P_HR_ON_LD = 0.028

# Double play probability (on ground ball out with runner on 1st and <2 outs)
# Baseball Reference 2024: 3227 GIDP / ~10818 GB outs in DP situations = 29.6%
P_DOUBLE_PLAY_ON_GB = 0.30

# Sacrifice fly probability (fly out with runner on 3rd and <2 outs)
# Baseball Reference 2024: lt_2_out_on_third scored 51.2% (all outcomes)
# Sac fly specifically ~65% of fly outs with runner on 3rd
P_SAC_FLY = 0.65

# Baserunner advancement probabilities (Baseball Reference 2024)
P_SCORE_FROM_THIRD_ON_SINGLE = 0.89   # estimated from run expectancy
P_SCORE_FROM_SECOND_ON_SINGLE = 0.617 # 3010/4881 — was 0.45!
P_ADVANCE_FIRST_TO_THIRD_ON_SINGLE = 0.317  # 2615/8241 — was 0.28
P_SCORE_FROM_SECOND_ON_DOUBLE = 0.95  # confirmed near-certain
P_SCORE_FROM_FIRST_ON_DOUBLE = 0.381  # 936/2455 — was 0.44
P_SCORE_FROM_FIRST_ON_TRIPLE = 0.98   # confirmed near-certain

# Extra innings (Manfred runner rule, since 2020)
MANFRED_RUNNER = True
EXTRA_INNING_EXPECTED_RUNS = 0.80  # per half-inning with runner on 2nd

# Pythagorean expectation exponent for MLB
PYTHAG_EXPONENT = 1.83

# Starter vs bullpen innings share
STARTER_INNINGS_SHARE = 0.30  # dampened from 0.60 — early-season pitcher stats are noisy

# Momentum adjustment (runs per unit of avg margin in last 5 games)
MOMENTUM_FACTOR = 0.02

# Negative binomial dispersion for box score model
# MLB variance ≈ 2.1 × mean → r = mu²/(var - mu) ≈ 4.0
NEGATIVE_BINOMIAL_R = 4.0

# Home field advantage for detailed model (PA-level boosts)
# Calibrated to produce ~0.24 runs/game advantage (~54% home win rate)
HOME_BABIP_BOOST = 0.012    # +1.2% BABIP for home batters
HOME_BB_BOOST = 0.005       # +0.5% BB rate for home batters

# Ensemble weights (initial, to be trained via backtest)
TRAINED_WEIGHT_BOX_SCORE = 0.60
TRAINED_WEIGHT_DETAILED = 0.40
TRAINED_RECORD_ANCHOR = 0.30
TRAINED_CALIBRATION_A = 1.20
TRAINED_CALIBRATION_B = 0.01

# Times Through the Order (TTO) penalty multipliers
# Batters get dramatically better each time they face the same pitcher.
# Source: FanGraphs/Baseball Prospectus research, stable year-to-year.
# Applied as multipliers to pitcher's baseline rates.
TTO_MULTIPLIERS = {
    1: {"k_pct": 1.000, "bb_pct": 1.000, "hr_per_fb": 1.000, "babip": 1.000},
    2: {"k_pct": 0.915, "bb_pct": 1.065, "hr_per_fb": 1.070, "babip": 1.017},
    3: {"k_pct": 0.830, "bb_pct": 1.130, "hr_per_fb": 1.180, "babip": 1.050},
    4: {"k_pct": 0.770, "bb_pct": 1.180, "hr_per_fb": 1.280, "babip": 1.070},
}

# When combining TTO + pitch count fatigue, scale fatigue down by 30%
# to avoid double-counting the correlated portion
FATIGUE_TTO_OVERLAP_FACTOR = 0.70

# Count-based pitch outcome probabilities
# P(ball | batter takes) — NOT P(ball | pitch). Since batters preferentially
# swing at strikes and take balls, P(ball|take) >> P(ball|pitch).
# Calibrated to produce ~8% BB rate and ~23% K rate at the PA level.
COUNT_BALL_PROB = {
    (0, 0): 0.63, (1, 0): 0.64, (2, 0): 0.69, (3, 0): 0.73,
    (0, 1): 0.56, (1, 1): 0.58, (2, 1): 0.63, (3, 1): 0.69,
    (0, 2): 0.50, (1, 2): 0.53, (2, 2): 0.57, (3, 2): 0.64,
}

# P(batter swings) by count — MLB avg ~47% overall swing rate
# Batters swing more with 2 strikes (protect the plate), less in hitter's counts
COUNT_SWING_RATE = {
    (0, 0): 0.47, (1, 0): 0.44, (2, 0): 0.41, (3, 0): 0.32,
    (0, 1): 0.52, (1, 1): 0.50, (2, 1): 0.47, (3, 1): 0.40,
    (0, 2): 0.60, (1, 2): 0.58, (2, 2): 0.55, (3, 2): 0.52,
}

# P(contact | swing) by count — MLB avg ~78% contact rate
# Must be high enough that K% stays at ~22-23% with the swing rates above
COUNT_CONTACT_RATE = {
    (0, 0): 0.86, (1, 0): 0.87, (2, 0): 0.89, (3, 0): 0.92,
    (0, 1): 0.83, (1, 1): 0.84, (2, 1): 0.86, (3, 1): 0.89,
    (0, 2): 0.78, (1, 2): 0.79, (2, 2): 0.81, (3, 2): 0.83,
}

# P(contact is foul | contact made)
# Fouls are very common — ~40-45% of contact is foul in MLB
# With 2 strikes, the 2-strike foul protection extends ABs significantly
COUNT_FOUL_RATE = {
    (0, 0): 0.45, (1, 0): 0.45, (2, 0): 0.45, (3, 0): 0.45,
    (0, 1): 0.47, (1, 1): 0.47, (2, 1): 0.47, (3, 1): 0.47,
    (0, 2): 0.60, (1, 2): 0.60, (2, 2): 0.60, (3, 2): 0.60,
}

# HBP probability per pitch (~0.4%)
HBP_PER_PITCH = 0.004

# Pitcher command (strike%)
LEAGUE_AVG_STRIKE_PCT = 0.62  # MLB avg ~62% of pitches are strikes

# Market prior (Kalshi/Vegas) blending
# Blend model prediction with market odds for better calibration
# NBA sim uses 0.70 after grid search on 200 games; starting higher based on
# paper trading losses showing old 0.50 weight was too lenient vs sharp markets.
TRAINED_MARKET_WEIGHT = 0.65  # 0=pure model, 1=pure market. Grid search to optimize.
MARKET_CONFIDENCE_DISCOUNT = 0.30  # reduce market weight by 30% when engines agree

# Bet selection thresholds (tightened after day-2 paper losses against sharp markets)
MIN_EDGE_THRESHOLD = 0.04  # minimum 4% edge to log a bet (was 2%)
MIN_MARKET_PRICE = 0.15    # skip bets where market price < 15% (don't bet vs strong conviction)
MAX_MARKET_PRICE = 0.85    # skip bets where market price > 85% (don't fade heavy favorites)

# Betting parameters
MIN_BET_EV = 0.03          # 3% minimum edge to recommend a bet
KELLY_FRACTION = 0.125     # eighth-Kelly for conservative sizing
MAX_BET_PCT = 0.03         # max 3% of bankroll per game

# Weather adjustment constants
WEATHER_BASELINE_TEMP_C = 20.0     # baseline temperature (°C)
WEATHER_HR_PCT_PER_DEGREE = 0.0196 # +1.96% HR per 1°C above baseline
WEATHER_WIND_OUT_ADJUSTMENT = 0.5  # +0.5 runs for 10+ mph wind out to CF
WEATHER_WIND_IN_ADJUSTMENT = -0.5  # -0.5 runs for 10+ mph wind in from CF

# Schedule adjustment constants (additive runs)
SCHEDULE_REST_DAY_BOOST = 0.10       # off-day before game
SCHEDULE_DAY_AFTER_NIGHT = -0.15     # day game after night game
SCHEDULE_EAST_TRAVEL_PENALTY = -0.10 # eastward 2+ timezone change
SCHEDULE_FATIGUE_PENALTY = -0.05     # >6 games in last 7 days
