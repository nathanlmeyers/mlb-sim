"""Box score model (simple engine) for MLB simulation.

Estimates game outcomes from team-level stats using negative binomial
scoring distributions. Analogous to the NBA sim's simple engine.
This is the faster model that doesn't simulate individual PAs.
"""

from dataclasses import dataclass
import numpy as np
from models.pitcher_model import PitcherModel
from models.team_model import TeamModel, ParkFactor
import config


@dataclass
class GameResult:
    home_score: int
    away_score: int
    innings: int = 9

    @property
    def home_win(self) -> bool:
        return self.home_score > self.away_score


def _starter_quality_factor(pitcher: PitcherModel) -> float:
    """Compute how much a starter adjusts opponent run scoring.

    Uses a FIP-like composite from pitcher component stats, normalized
    so that league-average rates produce a factor of exactly 1.0.

    Returns a multiplier: <1.0 means elite pitcher (suppresses runs),
    >1.0 means weak pitcher (allows more runs).
    """
    # Component ratios vs league average (>1.0 = worse than average)
    hr_ratio = (pitcher.hr_per_fb * pitcher.fb_pct) / (config.LEAGUE_AVG_HR_PER_FB * config.LEAGUE_AVG_FB_PCT)
    bb_ratio = pitcher.bb_pct / config.LEAGUE_AVG_BB_PCT
    k_ratio = pitcher.k_pct / config.LEAGUE_AVG_K_PCT  # >1.0 = BETTER (more K)

    # Weighted composite: HR most important, then BB, then K (inverse)
    # Weights roughly mirror FIP coefficient magnitudes (13, 3, 2)
    factor = 0.45 * hr_ratio + 0.30 * bb_ratio + 0.25 * (1.0 / k_ratio)

    return np.clip(factor, 0.60, 1.50)


def simulate_box_score_game(
    home_team: TeamModel,
    away_team: TeamModel,
    home_starter: PitcherModel,
    away_starter: PitcherModel,
    park: ParkFactor,
    rng: np.random.Generator | None = None,
) -> GameResult:
    """Simulate a game using team-level stats and starter quality.

    Uses negative binomial distribution for run scoring (handles
    overdispersion better than Poisson).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Starter/bullpen split: starter only pitches ~60% of innings
    # Blend starter quality with league-average bullpen (1.0)
    sw = config.STARTER_INNINGS_SHARE
    away_pitching_factor = sw * _starter_quality_factor(away_starter) + (1 - sw) * 1.0
    home_pitching_factor = sw * _starter_quality_factor(home_starter) + (1 - sw) * 1.0

    home_expected = (
        home_team.runs_per_game
        * away_pitching_factor
        * park.overall_factor
        + config.HOME_FIELD_ADVANTAGE_RUNS
    )
    away_expected = (
        away_team.runs_per_game
        * home_pitching_factor
        * park.overall_factor
    )

    home_expected = np.clip(home_expected, 1.0, 12.0)
    away_expected = np.clip(away_expected, 1.0, 12.0)

    # Negative binomial: mean = mu, variance = mu + mu²/r
    r = config.NEGATIVE_BINOMIAL_R

    home_score = rng.negative_binomial(r, r / (r + home_expected))
    away_score = rng.negative_binomial(r, r / (r + away_expected))

    innings = 9

    while home_score == away_score:
        innings += 1
        home_extra = rng.poisson(config.EXTRA_INNING_EXPECTED_RUNS)
        away_extra = rng.poisson(config.EXTRA_INNING_EXPECTED_RUNS)
        home_score += home_extra
        away_score += away_extra
        if innings > 18:
            home_score += 1
            break

    return GameResult(home_score=home_score, away_score=away_score, innings=innings)


def simulate_n_box_score_games(
    home_team: TeamModel,
    away_team: TeamModel,
    home_starter: PitcherModel,
    away_starter: PitcherModel,
    park: ParkFactor,
    n_sims: int = config.DEFAULT_NUM_SIMS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run N simulations and return aggregate results with distributions."""
    if rng is None:
        rng = np.random.default_rng()

    home_scores = []
    away_scores = []
    home_wins = 0

    for _ in range(n_sims):
        result = simulate_box_score_game(
            home_team, away_team, home_starter, away_starter, park, rng
        )
        home_scores.append(result.home_score)
        away_scores.append(result.away_score)
        if result.home_win:
            home_wins += 1

    home_scores = np.array(home_scores)
    away_scores = np.array(away_scores)
    margins = home_scores - away_scores
    totals = home_scores + away_scores

    # Run line: P(home wins by 2+) = P(margin >= 2)
    home_cover_pct = float(np.mean(margins >= 2))

    # Over/under probabilities for common lines
    over_pct_by_line = {}
    for line in np.arange(6.0, 12.5, 0.5):
        over_pct_by_line[float(line)] = float(np.mean(totals > line))

    return {
        "home_win_pct": home_wins / n_sims,
        "home_score_mean": float(home_scores.mean()),
        "away_score_mean": float(away_scores.mean()),
        "home_score_std": float(home_scores.std()),
        "away_score_std": float(away_scores.std()),
        "total_mean": float(totals.mean()),
        "total_std": float(totals.std()),
        "spread_mean": float(margins.mean()),
        "home_cover_pct": home_cover_pct,
        "over_pct_by_line": over_pct_by_line,
        "n_sims": n_sims,
    }
