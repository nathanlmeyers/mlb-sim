"""Detailed game simulation with TTO tracking and count-progression PAs.

Simulates a full game inning-by-inning using the count-progression
plate appearance model. Tracks pitcher fatigue via actual pitch counts,
Times Through the Order (TTO) penalty, bullpen transitions, and the
Manfred runner rule in extra innings.
"""

from dataclasses import dataclass, field
import numpy as np
from models.batter_model import BatterModel
from models.pitcher_model import PitcherModel
from models.team_model import ParkFactor
from sim.inning import simulate_half_inning
import config


@dataclass
class DetailedGameResult:
    home_score: int
    away_score: int
    innings: int
    home_hits: int = 0
    away_hits: int = 0
    home_score_by_inning: list[int] = field(default_factory=list)
    away_score_by_inning: list[int] = field(default_factory=list)

    @property
    def home_win(self) -> bool:
        return self.home_score > self.away_score


@dataclass
class PitcherState:
    """Tracks a pitcher's state during a game."""
    model: PitcherModel
    pitch_count: int = 0
    outs_recorded: int = 0
    runs_allowed: int = 0
    batters_faced: int = 0
    is_active: bool = True

    @property
    def times_through_order(self) -> int:
        """How many times through the batting order (1-based)."""
        return 1 + self.batters_faced // 9


def _should_pull_starter(
    state: PitcherState,
    inning: int,
    rng: np.random.Generator,
) -> bool:
    """Decide whether to pull the starting pitcher.

    TTO-aware: pitchers entering 3rd TTO face much higher pull probability
    unless they're aces (high K%).
    """
    # Hard limit
    if state.pitch_count >= 110:
        return True

    # TTO-aware: 3rd time through is dangerous for non-aces
    tto = state.times_through_order
    if tto >= 3 and state.model.k_pct < 0.28:
        # Non-ace facing 3rd TTO: high pull probability
        pull_prob = 0.40 + (state.pitch_count - 70) * 0.02 if state.pitch_count > 70 else 0.30
        return rng.random() < pull_prob

    # Soft threshold: probability increases per pitch past threshold
    if state.pitch_count >= config.STARTER_PITCH_COUNT_THRESHOLD:
        pull_prob = (state.pitch_count - config.STARTER_PITCH_COUNT_THRESHOLD) * 0.05
        return rng.random() < pull_prob

    # Early pull if getting shelled
    if inning <= 5 and state.runs_allowed >= 5:
        return True

    return False


def _select_reliever(
    bullpen: list[PitcherState],
    inning: int,
    score_diff: int,
) -> PitcherState | None:
    """Select the next reliever from the bullpen."""
    available = [p for p in bullpen if p.is_active and p.outs_recorded == 0]

    if not available:
        available = [p for p in bullpen if p.is_active]

    if not available:
        return None

    # Save situation (9th inning, leading by 1-3): use closer
    if inning >= 9 and 0 < score_diff <= 3:
        closers = [p for p in available if p.model.is_closer]
        if closers:
            return closers[0]

    # Otherwise: best available by K% - BB%
    available.sort(key=lambda p: p.model.k_pct - p.model.bb_pct, reverse=True)
    return available[0]


def simulate_detailed_game(
    home_lineup: list[BatterModel],
    away_lineup: list[BatterModel],
    home_starter: PitcherModel,
    away_starter: PitcherModel,
    home_bullpen: list[PitcherModel],
    away_bullpen: list[PitcherModel],
    park: ParkFactor,
    rng: np.random.Generator | None = None,
) -> DetailedGameResult:
    """Simulate a full MLB game with TTO tracking and count-progression PAs."""
    if rng is None:
        rng = np.random.default_rng()

    home_pitcher_state = PitcherState(model=home_starter)
    away_pitcher_state = PitcherState(model=away_starter)
    home_bp_states = [PitcherState(model=p) for p in home_bullpen]
    away_bp_states = [PitcherState(model=p) for p in away_bullpen]

    home_score = 0
    away_score = 0
    home_lineup_pos = 0
    away_lineup_pos = 0
    home_score_by_inning = []
    away_score_by_inning = []

    inning = 1
    max_innings = 18

    while inning <= max_innings:
        is_extra = inning > 9

        # --- TOP OF INNING (away team bats, home team pitches) ---
        if home_pitcher_state.model.is_starter and inning > 1:
            if _should_pull_starter(home_pitcher_state, inning, rng):
                home_pitcher_state.is_active = False
                new_pitcher = _select_reliever(
                    home_bp_states, inning, home_score - away_score
                )
                if new_pitcher:
                    home_pitcher_state = new_pitcher
        elif not home_pitcher_state.model.is_starter:
            if home_pitcher_state.outs_recorded >= config.RELIEVER_MAX_OUTS:
                home_pitcher_state.is_active = False
                new_pitcher = _select_reliever(
                    home_bp_states, inning, home_score - away_score
                )
                if new_pitcher:
                    home_pitcher_state = new_pitcher

        manfred_id = None
        if is_extra and config.MANFRED_RUNNER:
            prev_pos = (away_lineup_pos - 1) % 9
            manfred_id = away_lineup[prev_pos].player_id

        top_runs, away_lineup_pos, pitches, batters = simulate_half_inning(
            lineup=away_lineup,
            lineup_pos=away_lineup_pos,
            pitcher=home_pitcher_state.model,
            park=park,
            pitcher_pitch_count=home_pitcher_state.pitch_count,
            times_through_order=home_pitcher_state.times_through_order,
            is_home=False,
            manfred_runner_id=manfred_id,
            rng=rng,
        )
        away_score += top_runs
        home_pitcher_state.pitch_count += pitches
        home_pitcher_state.outs_recorded += 3
        home_pitcher_state.runs_allowed += top_runs
        home_pitcher_state.batters_faced += batters
        away_score_by_inning.append(top_runs)

        # --- BOTTOM OF INNING (home team bats, away team pitches) ---
        if inning >= 9 and home_score > away_score:
            home_score_by_inning.append(0)
            break

        if away_pitcher_state.model.is_starter and inning > 1:
            if _should_pull_starter(away_pitcher_state, inning, rng):
                away_pitcher_state.is_active = False
                new_pitcher = _select_reliever(
                    away_bp_states, inning, away_score - home_score
                )
                if new_pitcher:
                    away_pitcher_state = new_pitcher
        elif not away_pitcher_state.model.is_starter:
            if away_pitcher_state.outs_recorded >= config.RELIEVER_MAX_OUTS:
                away_pitcher_state.is_active = False
                new_pitcher = _select_reliever(
                    away_bp_states, inning, away_score - home_score
                )
                if new_pitcher:
                    away_pitcher_state = new_pitcher

        manfred_id = None
        if is_extra and config.MANFRED_RUNNER:
            prev_pos = (home_lineup_pos - 1) % 9
            manfred_id = home_lineup[prev_pos].player_id

        bot_runs, home_lineup_pos, pitches, batters = simulate_half_inning(
            lineup=home_lineup,
            lineup_pos=home_lineup_pos,
            pitcher=away_pitcher_state.model,
            park=park,
            pitcher_pitch_count=away_pitcher_state.pitch_count,
            times_through_order=away_pitcher_state.times_through_order,
            is_home=True,
            manfred_runner_id=manfred_id,
            rng=rng,
        )
        home_score += bot_runs
        away_pitcher_state.pitch_count += pitches
        away_pitcher_state.outs_recorded += 3
        away_pitcher_state.runs_allowed += bot_runs
        away_pitcher_state.batters_faced += batters
        home_score_by_inning.append(bot_runs)

        if inning >= 9 and home_score > away_score:
            break
        if inning >= 9 and home_score != away_score:
            break

        inning += 1

    if home_score == away_score:
        home_score += 1

    return DetailedGameResult(
        home_score=home_score,
        away_score=away_score,
        innings=inning,
        home_score_by_inning=home_score_by_inning,
        away_score_by_inning=away_score_by_inning,
    )


def simulate_n_detailed_games(
    home_lineup: list[BatterModel],
    away_lineup: list[BatterModel],
    home_starter: PitcherModel,
    away_starter: PitcherModel,
    home_bullpen: list[PitcherModel],
    away_bullpen: list[PitcherModel],
    park: ParkFactor,
    n_sims: int = config.DEFAULT_NUM_SIMS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run N detailed simulations and return aggregate results."""
    if rng is None:
        rng = np.random.default_rng()

    home_scores = []
    away_scores = []
    home_wins = 0

    for _ in range(n_sims):
        result = simulate_detailed_game(
            home_lineup, away_lineup,
            home_starter, away_starter,
            home_bullpen, away_bullpen,
            park, rng,
        )
        home_scores.append(result.home_score)
        away_scores.append(result.away_score)
        if result.home_win:
            home_wins += 1

    home_scores = np.array(home_scores)
    away_scores = np.array(away_scores)
    margins = home_scores - away_scores
    totals = home_scores + away_scores

    home_cover_pct = float(np.mean(margins >= 2))

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
