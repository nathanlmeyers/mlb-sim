"""Inning simulation for MLB engine.

Simulates a half-inning by running plate appearances through the
batting order until 3 outs are recorded. Tracks baserunner state,
TTO (Times Through the Order), and actual pitch counts from the
count-progression PA model.
"""

import numpy as np
from models.batter_model import BatterModel
from models.pitcher_model import PitcherModel, get_fatigue_multiplier
from models.team_model import ParkFactor
from sim.baserunners import BaseState
from sim.plate_appearance import simulate_plate_appearance, PAOutcome
import config


def simulate_half_inning(
    lineup: list[BatterModel],
    lineup_pos: int,
    pitcher: PitcherModel,
    park: ParkFactor,
    pitcher_pitch_count: int = 0,
    times_through_order: int = 1,
    is_home: bool = False,
    manfred_runner_id: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int, int]:
    """Simulate one half-inning.

    Args:
        lineup: 9-batter lineup (list of BatterModel)
        lineup_pos: current position in batting order (0-8)
        pitcher: opposing pitcher model
        park: park factor for venue
        pitcher_pitch_count: pitcher's current pitch count
        times_through_order: how many times this lineup has cycled (1-based)
        is_home: whether the batting team is the home team
        manfred_runner_id: if set, place this runner on 2nd (extra innings)
        rng: numpy random generator

    Returns:
        (runs_scored, new_lineup_pos, pitches_thrown, batters_faced)
    """
    if rng is None:
        rng = np.random.default_rng()

    outs = 0
    runs = 0
    bases = BaseState()
    pitches_this_inning = 0
    batters_this_inning = 0
    current_pos = lineup_pos

    if manfred_runner_id is not None and config.MANFRED_RUNNER:
        bases.place_manfred_runner(manfred_runner_id)

    while outs < 3:
        batter = lineup[current_pos % 9]
        total_pitches = pitcher_pitch_count + pitches_this_inning
        fatigue = get_fatigue_multiplier(total_pitches)

        result = simulate_plate_appearance(
            batter=batter, pitcher=pitcher, park=park,
            fatigue_mult=fatigue,
            times_through_order=times_through_order,
            is_home=is_home, rng=rng,
        )

        # Use actual pitch count from count-progression model
        pitches_this_inning += result.pitch_count
        batters_this_inning += 1

        if result.is_out:
            if result.outcome == PAOutcome.GROUNDOUT:
                go_runs, go_outs = bases.advance_on_groundout(outs, rng)
                runs += go_runs
                outs += go_outs
            elif result.outcome == PAOutcome.FLYOUT:
                fo_runs, fo_outs = bases.advance_on_flyout(outs, rng)
                runs += fo_runs
                outs += fo_outs
            else:
                outs += 1
        else:
            if result.outcome in (PAOutcome.WALK, PAOutcome.HIT_BY_PITCH):
                runs += bases.advance_on_walk(batter.player_id)
            elif result.outcome == PAOutcome.SINGLE:
                runs += bases.advance_on_single(batter.player_id, batter.speed_score, rng)
            elif result.outcome == PAOutcome.DOUBLE:
                runs += bases.advance_on_double(batter.player_id, batter.speed_score, rng)
            elif result.outcome == PAOutcome.TRIPLE:
                runs += bases.advance_on_triple(batter.player_id)
            elif result.outcome == PAOutcome.HOME_RUN:
                runs += bases.advance_on_hr()

        current_pos = (current_pos + 1) % 9

    return runs, current_pos, pitches_this_inning, batters_this_inning
