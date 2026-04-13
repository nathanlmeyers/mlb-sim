"""Schedule-based adjustments for MLB simulation.

Rest days, travel, day-after-night games, and fatigue effects.
Adapted from NBA sim patterns with MLB-specific values.
"""

from dataclasses import dataclass
import config


@dataclass
class ScheduleAdjustment:
    """Schedule-based run adjustment for a team."""
    total_adjustment: float = 0.0   # additive runs adjustment (positive = helps)
    components: dict = None         # breakdown of individual adjustments

    def __post_init__(self):
        if self.components is None:
            self.components = {}


def compute_schedule_adjustment(
    days_since_last_game: int = 1,
    is_day_after_night: bool = False,
    timezone_change: int = 0,
    games_in_last_7: int = 6,
    travel_direction: str | None = None,
) -> ScheduleAdjustment:
    """Compute schedule-based run adjustment for a team.

    Args:
        days_since_last_game: Days since team's last game (1 = normal)
        is_day_after_night: Day game following a night game
        timezone_change: Number of timezone changes since last series
                         (positive = eastward, negative = westward)
        games_in_last_7: Number of games played in last 7 days
        travel_direction: "east", "west", or None

    Returns:
        ScheduleAdjustment with total run adjustment
    """
    total = 0.0
    components = {}

    # Rest day boost (off-day before game)
    if days_since_last_game >= 2:
        total += config.SCHEDULE_REST_DAY_BOOST
        components["rest_day"] = config.SCHEDULE_REST_DAY_BOOST

    # Day-after-night penalty
    if is_day_after_night:
        total += config.SCHEDULE_DAY_AFTER_NIGHT
        components["day_after_night"] = config.SCHEDULE_DAY_AFTER_NIGHT

    # Travel/jet lag (eastward is worse)
    if timezone_change >= 2 or travel_direction == "east":
        total += config.SCHEDULE_EAST_TRAVEL_PENALTY
        components["east_travel"] = config.SCHEDULE_EAST_TRAVEL_PENALTY
    elif timezone_change <= -2 or travel_direction == "west":
        # Westward travel: minor effect
        adj = config.SCHEDULE_EAST_TRAVEL_PENALTY * 0.5
        total += adj
        components["west_travel"] = adj

    # Fatigue (many games in a short window)
    if games_in_last_7 > 6:
        total += config.SCHEDULE_FATIGUE_PENALTY
        components["fatigue"] = config.SCHEDULE_FATIGUE_PENALTY

    return ScheduleAdjustment(total_adjustment=total, components=components)
