"""Weather adjustments for MLB simulation.

Temperature, wind, and humidity effects on run scoring.
Research: +1.96% HR per 1°C above 20°C baseline.
"""

from dataclasses import dataclass
import config


@dataclass
class WeatherAdjustment:
    """Weather-based adjustments to simulation parameters."""
    hr_multiplier: float = 1.0      # multiplied into park HR factor
    total_adjustment: float = 0.0   # additive runs adjustment
    description: str = ""


def compute_weather_adjustment(
    temperature_c: float | None = None,
    wind_speed_mph: float | None = None,
    wind_direction: str | None = None,
    is_dome: bool = False,
) -> WeatherAdjustment:
    """Compute weather adjustments for a game.

    Args:
        temperature_c: Game-time temperature in Celsius
        wind_speed_mph: Wind speed in mph
        wind_direction: "out" (to CF), "in" (from CF), "cross", or None
        is_dome: Whether venue has a closed roof

    Returns:
        WeatherAdjustment with HR multiplier and total run adjustment
    """
    if is_dome:
        return WeatherAdjustment(
            hr_multiplier=1.0,
            total_adjustment=0.0,
            description="Dome/closed roof: no weather effect",
        )

    hr_mult = 1.0
    total_adj = 0.0
    parts = []

    # Temperature effect
    if temperature_c is not None:
        temp_diff = temperature_c - config.WEATHER_BASELINE_TEMP_C
        hr_pct_change = temp_diff * config.WEATHER_HR_PCT_PER_DEGREE
        hr_mult *= (1.0 + hr_pct_change)

        # Rough total adjustment: each 5°C adds ~0.2 runs
        total_adj += temp_diff * 0.04
        if abs(temp_diff) >= 5:
            parts.append(f"Temp {temperature_c:.0f}°C ({temp_diff:+.0f}° from baseline)")

    # Wind effect
    if wind_speed_mph is not None and wind_speed_mph >= 8 and wind_direction:
        if wind_direction == "out":
            # Wind blowing out to CF: more HR, more runs
            wind_factor = min(wind_speed_mph / 10.0, 2.0)
            hr_mult *= (1.0 + 0.08 * wind_factor)
            total_adj += config.WEATHER_WIND_OUT_ADJUSTMENT * wind_factor
            parts.append(f"Wind out {wind_speed_mph:.0f}mph")
        elif wind_direction == "in":
            # Wind blowing in: fewer HR, fewer runs
            wind_factor = min(wind_speed_mph / 10.0, 2.0)
            hr_mult *= (1.0 - 0.06 * wind_factor)
            total_adj += config.WEATHER_WIND_IN_ADJUSTMENT * wind_factor
            parts.append(f"Wind in {wind_speed_mph:.0f}mph")

    return WeatherAdjustment(
        hr_multiplier=hr_mult,
        total_adjustment=total_adj,
        description=", ".join(parts) if parts else "Normal conditions",
    )
