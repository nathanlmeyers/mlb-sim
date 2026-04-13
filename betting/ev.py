"""Expected value calculations, odds conversions, and Kelly criterion.

Core betting math for the MLB simulation engine.
"""

import numpy as np
import config


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))


def implied_probability(american: int) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if american < 0:
        return abs(american) / (abs(american) + 100)
    else:
        return 100 / (american + 100)


def remove_vig(home_ml: int, away_ml: int) -> tuple[float, float]:
    """Remove vigorish from a moneyline pair to get true probabilities.

    Returns (true_home_prob, true_away_prob) summing to 1.0.
    """
    raw_home = implied_probability(home_ml)
    raw_away = implied_probability(away_ml)
    total = raw_home + raw_away  # > 1.0 due to vig

    if total <= 0:
        return 0.5, 0.5

    return raw_home / total, raw_away / total


def calculate_ev(sim_prob: float, decimal_odds: float) -> float:
    """Calculate expected value per dollar wagered.

    EV = sim_prob * (decimal_odds - 1) - (1 - sim_prob)

    Positive EV means profitable bet in the long run.
    """
    return sim_prob * (decimal_odds - 1) - (1 - sim_prob)


def calculate_ev_american(sim_prob: float, american_odds: int) -> float:
    """Calculate EV from American odds."""
    return calculate_ev(sim_prob, american_to_decimal(american_odds))


def fractional_kelly(
    sim_prob: float,
    decimal_odds: float,
    fraction: float = config.KELLY_FRACTION,
    max_bet: float = config.MAX_BET_PCT,
) -> float:
    """Calculate fractional Kelly criterion bet size.

    Returns fraction of bankroll to wager (0 if no edge).

    Args:
        sim_prob: Model's estimated win probability
        decimal_odds: Decimal odds for the bet
        fraction: Kelly fraction (0.125 = eighth-Kelly)
        max_bet: Maximum bet as fraction of bankroll
    """
    b = decimal_odds - 1  # net odds (profit per dollar wagered)
    q = 1 - sim_prob

    if b <= 0 or sim_prob <= 0:
        return 0.0

    # Full Kelly: f* = (bp - q) / b
    kelly_full = (b * sim_prob - q) / b

    if kelly_full <= 0:
        return 0.0  # no edge

    bet_size = kelly_full * fraction
    return min(bet_size, max_bet)


def win_probability_to_moneyline(p: float) -> int:
    """Convert win probability to American moneyline odds."""
    p = np.clip(p, 0.01, 0.99)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    else:
        return int(round(100 * (1 - p) / p))


def format_odds(american: int) -> str:
    """Format American odds with sign."""
    return f"+{american}" if american > 0 else str(american)


def breakeven_win_rate(american: int) -> float:
    """Calculate the win rate needed to break even at given odds."""
    return implied_probability(american)
