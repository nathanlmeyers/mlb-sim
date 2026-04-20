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


# ---------------------------------------------------------------------------
# Kalshi fee-aware EV
# ---------------------------------------------------------------------------
#
# Kalshi's per-trade fee schedule (as of 2026-04, per public docs):
#
#   fee_per_contract = ceil(0.07 * count * price * (1 - price) * 100) cents
#
# i.e. 0.07 × N × p × (1-p), evaluated in dollars where N is contracts and
# p is the trade price (between 0 and 1). The fee is rounded UP to the next
# cent. Fees apply on the buy side; settlement of winning shares pays $1
# without additional fee. Losing shares pay $0 (fee already paid at buy).
#
# The fee is a parabola maxed at p=0.5 (≈$0.0175 per share) and falls to
# zero at extremes — sharp prices are nearly fee-free. This matters for our
# breakeven math: a 50% bet needs to win ~52% to break even after fee, while
# a 90% bet only needs ~90.2%.
#
# Source: https://help.kalshi.com/articles/4399  (verify before going live —
# Kalshi has changed fee schedules historically). If this rate changes the
# constant below should be the only edit needed.

KALSHI_FEE_RATE = 0.07  # multiplier on p*(1-p) per contract


def kalshi_fee_per_share(price: float) -> float:
    """Per-share fee in dollars at the given Kalshi buy price.

    Args:
        price: trade price as a probability in (0, 1)

    Returns: dollars per contract, rounded up to next cent.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    raw = KALSHI_FEE_RATE * price * (1 - price)
    # Round up to next cent
    cents = int(np.ceil(raw * 100))
    return cents / 100.0


def calculate_ev_kalshi(sim_prob: float, price: float) -> float:
    """Expected value per dollar staked on a Kalshi YES contract.

    EV = sim_prob × (1 - price - fee) + (1 - sim_prob) × (-price - fee)
       = sim_prob - price - fee_per_share

    Note we still divide by `price` to express EV per dollar staked (same
    convention as `calculate_ev`). Returns 0 if price is degenerate.

    For symmetric NO contracts, just call this with `1 - price` and
    `1 - sim_prob`.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    fee = kalshi_fee_per_share(price)
    profit_if_win = 1.0 - price - fee
    loss_if_lose = -price - fee
    ev_per_share = sim_prob * profit_if_win + (1 - sim_prob) * loss_if_lose
    # Normalize per dollar staked
    return ev_per_share / price


def kalshi_breakeven(price: float) -> float:
    """Win probability needed to break even on a Kalshi YES at the given price."""
    if not (0.0 < price < 1.0):
        return 0.5
    fee = kalshi_fee_per_share(price)
    # 0 = p*(1-price-fee) + (1-p)*(-price-fee)
    # 0 = p - p*price - p*fee - price - fee + p*price + p*fee
    # 0 = p - price - fee
    return price + fee
