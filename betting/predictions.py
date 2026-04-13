"""Betting prediction dataclass for MLB simulation engine.

Central structure for all bet types: moneyline, run line, over/under.
"""

from dataclasses import dataclass, field
from betting.ev import (
    calculate_ev_american, fractional_kelly, american_to_decimal,
    win_probability_to_moneyline, format_odds, remove_vig,
)
import config


@dataclass
class BettingPrediction:
    """Complete betting prediction for an MLB game."""

    # Moneyline
    home_win_prob: float
    away_win_prob: float

    # Run line (-1.5 / +1.5)
    home_cover_prob: float  # P(home wins by 2+)
    away_cover_prob: float  # P(away wins or loses by 0-1)

    # Over/Under
    total_mean: float
    total_std: float
    over_prob: dict[float, float] = field(default_factory=dict)  # line -> P(over)

    # Confidence
    confidence: float = 1.0
    engine_agreement: float = 1.0

    # Score predictions
    home_score_mean: float = 0.0
    away_score_mean: float = 0.0

    def evaluate_moneyline(self, home_ml: int, away_ml: int) -> dict:
        """Evaluate moneyline bet EV against market odds."""
        true_home, true_away = remove_vig(home_ml, away_ml)

        home_ev = calculate_ev_american(self.home_win_prob, home_ml)
        away_ev = calculate_ev_american(self.away_win_prob, away_ml)

        home_kelly = fractional_kelly(
            self.home_win_prob, american_to_decimal(home_ml)
        ) * self.confidence
        away_kelly = fractional_kelly(
            self.away_win_prob, american_to_decimal(away_ml)
        ) * self.confidence

        return {
            "home_ev": home_ev,
            "away_ev": away_ev,
            "home_kelly": home_kelly,
            "away_kelly": away_kelly,
            "home_edge": self.home_win_prob - true_home,
            "away_edge": self.away_win_prob - true_away,
            "best_bet": "home" if home_ev > away_ev else "away",
            "best_ev": max(home_ev, away_ev),
            "has_edge": max(home_ev, away_ev) >= config.MIN_BET_EV,
        }

    def evaluate_run_line(self, home_rl_odds: int = -110, away_rl_odds: int = -110) -> dict:
        """Evaluate -1.5 run line bet."""
        home_ev = calculate_ev_american(self.home_cover_prob, home_rl_odds)
        away_ev = calculate_ev_american(self.away_cover_prob, away_rl_odds)

        return {
            "home_cover_ev": home_ev,
            "away_cover_ev": away_ev,
            "home_cover_prob": self.home_cover_prob,
            "away_cover_prob": self.away_cover_prob,
            "best_bet": "home -1.5" if home_ev > away_ev else "away +1.5",
            "best_ev": max(home_ev, away_ev),
            "has_edge": max(home_ev, away_ev) >= config.MIN_BET_EV,
        }

    def evaluate_total(self, line: float, over_odds: int = -110, under_odds: int = -110) -> dict:
        """Evaluate over/under bet at a specific line."""
        over_prob = self.over_prob.get(line, 0.5)
        under_prob = 1.0 - over_prob

        over_ev = calculate_ev_american(over_prob, over_odds)
        under_ev = calculate_ev_american(under_prob, under_odds)

        return {
            "line": line,
            "over_prob": over_prob,
            "under_prob": under_prob,
            "over_ev": over_ev,
            "under_ev": under_ev,
            "best_bet": f"over {line}" if over_ev > under_ev else f"under {line}",
            "best_ev": max(over_ev, under_ev),
            "has_edge": max(over_ev, under_ev) >= config.MIN_BET_EV,
        }

    def summary(self) -> str:
        """Generate a text summary of the prediction."""
        fair_home = win_probability_to_moneyline(self.home_win_prob)
        fair_away = win_probability_to_moneyline(self.away_win_prob)

        lines = [
            f"Moneyline:  Home {self.home_win_prob:.1%} ({format_odds(fair_home)})  |  Away {self.away_win_prob:.1%} ({format_odds(fair_away)})",
            f"Run Line:   Home -1.5 {self.home_cover_prob:.1%}  |  Away +1.5 {self.away_cover_prob:.1%}",
            f"Total:      {self.total_mean:.1f} (σ={self.total_std:.1f})",
            f"Scores:     Home {self.home_score_mean:.1f}  |  Away {self.away_score_mean:.1f}",
            f"Confidence: {self.confidence:.0%}",
        ]

        # Add O/U probs for nearest half-run lines
        nearest_line = round(self.total_mean * 2) / 2  # round to nearest 0.5
        if nearest_line in self.over_prob:
            lines.append(f"O/U {nearest_line}:   Over {self.over_prob[nearest_line]:.1%}")

        return "\n".join(lines)
