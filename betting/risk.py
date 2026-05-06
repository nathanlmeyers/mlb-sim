"""Risk controls / circuit breakers for live MLB betting.

Pre-built per Phase F1 of the remediation plan. Live trading is NOT yet
enabled (paper_trade.py has no `live` subcommand), but the gate logic
needs to exist before that subcommand is added so the policy is not
half-built when money is on the line.

The rules below are encoded as data, then evaluated by `evaluate_risk()`
against the current paper-ledger state. They are intentionally strict;
loosening them requires a code change, not a config flag.

Usage:
    from betting.risk import evaluate_risk, LIMITS
    decision = evaluate_risk(ledger)
    if not decision.permitted:
        print(decision.reason)
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Limits — locked here, not in config.py, so they cannot be silently raised.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskLimits:
    # Per-bet
    max_bet_first_50_dollars: float = 2.00      # cap stake until 50 live bets done
    max_bet_after_50_pct: float = 0.03          # max 3% bankroll per bet thereafter

    # Per-day
    daily_loss_cap_pct: float = 0.10            # stop at 10% intraday drawdown
    max_bets_per_day: int = 5                   # quality-weight the day
    max_daily_exposure_pct: float = 0.15        # cap total open stake / bankroll

    # Streaks / drawdowns
    consecutive_loss_halt: int = 4              # pause after 4 losses in a row
    rolling_clv_window: int = 20                # check rolling CLV over last N bets
    rolling_clv_min: float = 0.0                # halt if rolling CLV goes negative
    rolling_roi_window: int = 50                # check rolling ROI over last N bets
    rolling_roi_min: float = -0.10              # halt if 50-bet rolling ROI < -10%

    # Bankroll floor
    bankroll_floor_pct: float = 0.80            # full halt at 80% of starting bankroll

    # Live-trading entry gates (must pass walk-forward gate first)
    min_paper_bets_before_live: int = 50
    min_paper_clv_before_live: float = 0.01     # +1% rolling CLV on 30 bets

LIMITS = RiskLimits()


@dataclass
class RiskDecision:
    permitted: bool
    reason: str
    suggested_max_stake: float | None = None    # if permitted, the cap to apply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settled_bets(ledger: dict) -> list[dict]:
    return [b for b in ledger.get("bets", []) if b.get("status") == "settled"]


def _open_bets(ledger: dict) -> list[dict]:
    return [b for b in ledger.get("bets", []) if b.get("status") == "open"]


def _bets_today(ledger: dict, today: str) -> list[dict]:
    return [b for b in ledger.get("bets", []) if b.get("date") == today]


def _live_bets(ledger: dict) -> list[dict]:
    """Bets placed via the live subcommand (marked with mode='live')."""
    return [b for b in ledger.get("bets", []) if b.get("mode") == "live"]


def _consecutive_losses(settled: list[dict]) -> int:
    streak = 0
    for b in reversed(sorted(settled, key=lambda x: (x.get("date", ""), x.get("game", "")))):
        if b.get("result") == "loss":
            streak += 1
        else:
            break
    return streak


def _rolling_metric(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    take = values[-window:]
    return sum(take) / len(take) if take else 0.0


# ---------------------------------------------------------------------------
# Decision function
# ---------------------------------------------------------------------------

def evaluate_risk(
    ledger: dict,
    today: str,
    starting_bankroll: float,
    proposed_stake: float,
    bet_clvs: list[float] | None = None,
    bet_rois: list[float] | None = None,
    is_live: bool = True,
    limits: RiskLimits = LIMITS,
) -> RiskDecision:
    """Evaluate whether `proposed_stake` is permitted given current ledger state.

    Args:
        ledger: paper ledger dict (same shape as paper_ledger.json)
        today: ISO date for the current trading day
        starting_bankroll: initial bankroll (denominator for floor checks)
        proposed_stake: dollars about to be wagered
        bet_clvs: per-bet CLV values, oldest→newest; for rolling CLV check
        bet_rois: per-bet ROI values, oldest→newest; for rolling ROI check
        is_live: whether this is a real-money bet (paper bets bypass most checks)
        limits: limits dataclass (defaults to module LIMITS)
    """
    bankroll = ledger.get("bankroll", starting_bankroll)
    live = _live_bets(ledger)
    n_live = len(live)

    # ---- bankroll floor ----
    if bankroll < starting_bankroll * limits.bankroll_floor_pct:
        return RiskDecision(False,
            f"Bankroll ${bankroll:.2f} below {limits.bankroll_floor_pct:.0%} floor "
            f"of starting ${starting_bankroll:.2f}. Manual review required.")

    if not is_live:
        # Paper bets only require the bankroll floor; everything else is
        # for real money.
        cap = bankroll * limits.max_bet_after_50_pct
        return RiskDecision(True, "paper bet permitted",
                            suggested_max_stake=min(proposed_stake, cap))

    # ---- live-trading entry gate ----
    if n_live == 0:
        # First live bet ever. Check paper-validation prerequisites.
        n_paper = sum(1 for b in _settled_bets(ledger) if b.get("mode") != "live")
        if n_paper < limits.min_paper_bets_before_live:
            return RiskDecision(False,
                f"Cannot start live trading: only {n_paper} paper bets settled "
                f"(need ≥ {limits.min_paper_bets_before_live}).")
        if bet_clvs and len(bet_clvs) >= 30:
            recent_clv = _rolling_metric(bet_clvs, 30)
            if recent_clv < limits.min_paper_clv_before_live:
                return RiskDecision(False,
                    f"Cannot start live trading: 30-bet paper CLV "
                    f"{recent_clv:+.1%} < required {limits.min_paper_clv_before_live:+.1%}.")

    # ---- consecutive-loss streak halt ----
    streak = _consecutive_losses(_settled_bets(ledger))
    if streak >= limits.consecutive_loss_halt:
        return RiskDecision(False,
            f"Auto-halt: {streak} consecutive losses (limit {limits.consecutive_loss_halt}). "
            f"Manual restart required after review.")

    # ---- rolling CLV halt (live only) ----
    if bet_clvs and len(bet_clvs) >= limits.rolling_clv_window:
        recent_clv = _rolling_metric(bet_clvs, limits.rolling_clv_window)
        if recent_clv < limits.rolling_clv_min:
            return RiskDecision(False,
                f"Auto-halt: rolling {limits.rolling_clv_window}-bet CLV "
                f"{recent_clv:+.1%} < {limits.rolling_clv_min:+.1%}. "
                f"Stop placing live bets and re-evaluate config.")

    # ---- rolling ROI halt ----
    if bet_rois and len(bet_rois) >= limits.rolling_roi_window:
        recent_roi = _rolling_metric(bet_rois, limits.rolling_roi_window)
        if recent_roi < limits.rolling_roi_min:
            return RiskDecision(False,
                f"Auto-halt: rolling {limits.rolling_roi_window}-bet ROI "
                f"{recent_roi:+.1%} < {limits.rolling_roi_min:+.1%}.")

    # ---- daily loss cap ----
    today_bets = [b for b in _settled_bets(ledger) if b.get("date") == today]
    today_loss = sum(
        (b.get("cost", 0.0) if b.get("result") == "loss"
         else -(b.get("potential_payout", 0.0) - b.get("cost", 0.0)))
        for b in today_bets
    )
    if today_loss > starting_bankroll * limits.daily_loss_cap_pct:
        return RiskDecision(False,
            f"Daily loss cap hit: ${today_loss:.2f} > {limits.daily_loss_cap_pct:.0%} "
            f"of bankroll. Resume tomorrow.")

    # ---- max bets per day ----
    n_today = len(_bets_today(ledger, today))
    if n_today >= limits.max_bets_per_day:
        return RiskDecision(False,
            f"Max bets per day hit ({n_today}/{limits.max_bets_per_day}).")

    # ---- daily exposure ----
    open_today_exposure = sum(b.get("cost", 0.0) for b in _open_bets(ledger)
                               if b.get("date") == today)
    if (open_today_exposure + proposed_stake) > bankroll * limits.max_daily_exposure_pct:
        return RiskDecision(False,
            f"Daily exposure cap: open ${open_today_exposure:.2f} + new ${proposed_stake:.2f} "
            f"would exceed {limits.max_daily_exposure_pct:.0%} of bankroll.")

    # ---- per-bet stake cap ----
    if n_live < 50:
        cap = limits.max_bet_first_50_dollars
        if proposed_stake > cap:
            return RiskDecision(True,
                f"Stake clipped to ${cap:.2f} (first {50 - n_live} live bets "
                f"capped per Phase F2 of remediation plan).",
                suggested_max_stake=cap)
        return RiskDecision(True, "permitted (within first-50 cap)",
                            suggested_max_stake=proposed_stake)

    cap = bankroll * limits.max_bet_after_50_pct
    if proposed_stake > cap:
        return RiskDecision(True,
            f"Stake clipped to ${cap:.2f} ({limits.max_bet_after_50_pct:.0%} of bankroll).",
            suggested_max_stake=cap)

    return RiskDecision(True, "permitted", suggested_max_stake=proposed_stake)
