"""Paper trading tracker for MLB predictions.

Logs what we WOULD have bet based on model edges vs Kalshi prices,
then tracks results after games complete. No real money involved.

Usage:
    python scripts/paper_trade.py log 2026-04-13     # log today's bets
    python scripts/paper_trade.py settle 2026-04-13   # settle after games finish
    python scripts/paper_trade.py report              # print P&L report
"""

import sys
import json
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")

import statsapi
import config

LEDGER_PATH = Path(".context/paper_ledger.json")
BANKROLL_START = 20.00  # starting bankroll


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {"bankroll": BANKROLL_START, "bets": [], "settled": [], "total_wagered": 0, "total_returned": 0}


def _save_ledger(ledger: dict):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def log_bets(target_date: str):
    """Log paper bets from today's pipeline output."""
    pipeline_path = Path(f".context/pipeline_{target_date.replace('-', '_')}.json")
    if not pipeline_path.exists():
        print(f"No pipeline output found for {target_date}. Run daily_pipeline.py first.")
        return

    with open(pipeline_path) as f:
        pipeline = json.load(f)

    edges = pipeline.get("edges", [])
    if not edges:
        print("No edges found — no bets to log.")
        return

    ledger = _load_ledger()
    bankroll = ledger["bankroll"]
    new_bets = []

    print(f"Paper Trading — {target_date}")
    print(f"Bankroll: ${bankroll:.2f}")
    print(f"{'='*70}")

    skipped_extreme = 0
    skipped_small = 0
    for e in edges:
        if e.get("edge", 0) < config.MIN_EDGE_THRESHOLD:
            skipped_small += 1
            continue  # skip small edges

        buy_price = e.get("kalshi", 0.50)
        # Defensive: skip extreme markets (don't bet vs strong conviction)
        if buy_price < config.MIN_MARKET_PRICE or buy_price > config.MAX_MARKET_PRICE:
            skipped_extreme += 1
            continue

        # Kelly sizing
        kelly_pct = min(0.03, e["edge"] * 0.125)  # eighth-Kelly, max 3%
        stake = round(bankroll * kelly_pct, 2)
        if stake < 0.01:
            continue
        shares = int(stake / buy_price) if buy_price > 0 else 0
        if shares < 1:
            shares = 1
            stake = round(buy_price, 2)

        actual_cost = round(shares * buy_price, 2)

        bet = {
            "date": target_date,
            "game": e["game"],
            "type": e["type"],
            "side": e["type"].split()[0],  # HOME or AWAY
            "model_prob": e["model"],
            "kalshi_price": buy_price,
            "edge": e["edge"],
            "shares": shares,
            "cost": actual_cost,
            "potential_payout": round(shares * 1.00, 2),  # Kalshi pays $1/share on win
            "status": "open",
            "result": None,
        }

        new_bets.append(bet)
        print(f"  BET: {e['type']:<20} {e['game']:<30} {shares} shares @ {buy_price:.0%} = ${actual_cost:.2f}")

    if new_bets:
        # Check for duplicate bets
        existing_keys = {(b["date"], b["game"], b["type"]) for b in ledger["bets"]}
        added = 0
        for bet in new_bets:
            key = (bet["date"], bet["game"], bet["type"])
            if key not in existing_keys:
                ledger["bets"].append(bet)
                ledger["total_wagered"] += bet["cost"]
                ledger["bankroll"] -= bet["cost"]
                added += 1

        _save_ledger(ledger)
        total_cost = sum(b["cost"] for b in new_bets)
        print(f"\n  Logged {added} new bets, total cost: ${total_cost:.2f}")
        print(f"  Remaining bankroll: ${ledger['bankroll']:.2f}")
    else:
        print("  No bets met the threshold.")

    if skipped_small or skipped_extreme:
        print(f"  Filtered: {skipped_small} small-edge, {skipped_extreme} extreme-market")


def settle_bets(target_date: str):
    """Settle open bets using actual game results."""
    ledger = _load_ledger()
    open_bets = [b for b in ledger["bets"] if b["date"] == target_date and b["status"] == "open"]

    if not open_bets:
        print(f"No open bets to settle for {target_date}")
        return

    # Fetch actual results
    print(f"Settling bets for {target_date}...")
    sched = statsapi.schedule(start_date=target_date, end_date=target_date)
    results = {}
    for g in sched:
        if g.get("status") != "Final":
            continue
        results[g["home_name"]] = {
            "home": g["home_name"],
            "away": g["away_name"],
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "home_win": g["home_score"] > g["away_score"],
        }
        results[g["away_name"]] = results[g["home_name"]]

    settled = 0
    total_returned = 0

    for bet in ledger["bets"]:
        if bet["date"] != target_date or bet["status"] != "open":
            continue

        # Parse game from bet
        game_str = bet["game"]
        # Try to match to results
        matched = None
        for team_name, result in results.items():
            if team_name[:8] in game_str:
                matched = result
                break

        if not matched:
            print(f"  Could not match: {game_str}")
            continue

        # Determine if bet won
        bet_type = bet["type"]
        won = False

        if "HOME ML" in bet_type:
            won = matched["home_win"]
        elif "AWAY ML" in bet_type:
            won = not matched["home_win"]
        elif "UNDER" in bet_type:
            line = float(bet_type.split()[-1])
            total = matched["home_score"] + matched["away_score"]
            won = total < line
        elif "OVER" in bet_type:
            line = float(bet_type.split()[-1])
            total = matched["home_score"] + matched["away_score"]
            won = total > line

        bet["status"] = "settled"
        bet["result"] = "win" if won else "loss"

        if won:
            returned = bet["potential_payout"]
            profit = returned - bet["cost"]
            ledger["bankroll"] += returned
            ledger["total_returned"] += returned
            total_returned += returned
            print(f"  WIN:  {bet['type']:<20} {game_str:<30} +${profit:.2f}")
        else:
            print(f"  LOSS: {bet['type']:<20} {game_str:<30} -${bet['cost']:.2f}")

        settled += 1

    _save_ledger(ledger)
    print(f"\n  Settled {settled} bets. Bankroll: ${ledger['bankroll']:.2f}")


def print_report():
    """Print full P&L report."""
    ledger = _load_ledger()
    bets = ledger["bets"]
    settled = [b for b in bets if b["status"] == "settled"]
    open_bets = [b for b in bets if b["status"] == "open"]

    wins = [b for b in settled if b["result"] == "win"]
    losses = [b for b in settled if b["result"] == "loss"]

    total_cost = sum(b["cost"] for b in settled)
    total_returned = sum(b["potential_payout"] for b in wins)
    profit = total_returned - total_cost
    roi = profit / total_cost * 100 if total_cost > 0 else 0

    print("=" * 70)
    print("PAPER TRADING REPORT")
    print("=" * 70)
    print(f"  Starting bankroll:  ${BANKROLL_START:.2f}")
    print(f"  Current bankroll:   ${ledger['bankroll']:.2f}")
    print(f"  P&L:                ${profit:+.2f}")
    print(f"  ROI:                {roi:+.1f}%")
    print()
    print(f"  Total bets:         {len(settled)} settled, {len(open_bets)} open")
    print(f"  Record:             {len(wins)}-{len(losses)}")
    print(f"  Win rate:           {len(wins)/len(settled)*100:.1f}%" if settled else "  Win rate:           N/A")
    print(f"  Total wagered:      ${total_cost:.2f}")
    print(f"  Total returned:     ${total_returned:.2f}")

    if settled:
        print()
        print("  HISTORY:")
        print(f"  {'Date':<12} {'Bet':<20} {'Game':<30} {'Cost':>6} {'Result':>8} {'P&L':>8}")
        print("  " + "-" * 85)
        for b in settled:
            pnl = b["potential_payout"] - b["cost"] if b["result"] == "win" else -b["cost"]
            print(f"  {b['date']:<12} {b['type']:<20} {b['game']:<30} ${b['cost']:>5.2f} {b['result']:>8} ${pnl:>+7.2f}")

    if open_bets:
        print()
        print("  OPEN BETS:")
        for b in open_bets:
            print(f"  {b['date']:<12} {b['type']:<20} {b['game']:<30} ${b['cost']:.2f} @ {b['kalshi_price']:.0%}")

    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/paper_trade.py [log|settle|report] [date]")
        return

    cmd = sys.argv[1]
    target_date = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()

    if cmd == "log":
        log_bets(target_date)
    elif cmd == "settle":
        settle_bets(target_date)
    elif cmd == "report":
        print_report()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
