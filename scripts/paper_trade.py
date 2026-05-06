"""Paper trading tracker for MLB predictions.

Logs what we WOULD have bet based on model edges vs Kalshi prices,
then tracks results after games complete. No real money involved.

Usage:
    python scripts/paper_trade.py log 2026-04-13     # log today's bets
    python scripts/paper_trade.py settle 2026-04-13   # settle after games finish
    python scripts/paper_trade.py report              # print P&L report
    python scripts/paper_trade.py clv                 # CLV vs DK closing lines

Why CLV matters
---------------
Settled P&L on tiny samples is the worst signal in betting — variance
swamps signal until ~100+ bets. CLV (Closing Line Value) tells us whether
the price we got beat the closing market price. Positive CLV across 30+
bets is strong evidence of edge; negative settled P&L with positive CLV
means "the model is right, we got unlucky." Always trust CLV first.
"""

import sys
import json
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")

import statsapi
import config
from betting.ev import remove_vig

LEDGER_PATH = Path(".context/paper_ledger.json")
ESPN_ODDS_PATH = Path(".context/espn_odds_2026.json")
BANKROLL_START = 20.00  # starting bankroll


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {"bankroll": BANKROLL_START, "bets": [], "settled": [], "total_wagered": 0, "total_returned": 0}


def _save_ledger(ledger: dict):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


# ---------------------------------------------------------------------------
# CLV (Closing Line Value) — vs DraftKings closing line in espn_odds_2026.json
# ---------------------------------------------------------------------------

def _load_espn_odds_index() -> dict:
    """Index ESPN closing odds by date → list of (full_home, full_away, dk_ml_home, dk_ml_away)."""
    if not ESPN_ODDS_PATH.exists():
        return {}
    with open(ESPN_ODDS_PATH) as f:
        rows = json.load(f)
    by_date: dict[str, list] = {}
    for r in rows:
        d = r["date"]
        odds = r.get("odds") or {}
        ml_h = odds.get("ml_home")
        ml_a = odds.get("ml_away")
        if ml_h is None or ml_a is None:
            continue
        by_date.setdefault(d, []).append({
            "home_name": r.get("home_name", ""),
            "away_name": r.get("away_name", ""),
            "ml_home": ml_h,
            "ml_away": ml_a,
            "total_close": odds.get("total_close"),
        })
    return by_date


def _match_paper_to_espn(bet: dict, by_date: dict) -> dict | None:
    """Match a paper bet to its ESPN row by date + truncated team names.

    Paper bets store games as e.g. "Washington N @ Pittsburgh P" (12-char prefix).
    We compare against the full names from ESPN by checking the prefix.
    """
    candidates = by_date.get(bet["date"], [])
    if not candidates:
        return None
    # Format used by paper bets: "<away first 12> @ <home first 12>"
    game = bet.get("game", "")
    if "@" not in game:
        return None
    away_part, home_part = [s.strip() for s in game.split("@", 1)]
    for row in candidates:
        if (row["away_name"][:len(away_part)] == away_part
                and row["home_name"][:len(home_part)] == home_part):
            return row
    return None


def _compute_bet_clv(bet: dict, espn_row: dict) -> dict | None:
    """Compute CLV for a paper bet vs the DK closing line.

    Definitions:
      market_p   = de-vigged DK closing probability for the side we bet
      kalshi_p   = the price we paid on Kalshi (== implied prob on Kalshi)
      model_p    = our model's estimated probability for that side
      clv_vs_dk  = market_p - kalshi_p   ← did the closing market move toward us?
                   Positive: the price we got was better than where DK closed.
      model_edge_vs_dk = model_p - market_p
                   Positive: the model still disagreed with DK after their close.

    CLV vs the closing line is the canonical "did we get a good number?" metric.
    """
    side = bet.get("side")  # "HOME" or "AWAY"
    if side not in ("HOME", "AWAY"):
        return None
    if "OVER" in bet.get("type", "") or "UNDER" in bet.get("type", ""):
        # Totals CLV needs total_close + total_market — different math, skip for now.
        return None
    ml_home = espn_row["ml_home"]
    ml_away = espn_row["ml_away"]
    true_h, true_a = remove_vig(ml_home, ml_away)
    market_p = true_h if side == "HOME" else true_a
    kalshi_p = bet.get("kalshi_price")
    model_p = bet.get("model_prob")
    if kalshi_p is None or model_p is None:
        return None
    return {
        "market_p": market_p,
        "kalshi_p": kalshi_p,
        "model_p": model_p,
        "clv_vs_dk": market_p - kalshi_p,
        "model_edge_vs_dk": model_p - market_p,
    }


def _summarize_clv(bet_clv_pairs: list[tuple[dict, dict]]) -> dict:
    """Return summary stats for a list of (bet, clv_dict) pairs."""
    if not bet_clv_pairs:
        return {"n": 0}
    clvs = [c["clv_vs_dk"] for _, c in bet_clv_pairs]
    edges = [c["model_edge_vs_dk"] for _, c in bet_clv_pairs]
    wins = [b for b, _ in bet_clv_pairs if b.get("result") == "win"]
    n = len(bet_clv_pairs)
    return {
        "n": n,
        "mean_clv": sum(clvs) / n,
        "median_clv": sorted(clvs)[n // 2],
        "pct_positive_clv": sum(1 for c in clvs if c > 0) / n,
        "mean_model_edge_vs_dk": sum(edges) / n,
        "wins": len(wins),
        "win_rate": len(wins) / n,
    }


def print_clv_report():
    """Print CLV vs DK closing for all settled paper bets we can match."""
    ledger = _load_ledger()
    bets = [b for b in ledger.get("bets", []) if b.get("status") == "settled"]
    by_date = _load_espn_odds_index()
    if not by_date:
        print("No espn_odds_2026.json found — cannot compute CLV.")
        return
    if not bets:
        print("No settled bets in ledger.")
        return

    matched = []
    unmatched = []
    skipped_totals = 0
    for b in bets:
        if "OVER" in b.get("type", "") or "UNDER" in b.get("type", ""):
            skipped_totals += 1
            continue
        row = _match_paper_to_espn(b, by_date)
        if row is None:
            unmatched.append(b)
            continue
        clv = _compute_bet_clv(b, row)
        if clv is None:
            continue
        matched.append((b, clv))

    print("=" * 86)
    print("CLOSING LINE VALUE REPORT (vs DraftKings closing, from espn_odds_2026.json)")
    print("=" * 86)
    print(f"  Settled bets:   {len(bets)}")
    print(f"  Matched to DK:  {len(matched)}")
    print(f"  Unmatched:      {len(unmatched)}  (no DK closing line found for date+game)")
    if skipped_totals:
        print(f"  Skipped totals: {skipped_totals}  (CLV vs DK totals not yet implemented)")
    if not matched:
        print()
        print("  No bets could be matched to DK closing data. Cannot compute CLV.")
        return

    summary = _summarize_clv(matched)
    print()
    print(f"  CLV vs DK close (positive = we got a better price than DK closed):")
    print(f"    n:                       {summary['n']}")
    print(f"    mean CLV:                {summary['mean_clv']:+.3f}")
    print(f"    median CLV:              {summary['median_clv']:+.3f}")
    print(f"    % bets with CLV > 0:     {summary['pct_positive_clv']:.0%}")
    print(f"    mean (model − DK true):  {summary['mean_model_edge_vs_dk']:+.3f}")
    print(f"    win rate:                {summary['win_rate']:.0%} ({summary['wins']}/{summary['n']})")
    print()
    if summary["n"] < 30:
        print(f"  ⏸  Only {summary['n']} bets matched. CLV is most credible at n ≥ 30.")
        print(f"     Treat this number as directional, not conclusive.")
    elif summary["mean_clv"] > 0.01:
        print(f"  ✅ Mean CLV +{summary['mean_clv']:.1%} suggests the model was finding real value")
        print(f"     vs the DK closing line. Continue paper-trading; this is the best leading")
        print(f"     indicator of long-term edge.")
    elif summary["mean_clv"] < -0.01:
        print(f"  ❌ Mean CLV {summary['mean_clv']:+.1%} means we were systematically getting WORSE")
        print(f"     prices than DK's close. Likely the model is biased and Kalshi prices were")
        print(f"     already reflecting an edge we missed. Stop placing live bets.")
    else:
        print(f"  ⚠  Mean CLV is roughly zero. Model is not detectably beating DK close.")
        print(f"     Settled P&L on this sample is pure variance.")

    print()
    print("  PER-BET DETAIL:")
    print(f"  {'Date':<12} {'Side':<6} {'Game':<32} {'Kalshi':>8} {'Model':>8} {'DK true':>9} "
          f"{'CLV':>8} {'Result':>7}")
    print("  " + "-" * 84)
    for b, c in matched:
        result = b.get("result", "?")
        mark = "✓" if result == "win" else "✗"
        print(f"  {b['date']:<12} {b['side']:<6} {b['game'][:30]:<32} "
              f"{c['kalshi_p']:>7.1%} {c['model_p']:>7.1%} {c['market_p']:>8.1%} "
              f"{c['clv_vs_dk']:>+7.1%} {mark} {result:<5}")

    if unmatched:
        print()
        print(f"  UNMATCHED ({len(unmatched)} bets — no DK closing line cached for these games):")
        for b in unmatched[:10]:
            print(f"    {b['date']} {b['game']} ({b['type']})")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")
        print(f"  → To improve CLV coverage, fetch DK closing lines for these dates")
        print(f"    via scripts/fetch_espn_odds.py.")


def _is_total_edge(edge: dict) -> bool:
    t = edge.get("type", "")
    return "OVER" in t or "UNDER" in t


def log_bets(target_date: str, allow_ml: bool = False):
    """Log paper bets from today's pipeline output.

    By default this only logs TOTALS bets. The walk-forward backtest showed
    the moneyline model under-performs the always-home baseline out-of-sample,
    so we restrict paper trading to totals where the backtest reported 60%+
    accuracy. Pass `allow_ml=True` (or `--allow-ml` on the CLI) to override
    once moneyline has demonstrated CLV > 0 over 30+ paper bets.
    """
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

    # Totals-first: sort totals before ML so the daily exposure cap (when added)
    # spends on the higher-confidence signal first.
    edges = sorted(edges, key=lambda e: (0 if _is_total_edge(e) else 1, -e.get("ev", 0)))

    ledger = _load_ledger()
    bankroll = ledger["bankroll"]
    new_bets = []

    print(f"Paper Trading — {target_date}")
    print(f"Bankroll: ${bankroll:.2f}")
    print(f"Mode: {'TOTALS + ML' if allow_ml else 'TOTALS-ONLY (use --allow-ml to override)'}")
    print(f"{'='*70}")

    skipped_extreme = 0
    skipped_small = 0
    skipped_ml = 0
    for e in edges:
        if not allow_ml and not _is_total_edge(e):
            skipped_ml += 1
            continue

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

        # Determine side — totals use OVER/UNDER, ML uses HOME/AWAY
        if _is_total_edge(e):
            side = "OVER" if "OVER" in e["type"] else "UNDER"
        else:
            side = e["type"].split()[0]  # HOME or AWAY

        bet = {
            "date": target_date,
            "game": e["game"],
            "type": e["type"],
            "side": side,
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

    if skipped_small or skipped_extreme or skipped_ml:
        parts = []
        if skipped_small:
            parts.append(f"{skipped_small} small-edge")
        if skipped_extreme:
            parts.append(f"{skipped_extreme} extreme-market")
        if skipped_ml:
            parts.append(f"{skipped_ml} ML (totals-only mode)")
        print(f"  Filtered: {', '.join(parts)}")


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

    # CLV summary — the leading indicator. Always show this if we can match any bets.
    by_date = _load_espn_odds_index()
    matched = []
    if by_date and settled:
        for b in settled:
            if "OVER" in b.get("type", "") or "UNDER" in b.get("type", ""):
                continue
            row = _match_paper_to_espn(b, by_date)
            if row is None:
                continue
            clv = _compute_bet_clv(b, row)
            if clv is not None:
                matched.append((b, clv))
    if matched:
        s = _summarize_clv(matched)
        print()
        print(f"  CLV vs DK close: mean {s['mean_clv']:+.1%}, "
              f"{s['pct_positive_clv']:.0%} of {s['n']} matched bets had CLV > 0")
        print(f"  (See `paper_trade.py clv` for per-bet detail. CLV is the leading edge")
        print(f"  indicator — trust it before settled P&L on samples this small.)")

    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/paper_trade.py [log|settle|report|clv] [date] [--allow-ml]")
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]
    allow_ml = "--allow-ml" in args
    args = [a for a in args if a != "--allow-ml"]
    target_date = args[0] if args else date.today().isoformat()

    if cmd == "log":
        log_bets(target_date, allow_ml=allow_ml)
    elif cmd == "settle":
        settle_bets(target_date)
    elif cmd == "report":
        print_report()
    elif cmd == "clv":
        print_clv_report()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
