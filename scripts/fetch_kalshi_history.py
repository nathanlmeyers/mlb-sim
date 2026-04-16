"""Fetch historical settled MLB markets from Kalshi API.

Pulls all completed game results with closing prices for backtesting.

Usage:
    python scripts/fetch_kalshi_history.py
"""

import json
import ssl
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

API = "https://api.elections.kalshi.com/trade-api/v2"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def _get(path):
    url = f"{API}/{path}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15, context=CTX) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  API error: {e}")
        return {}


def fetch_settled_events(series: str, statuses=("settled", "closed", "finalized")):
    """Fetch all settled events for a series ticker."""
    all_events = {}
    for status in statuses:
        print(f"  Fetching {series} status={status}...")
        cursor = ""
        for page in range(10):  # max 10 pages
            path = f"events?series_ticker={series}&status={status}&limit=100&with_nested_markets=true"
            if cursor:
                path += f"&cursor={cursor}"
            data = _get(path)
            events = data.get("events", [])
            for e in events:
                all_events[e["event_ticker"]] = e
            cursor = data.get("cursor", "")
            if not cursor or not events:
                break
            time.sleep(0.3)
    return list(all_events.values())


def parse_date_from_ticker(ticker: str) -> str:
    """Parse date from ticker like KXMLBGAME-26APR092140TEXATH -> 2026-04-09."""
    # Format: KXMLB*-26MMMDDHHMMTEAMTEAM
    parts = ticker.split("-")
    if len(parts) < 2:
        return ""
    date_part = parts[1]  # e.g., "26APR092140TEXATH"
    try:
        year = "20" + date_part[:2]
        month_str = date_part[2:5]
        day = date_part[5:7]
        months = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05",
                  "JUN": "06", "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10",
                  "NOV": "11", "DEC": "12"}
        month = months.get(month_str, "01")
        return f"{year}-{month}-{day}"
    except (ValueError, IndexError):
        return ""


def fetch_market_prices(markets: list) -> list:
    """Get last price and result for each market."""
    results = []
    for m in markets:
        ticker = m.get("ticker", "")
        side = ticker.split("-")[-1] if "-" in ticker else ""

        # Try to get the market details for result/last_price
        data = _get(f"markets/{ticker}")
        mkt = data.get("market", m)
        time.sleep(0.15)

        results.append({
            "ticker": ticker,
            "side": side,
            "result": mkt.get("result", ""),
            "last_price": mkt.get("last_price"),
            "yes_sub_title": mkt.get("yes_sub_title", mkt.get("subtitle", "")),
            "volume": mkt.get("volume", 0),
            "open_interest": mkt.get("open_interest", 0),
        })
    return results


def main():
    print("Fetching historical Kalshi MLB data...")
    output = {"games": [], "totals": [], "spreads": [], "fetched_at": datetime.now().isoformat()}

    # 1. Game (moneyline) markets
    print("\n=== KXMLBGAME (Winner) ===")
    game_events = fetch_settled_events("KXMLBGAME")
    print(f"  Found {len(game_events)} settled game events")

    for e in game_events:
        game_date = parse_date_from_ticker(e["event_ticker"])
        markets = e.get("markets", [])

        # Get prices for each market
        market_data = []
        if not markets:
            # Fetch markets separately
            mkt_resp = _get(f"markets?event_ticker={e['event_ticker']}")
            markets = mkt_resp.get("markets", [])

        for m in markets:
            ticker = m.get("ticker", "")
            side = ticker.split("-")[-1] if "-" in ticker else ""
            market_data.append({
                "ticker": ticker,
                "side": side,
                "result": m.get("result", ""),
                "last_price": m.get("last_price"),
                "volume": m.get("volume", 0),
            })
            time.sleep(0.1)

        output["games"].append({
            "event_ticker": e["event_ticker"],
            "title": e.get("title", ""),
            "date": game_date,
            "markets": market_data,
        })

    # 2. Total runs markets
    print("\n=== KXMLBTOTAL (O/U) ===")
    total_events = fetch_settled_events("KXMLBTOTAL")
    print(f"  Found {len(total_events)} settled total events")

    for e in total_events:
        game_date = parse_date_from_ticker(e["event_ticker"])
        markets = e.get("markets", [])
        if not markets:
            mkt_resp = _get(f"markets?event_ticker={e['event_ticker']}")
            markets = mkt_resp.get("markets", [])

        market_data = []
        for m in markets:
            market_data.append({
                "ticker": m.get("ticker", ""),
                "subtitle": (m.get("subtitle") or m.get("title", ""))[:80],
                "result": m.get("result", ""),
                "last_price": m.get("last_price"),
                "volume": m.get("volume", 0),
            })
            time.sleep(0.1)

        output["totals"].append({
            "event_ticker": e["event_ticker"],
            "title": e.get("title", ""),
            "date": game_date,
            "markets": market_data,
        })

    # 3. Spread markets
    print("\n=== KXMLBSPREAD (Run Line) ===")
    spread_events = fetch_settled_events("KXMLBSPREAD")
    print(f"  Found {len(spread_events)} settled spread events")

    for e in spread_events:
        game_date = parse_date_from_ticker(e["event_ticker"])
        markets = e.get("markets", [])
        if not markets:
            mkt_resp = _get(f"markets?event_ticker={e['event_ticker']}")
            markets = mkt_resp.get("markets", [])

        market_data = []
        for m in markets:
            market_data.append({
                "ticker": m.get("ticker", ""),
                "subtitle": (m.get("subtitle") or m.get("title", ""))[:80],
                "result": m.get("result", ""),
                "last_price": m.get("last_price"),
                "volume": m.get("volume", 0),
            })
            time.sleep(0.1)

        output["spreads"].append({
            "event_ticker": e["event_ticker"],
            "title": e.get("title", ""),
            "date": game_date,
            "markets": market_data,
        })

    # Save
    out_path = Path(".context/kalshi_historical.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"KALSHI HISTORICAL DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Games (ML):  {len(output['games'])}")
    print(f"  Totals (O/U): {len(output['totals'])}")
    print(f"  Spreads (RL): {len(output['spreads'])}")

    if output["games"]:
        dates = sorted(set(g["date"] for g in output["games"] if g["date"]))
        if dates:
            print(f"  Date range:  {dates[0]} to {dates[-1]}")
        total_vol = sum(m.get("volume", 0) for g in output["games"] for m in g["markets"])
        print(f"  Total volume: {total_vol:,}")

    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
