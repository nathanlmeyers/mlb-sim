"""Kalshi API client for fetching MLB market prices.

Uses Kalshi's public (no-auth) API to get market data:
- KXMLBGAME: Winner (moneyline)
- KXMLBTOTAL: Total runs (O/U)
- KXMLBSPREAD: Spread (run line)
- KXMLBHR: Player home run props

Prices come from the orderbook endpoint (best bid/ask).
"""

import json
import ssl
import time
import urllib.request
import urllib.error


KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# Create SSL context that doesn't verify certs (Kalshi API has cert chain issues on some systems)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# Map Kalshi team abbreviations to standard
KALSHI_TEAM_MAP = {
    "NYM": "NYM", "LAD": "LAD", "TEX": "TEX", "ATH": "OAK", "HOU": "HOU",
    "SEI": "SEA", "SEA": "SEA", "BOS": "BOS", "MIN": "MIN", "CHC": "CHC",
    "PHI": "PHI", "LAA": "LAA", "NYY": "NYY", "MIA": "MIA", "ATL": "ATL",
    "WSH": "WSH", "PIT": "PIT", "CLE": "CLE", "STL": "STL", "AZ": "AZ",
    "ARI": "AZ", "BAL": "BAL", "BALT": "BAL", "CIN": "CIN", "COL": "COL",
    "DET": "DET", "KC": "KC", "MIL": "MIL", "OAK": "OAK", "SD": "SD",
    "SF": "SF", "TB": "TB", "TOR": "TOR", "CWS": "CWS",
}


def _api_get(path: str) -> dict:
    """Make a GET request to the Kalshi API."""
    url = f"{KALSHI_API}/{path}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"    Kalshi API error: {e}")
        return {}


def _get_orderbook_midpoint(ticker: str) -> tuple[float | None, float | None]:
    """Get best bid and ask for a market from its orderbook.

    Returns (yes_bid, yes_ask) or (None, None) if no liquidity.
    """
    data = _api_get(f"markets/{ticker}/orderbook")
    ob = data.get("orderbook_fp", data.get("orderbook", {}))

    yes_bids = ob.get("yes_dollars", [])
    no_bids = ob.get("no_dollars", [])

    best_yes_bid = float(yes_bids[-1][0]) if yes_bids else None
    best_no_bid = float(no_bids[-1][0]) if no_bids else None

    yes_ask = (1.0 - best_no_bid) if best_no_bid is not None else None

    return best_yes_bid, yes_ask


def fetch_mlb_markets(target_date: str) -> dict:
    """Fetch all MLB markets from Kalshi for a given date.

    Args:
        target_date: "YYYY-MM-DD" format

    Returns dict keyed by game identifier with market data:
    {
        "TEX@OAK": {
            "ml": {"home": {"ticker": ..., "bid": 0.47, "ask": 0.48}, "away": {...}},
            "total": [{"line": 8.5, "over_bid": 0.52, "over_ask": 0.54}, ...],
            "spread": [{"line": -1.5, "home_bid": 0.38, "home_ask": 0.40}, ...],
            "hr": [{"player": "Ohtani", "ticker": ..., "bid": 0.23, "ask": 0.24}, ...],
        }
    }
    """
    # Parse date for ticker matching (APR13 format)
    from datetime import datetime
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    date_str = dt.strftime("%b%d").upper()  # e.g., "APR13"

    result = {}

    # 1. Moneyline (KXMLBGAME)
    print("    Fetching Kalshi moneyline markets...")
    data = _api_get(f"events?series_ticker=KXMLBGAME&status=open&limit=50&with_nested_markets=true")
    for event in data.get("events", []):
        if date_str not in event.get("event_ticker", ""):
            continue

        title = event.get("title", "")
        markets = event.get("markets", [])

        # Parse teams from ticker: KXMLBGAME-26APR132140TEXATH
        eticker = event["event_ticker"]

        game_key = title
        if game_key not in result:
            result[game_key] = {"ml": {}, "total": [], "spread": [], "hr": [], "event_ticker": eticker}

        for m in markets:
            side = m["ticker"].split("-")[-1]  # e.g., "TEX" or "ATH"
            bid, ask = _get_orderbook_midpoint(m["ticker"])
            time.sleep(0.15)

            result[game_key]["ml"][side] = {
                "ticker": m["ticker"],
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2 if bid and ask else bid or ask,
            }

    # 2. Totals (KXMLBTOTAL)
    print("    Fetching Kalshi total (O/U) markets...")
    data = _api_get(f"events?series_ticker=KXMLBTOTAL&status=open&limit=50&with_nested_markets=true")
    for event in data.get("events", []):
        if date_str not in event.get("event_ticker", ""):
            continue

        title = event.get("title", "").replace(": Total Runs", "")
        if title not in result:
            result[title] = {"ml": {}, "total": [], "spread": [], "hr": [], "event_ticker": event["event_ticker"]}

        for m in event.get("markets", []):
            subtitle = m.get("subtitle", m.get("title", ""))
            bid, ask = _get_orderbook_midpoint(m["ticker"])
            time.sleep(0.15)

            # Parse line from subtitle (e.g., "Over 8.5 runs")
            line = None
            for word in subtitle.split():
                try:
                    line = float(word)
                    break
                except ValueError:
                    pass

            result[title]["total"].append({
                "ticker": m["ticker"],
                "subtitle": subtitle,
                "line": line,
                "over_bid": bid,
                "over_ask": ask,
            })

    # 3. Spread (KXMLBSPREAD)
    print("    Fetching Kalshi spread markets...")
    data = _api_get(f"events?series_ticker=KXMLBSPREAD&status=open&limit=50&with_nested_markets=true")
    for event in data.get("events", []):
        if date_str not in event.get("event_ticker", ""):
            continue

        title = event.get("title", "").replace(": Spread", "")
        if title not in result:
            result[title] = {"ml": {}, "total": [], "spread": [], "hr": [], "event_ticker": event["event_ticker"]}

        for m in event.get("markets", []):
            subtitle = m.get("subtitle", m.get("title", ""))
            bid, ask = _get_orderbook_midpoint(m["ticker"])
            time.sleep(0.15)

            result[title]["spread"].append({
                "ticker": m["ticker"],
                "subtitle": subtitle,
                "bid": bid,
                "ask": ask,
            })

    # 4. HR props (KXMLBHR) — lighter fetch, just count
    print("    Fetching Kalshi HR prop markets...")
    data = _api_get(f"events?series_ticker=KXMLBHR&status=open&limit=50&with_nested_markets=true")
    hr_count = 0
    for event in data.get("events", []):
        if date_str not in event.get("event_ticker", ""):
            continue

        title = event.get("title", "").replace(": Home Runs", "")
        if title not in result:
            result[title] = {"ml": {}, "total": [], "spread": [], "hr": [], "event_ticker": event["event_ticker"]}

        for m in event.get("markets", []):
            hr_count += 1
            result[title]["hr"].append({
                "ticker": m["ticker"],
                "subtitle": m.get("subtitle", m.get("title", "")),
            })

    print(f"    Found {len(result)} games with {sum(len(v['ml']) for v in result.values())} ML, "
          f"{sum(len(v['total']) for v in result.values())} total, "
          f"{sum(len(v['spread']) for v in result.values())} spread, "
          f"{hr_count} HR markets")

    return result


def format_kalshi_report(markets: dict) -> str:
    """Format Kalshi markets into a readable report."""
    lines = []
    for game, data in sorted(markets.items()):
        lines.append(f"\n  {game}")

        # ML
        if data["ml"]:
            parts = []
            for side, info in data["ml"].items():
                mid = info.get("mid")
                if mid:
                    parts.append(f"{side} {mid:.0%}")
            if parts:
                lines.append(f"    ML: {' | '.join(parts)}")

        # Totals
        if data["total"]:
            parts = []
            for t in sorted(data["total"], key=lambda x: x.get("line") or 0):
                if t.get("over_bid") and t.get("line"):
                    mid = (t["over_bid"] + (t["over_ask"] or t["over_bid"])) / 2
                    parts.append(f"O{t['line']}:{mid:.0%}")
            if parts:
                lines.append(f"    O/U: {' | '.join(parts)}")

        # Spread
        if data["spread"]:
            parts = []
            for s in data["spread"]:
                if s.get("bid"):
                    mid = (s["bid"] + (s["ask"] or s["bid"])) / 2
                    parts.append(f"{s['subtitle'][:30]}:{mid:.0%}")
            if parts:
                lines.append(f"    Spread: {' | '.join(parts)}")

    return "\n".join(lines)
