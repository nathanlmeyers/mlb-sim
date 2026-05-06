"""Snapshot Kalshi MLB market prices late in the day to approximate closing.

Why this exists
---------------
Kalshi's public API exposes orderbook prices for *open* markets only — once
a market settles, its last-traded price is gone. To compute Closing Line
Value (CLV) for our paper bets honestly, we need to capture prices as
close to first-pitch as possible.

The pragmatic approach: run this script every 30 minutes during MLB game
hours via GHA. Each run snapshots all currently-open MLB markets to a
timestamped history file. The "closing" price for any given game is the
LAST snapshot in which that market was still open before settling — which
in practice is whichever snapshot ran just before the game ended.

Output schema (`.context/kalshi_closing_history.json`):
    {
      "<YYYY-MM-DD HH:MM (UTC)>": {
        "<game_title>": {
          "ml": {"<side>": {"bid": ..., "ask": ..., "mid": ...}},
          "total": [{"line": 8.5, "over_bid": ..., "over_ask": ...}],
          "captured_at": "<iso ts>"
        }
      }
    }

To extract "closing" prices for analysis:
    python scripts/extract_kalshi_closing.py 2026-04-21
which collapses the per-snapshot history to a per-game last-known price
in a separate file. (That extractor is a future enhancement; for now,
just keep collecting.)

GitHub Actions schedule (add to .github/workflows/daily_pipeline.yml):

    snapshot:
      runs-on: ubuntu-latest
      schedule:
        - cron: "0,30 22-04 * * *"   # every 30 min, 18:00-00:00 ET
      steps:
        - uses: actions/checkout@v4
        - name: setup python
          uses: actions/setup-python@v5
          with: { python-version: "3.12" }
        - name: install
          run: pip install -e .
        - name: snapshot
          run: python scripts/snapshot_kalshi_closing.py
        - name: commit
          run: |
            git add .context/kalshi_closing_history.json
            git -c user.email=actions@github.com -c user.name=GHA \\
                commit -m "auto: kalshi snapshot $(date -u +%H:%M)" || true
            git push || true
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from data.kalshi import fetch_mlb_markets


HISTORY_PATH = Path(".context/kalshi_closing_history.json")


def snapshot_today() -> dict:
    """Capture all currently-open MLB market prices."""
    today = date.today().isoformat()
    print(f"Snapshotting Kalshi MLB markets for {today}...")
    markets = fetch_mlb_markets(today)
    if not markets:
        print("  No open markets found.")
        return {}

    captured_at = datetime.now(timezone.utc).isoformat(timespec="minutes")
    snapshot = {}
    for game_title, mkt in markets.items():
        ml = {}
        for side, info in mkt.get("ml", {}).items():
            if info.get("bid") is None and info.get("ask") is None:
                continue
            ml[side] = {
                "ticker": info.get("ticker"),
                "bid": info.get("bid"),
                "ask": info.get("ask"),
                "mid": info.get("mid"),
            }
        totals = []
        for t in mkt.get("total", []):
            if t.get("line") is None:
                continue
            totals.append({
                "ticker": t.get("ticker"),
                "line": t["line"],
                "over_bid": t.get("over_bid"),
                "over_ask": t.get("over_ask"),
            })
        if not ml and not totals:
            continue
        snapshot[game_title] = {
            "ml": ml,
            "total": totals,
            "captured_at": captured_at,
        }

    print(f"  Captured {len(snapshot)} games at {captured_at}")
    return {captured_at: snapshot}


def append_to_history(new_entry: dict):
    history = {}
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
    history.update(new_entry)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Wrote to {HISTORY_PATH} (total snapshots: {len(history)})")


def main():
    entry = snapshot_today()
    if entry:
        append_to_history(entry)


if __name__ == "__main__":
    main()
