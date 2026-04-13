"""Fetch and store daily MLB game data.

Pulls yesterday's (or a specified date's) games via the MLB Stats API
and saves structured box scores to .context/ for the simulation engine.

Usage:
    python scripts/fetch_daily.py              # fetches yesterday
    python scripts/fetch_daily.py 2026-04-09   # fetches specific date
"""

import sys
import json
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")
import statsapi


def fetch_games_for_date(game_date: str) -> list[dict]:
    """Fetch all completed games and box scores for a date."""
    sched = statsapi.schedule(start_date=game_date, end_date=game_date)
    games_data = []

    for g in sched:
        if g.get("status") != "Final":
            continue

        game_id = g["game_id"]
        print(f"  Fetching {g['away_name']} @ {g['home_name']} (game {game_id})...")

        try:
            box = statsapi.boxscore_data(game_id)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        game_record = {
            "game_id": str(game_id),
            "game_date": game_date,
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "venue": g.get("venue_name", ""),
            "batting": {"home": [], "away": []},
            "pitching": {"home": [], "away": []},
        }

        for side in ["home", "away"]:
            team_data = box.get(side, {})
            batter_ids = team_data.get("batters", [])
            pitcher_ids = team_data.get("pitchers", [])
            players = team_data.get("players", {})

            for pid in batter_ids:
                pdata = players.get(f"ID{pid}", {})
                bat = pdata.get("stats", {}).get("batting", {})
                ab = bat.get("atBats", 0)
                bb = bat.get("baseOnBalls", 0)
                hbp = bat.get("hitByPitch", 0)
                sf = bat.get("sacFlies", 0)
                sh = bat.get("sacBunts", 0)
                pa = ab + bb + hbp + sf + sh
                if pa == 0:
                    continue
                game_record["batting"][side].append({
                    "name": pdata.get("person", {}).get("fullName", ""),
                    "player_id": pid,
                    "pa": pa,
                    "ab": ab,
                    "h": bat.get("hits", 0),
                    "doubles": bat.get("doubles", 0),
                    "triples": bat.get("triples", 0),
                    "hr": bat.get("homeRuns", 0),
                    "rbi": bat.get("rbi", 0),
                    "bb": bb,
                    "k": bat.get("strikeOuts", 0),
                    "hbp": hbp,
                    "sb": bat.get("stolenBases", 0),
                    "sf": sf,
                })

            for pid in pitcher_ids:
                pdata = players.get(f"ID{pid}", {})
                pit = pdata.get("stats", {}).get("pitching", {})
                if not pit or not pit.get("inningsPitched"):
                    continue
                game_record["pitching"][side].append({
                    "name": pdata.get("person", {}).get("fullName", ""),
                    "player_id": pid,
                    "ip": pit.get("inningsPitched", "0"),
                    "h": pit.get("hits", 0),
                    "r": pit.get("runs", 0),
                    "er": pit.get("earnedRuns", 0),
                    "bb": pit.get("baseOnBalls", 0),
                    "k": pit.get("strikeOuts", 0),
                    "hr": pit.get("homeRuns", 0),
                    "pitches": pit.get("numberOfPitches", 0),
                    "batters_faced": pit.get("battersFaced", 0),
                })

        games_data.append(game_record)

    return games_data


def main():
    if len(sys.argv) > 1:
        game_date = sys.argv[1]
    else:
        game_date = (date.today() - timedelta(days=1)).isoformat()

    print(f"Fetching MLB games for {game_date}...")
    games = fetch_games_for_date(game_date)

    if not games:
        print("No completed games found.")
        return

    # Save to .context/
    context_dir = Path(".context")
    context_dir.mkdir(exist_ok=True)
    out_path = context_dir / f"games_{game_date.replace('-', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(games, f, indent=2)

    print(f"\nSaved {len(games)} games to {out_path}")

    # Print summary
    total_pa = 0
    for g in games:
        hb = len(g["batting"]["home"])
        ab = len(g["batting"]["away"])
        gpa = sum(b["pa"] for b in g["batting"]["home"]) + sum(b["pa"] for b in g["batting"]["away"])
        total_pa += gpa
        print(f"  {g['away_team']:>25} {g['away_score']} @ {g['home_team']} {g['home_score']}  ({gpa} PA)")

    print(f"\nTotal: {len(games)} games, {total_pa} plate appearances")


if __name__ == "__main__":
    main()
