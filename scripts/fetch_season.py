"""Fetch all completed games for a season and save to .context/.

Usage:
    python scripts/fetch_season.py          # fetches 2026 season to date
    python scripts/fetch_season.py 2025     # fetches specific season
"""

import sys
import json
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")
import statsapi


def fetch_season_to_date(season: int = 2026) -> tuple[list[dict], list[dict]]:
    """Fetch all completed games for a season through yesterday.

    Returns (full_games, summary_games) where full_games has box scores
    and summary_games has just scores/metadata.
    """
    # MLB Opening Day is typically late March
    start = date(season, 3, 20)
    end = min(date(season, 11, 5), date.today() - timedelta(days=1))

    print(f"Fetching {season} season: {start} to {end}")

    all_games = []
    all_summaries = []
    current = start

    while current <= end:
        # Fetch schedule for this date
        try:
            sched = statsapi.schedule(
                start_date=current.isoformat(),
                end_date=current.isoformat(),
            )
        except Exception as e:
            print(f"  Error fetching {current}: {e}")
            current += timedelta(days=1)
            continue

        final_games = [g for g in sched if g.get("status") == "Final"]

        if final_games:
            print(f"  {current}: {len(final_games)} games", end="", flush=True)

            for g in final_games:
                game_id = g["game_id"]

                # Summary record (lightweight, no box score)
                summary = {
                    "game_id": str(game_id),
                    "game_date": current.isoformat(),
                    "home_team": g["home_name"],
                    "away_team": g["away_name"],
                    "home_id": g["home_id"],
                    "away_id": g["away_id"],
                    "home_score": g["home_score"],
                    "away_score": g["away_score"],
                    "venue": g.get("venue_name", ""),
                    "home_starter": g.get("home_probable_pitcher", ""),
                    "away_starter": g.get("away_probable_pitcher", ""),
                    "winning_pitcher": g.get("winning_pitcher", ""),
                    "losing_pitcher": g.get("losing_pitcher", ""),
                }
                all_summaries.append(summary)

                # Full box score
                try:
                    box = statsapi.boxscore_data(game_id)
                    game_record = _parse_box_score(g, box, current)
                    all_games.append(game_record)
                except Exception as e:
                    print(f" [err:{game_id}]", end="", flush=True)
                    # Still include summary even if box score fails
                    all_games.append({**summary, "batting": {"home": [], "away": []}, "pitching": {"home": [], "away": []}})

                time.sleep(0.3)  # rate limit

            print(f" ✓")

        current += timedelta(days=1)

    return all_games, all_summaries


def _parse_box_score(g: dict, box: dict, game_date: date) -> dict:
    """Parse a statsapi box score into our game record format."""
    game_record = {
        "game_id": str(g["game_id"]),
        "game_date": game_date.isoformat(),
        "home_team": g["home_name"],
        "away_team": g["away_name"],
        "home_id": g["home_id"],
        "away_id": g["away_id"],
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
                "pa": pa, "ab": ab,
                "h": bat.get("hits", 0),
                "doubles": bat.get("doubles", 0),
                "triples": bat.get("triples", 0),
                "hr": bat.get("homeRuns", 0),
                "rbi": bat.get("rbi", 0),
                "bb": bb, "k": bat.get("strikeOuts", 0),
                "hbp": hbp, "sb": bat.get("stolenBases", 0), "sf": sf,
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

    return game_record


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

    full_games, summaries = fetch_season_to_date(season)

    context_dir = Path(".context")
    context_dir.mkdir(exist_ok=True)

    # Save full box scores
    full_path = context_dir / f"season_{season}.json"
    with open(full_path, "w") as f:
        json.dump(full_games, f, indent=2)

    # Save lightweight summaries
    summary_path = context_dir / f"season_{season}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSaved {len(full_games)} games to {full_path}")
    print(f"Saved {len(summaries)} summaries to {summary_path}")

    # Print season summary
    if summaries:
        dates = sorted(set(s["game_date"] for s in summaries))
        total_runs = sum(s["home_score"] + s["away_score"] for s in summaries)
        avg_total = total_runs / len(summaries) if summaries else 0
        home_wins = sum(1 for s in summaries if s["home_score"] > s["away_score"])
        print(f"\nSeason summary:")
        print(f"  Games: {len(summaries)}")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Avg total runs: {avg_total:.1f}")
        print(f"  Home win rate: {home_wins/len(summaries):.1%}")


if __name__ == "__main__":
    main()
