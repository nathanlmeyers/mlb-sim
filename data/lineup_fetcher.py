"""Fetch real MLB lineups and build per-player models from the Stats API.

Handles lineup fetching, player stat retrieval, and model construction
for both batters and pitchers using live API data + cached season history.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsapi

from models.batter_model import BatterModel, build_league_average_batter
from models.pitcher_model import PitcherModel, build_league_average_pitcher
import config


# Cache player stats for the day to avoid repeated API calls
_player_cache: dict[int, dict] = {}


def fetch_game_lineups(game_id: str) -> dict | None:
    """Fetch confirmed lineup, bench, and bullpen for a game.

    Returns None if lineups aren't posted yet.
    """
    try:
        game_data = statsapi.get("game", {"gamePk": int(game_id)})
    except Exception as e:
        print(f"    Error fetching game {game_id}: {e}")
        return None

    live = game_data.get("liveData", {})
    box = live.get("boxscore", {})

    result = {}
    for side in ["home", "away"]:
        team = box.get("teams", {}).get(side, {})
        batting_order = team.get("battingOrder", [])
        bench = team.get("bench", [])
        bullpen = team.get("bullpen", [])
        players = team.get("players", {})

        # Get starter (first pitcher listed or from probable pitcher)
        pitchers = team.get("pitchers", [])
        starter_id = pitchers[0] if pitchers else 0

        # Extract bat/throw hand and position from player data
        lineup_info = []
        for pid in batting_order:
            pdata = players.get(f"ID{pid}", {})
            lineup_info.append({
                "id": pid,
                "name": pdata.get("person", {}).get("fullName", ""),
                "position": pdata.get("position", {}).get("abbreviation", ""),
                "bat_side": pdata.get("batSide", {}).get("code", "R"),
            })

        bench_info = []
        for pid in bench:
            pdata = players.get(f"ID{pid}", {})
            bench_info.append({
                "id": pid,
                "name": pdata.get("person", {}).get("fullName", ""),
                "bat_side": pdata.get("batSide", {}).get("code", "R"),
            })

        bp_info = []
        for pid in bullpen:
            pdata = players.get(f"ID{pid}", {})
            bp_info.append({
                "id": pid,
                "name": pdata.get("person", {}).get("fullName", ""),
                "throw_hand": pdata.get("pitchHand", {}).get("code", "R"),
            })

        result[f"{side}_lineup"] = lineup_info
        result[f"{side}_bench"] = bench_info
        result[f"{side}_bullpen"] = bp_info
        result[f"{side}_starter_id"] = starter_id

    return result


def fetch_lineups_from_schedule(target_date: str) -> dict[str, dict]:
    """Fetch lineups for all games on a date using the schedule endpoint.

    Returns dict: game_id -> lineup data.
    Faster than fetching each game individually.
    """
    try:
        sched = statsapi.get("schedule", {
            "date": target_date,
            "sportId": 1,
            "hydrate": "probablePitcher,lineups",
        })
    except Exception as e:
        print(f"  Error fetching schedule: {e}")
        return {}

    lineups = {}
    for date_entry in sched.get("dates", []):
        for g in date_entry.get("games", []):
            game_id = str(g.get("gamePk", ""))
            game_lineups = g.get("lineups", {})

            home_players = game_lineups.get("homePlayers", [])
            away_players = game_lineups.get("awayPlayers", [])

            home_team = g.get("teams", {}).get("home", {})
            away_team = g.get("teams", {}).get("away", {})
            home_sp = home_team.get("probablePitcher", {})
            away_sp = away_team.get("probablePitcher", {})

            lineups[game_id] = {
                "home_lineup_ids": [p.get("id") for p in home_players],
                "away_lineup_ids": [p.get("id") for p in away_players],
                "home_starter_id": home_sp.get("id", 0),
                "away_starter_id": away_sp.get("id", 0),
                "home_starter_name": home_sp.get("fullName", "TBD"),
                "away_starter_name": away_sp.get("fullName", "TBD"),
            }

    return lineups


def _fetch_player_stats(player_id: int, stat_group: str = "hitting") -> dict:
    """Fetch a player's current season stats from the API. Cached per session."""
    cache_key = (player_id, stat_group)
    if player_id in _player_cache and stat_group in _player_cache.get(player_id, {}):
        return _player_cache[player_id][stat_group]

    try:
        data = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=[{stat_group}],type=[season])",
        })
    except Exception:
        return {}

    people = data.get("people", [])
    if not people:
        return {}

    p = people[0]
    result = {"fullName": p.get("fullName", ""), "id": player_id}

    if stat_group == "hitting":
        result["bat_side"] = p.get("batSide", {}).get("code", "R")
    elif stat_group == "pitching":
        result["throw_hand"] = p.get("pitchHand", {}).get("code", "R")

    for sg in p.get("stats", []):
        splits = sg.get("splits", [])
        if splits:
            result["stats"] = splits[0].get("stat", {})
            break

    # Cache
    if player_id not in _player_cache:
        _player_cache[player_id] = {}
    _player_cache[player_id][stat_group] = result

    time.sleep(0.2)  # rate limit
    return result


def _lookup_prior_season_stats(
    player_id: int,
    player_name: str,
    prior_games: list[dict] | None,
    role: str = "batting",
) -> dict:
    """Look up a player's 2025 stats from cached season data."""
    if not prior_games:
        return {}

    totals = defaultdict(int)
    games_found = 0

    for g in prior_games:
        for side in ["home", "away"]:
            players = g.get(role if role == "batting" else "pitching", {}).get(side, [])
            for p in players:
                if p.get("player_id") == player_id or p.get("name") == player_name:
                    games_found += 1
                    for key in ["pa", "ab", "h", "doubles", "triples", "hr", "bb", "k",
                                "hbp", "sb", "sf", "ip", "er", "pitches"]:
                        if key in p:
                            val = p[key]
                            if isinstance(val, str):
                                try:
                                    val = float(val)
                                except (ValueError, TypeError):
                                    val = 0
                            totals[key] += val

    if games_found == 0:
        return {}

    totals["games"] = games_found
    return dict(totals)


def build_batter_model(
    player_id: int,
    prior_season_games: list[dict] | None = None,
) -> BatterModel:
    """Build a BatterModel for a specific player from API + cached data."""
    api_data = _fetch_player_stats(player_id, "hitting")
    stats = api_data.get("stats", {})
    name = api_data.get("fullName", f"Player {player_id}")
    bat_side = api_data.get("bat_side", "R")

    # Current season stats from API
    pa = stats.get("plateAppearances", 0) or 0
    ab = stats.get("atBats", 0) or 0
    h = stats.get("hits", 0) or 0
    hr = stats.get("homeRuns", 0) or 0
    bb = stats.get("baseOnBalls", 0) or 0
    k = stats.get("strikeOuts", 0) or 0
    hbp = stats.get("hitByPitch", 0) or 0
    doubles = stats.get("doubles", 0) or 0
    triples = stats.get("triples", 0) or 0
    sb = stats.get("stolenBases", 0) or 0
    go = stats.get("groundOuts", 0) or 0
    ao = stats.get("airOuts", 0) or 0

    # Look up prior season stats
    prior = _lookup_prior_season_stats(player_id, name, prior_season_games, "batting")
    prior_pa = prior.get("pa", 0)

    # Compute rates from current + prior season
    total_pa = pa + prior_pa * 0.5  # discount prior season by 50%
    total_k = k + prior.get("k", 0) * 0.5
    total_bb = bb + prior.get("bb", 0) * 0.5
    total_hbp = hbp + prior.get("hbp", 0) * 0.5
    total_hr = hr + prior.get("hr", 0) * 0.5

    if total_pa < 10:
        # Not enough data — return league average
        # imported at top level
        avg = build_league_average_batter()
        avg.player_id = player_id
        avg.player_name = name
        avg.bats = bat_side
        return avg

    raw_k = total_k / total_pa
    raw_bb = total_bb / total_pa
    raw_hbp = total_hbp / total_pa

    # Bayesian regression
    reg_k = config.REGRESSION_SAMPLES["k_pct"]
    reg_bb = config.REGRESSION_SAMPLES["bb_pct"]
    k_pct = (total_pa * raw_k + reg_k * config.LEAGUE_AVG_K_PCT) / (total_pa + reg_k)
    bb_pct = (total_pa * raw_bb + reg_bb * config.LEAGUE_AVG_BB_PCT) / (total_pa + reg_bb)
    hbp_pct = (total_pa * raw_hbp + 300 * config.LEAGUE_AVG_HBP_PCT) / (total_pa + 300)

    # BABIP
    bip = ab - k
    h_minus_hr = h - hr
    raw_babip = h_minus_hr / max(bip, 1)
    babip = (max(bip, 0) * raw_babip + 400 * config.LEAGUE_AVG_BABIP) / (max(bip, 0) + 400)

    # HR/FB (estimate)
    raw_hr_rate = total_hr / total_pa
    hr_per_fb = raw_hr_rate / config.LEAGUE_AVG_FB_PCT if config.LEAGUE_AVG_FB_PCT > 0 else config.LEAGUE_AVG_HR_PER_FB
    hr_per_fb = (total_pa * hr_per_fb + 300 * config.LEAGUE_AVG_HR_PER_FB) / (total_pa + 300)

    # GB/FB from API ground/air outs
    total_batted = go + ao
    if total_batted > 20:
        gb_pct = go / total_batted
        fb_pct = ao / total_batted
        ld_pct = max(0.10, 1.0 - gb_pct - fb_pct)
        # Normalize
        total_bb_pct = gb_pct + fb_pct + ld_pct
        gb_pct /= total_bb_pct
        fb_pct /= total_bb_pct
        ld_pct /= total_bb_pct
    else:
        gb_pct = config.LEAGUE_AVG_GB_PCT
        fb_pct = config.LEAGUE_AVG_FB_PCT
        ld_pct = config.LEAGUE_AVG_LD_PCT

    # Speed score
    speed_score = min(10.0, (sb / max(pa / 100, 1)) * 3.0)

    # Simple platoon adjustments (same as before)
    return BatterModel(
        player_id=player_id,
        player_name=name,
        team_id=0,
        bats=bat_side,
        k_pct=k_pct, bb_pct=bb_pct, hbp_pct=hbp_pct,
        gb_pct=gb_pct, fb_pct=fb_pct, ld_pct=ld_pct,
        hr_per_fb=hr_per_fb, babip=babip,
        k_pct_vs_lhp=k_pct + config.PLATOON_ADJ_BATTER_VS_OPPOSITE * 0.5,
        bb_pct_vs_lhp=bb_pct + abs(config.PLATOON_ADJ_BATTER_VS_OPPOSITE) * 0.3,
        hr_per_fb_vs_lhp=hr_per_fb * 1.05,
        babip_vs_lhp=babip + 0.010,
        k_pct_vs_rhp=k_pct + config.PLATOON_ADJ_BATTER_VS_SAME * 0.5,
        bb_pct_vs_rhp=bb_pct - abs(config.PLATOON_ADJ_BATTER_VS_SAME) * 0.3,
        hr_per_fb_vs_rhp=hr_per_fb * 0.95,
        babip_vs_rhp=babip - 0.010,
        speed_score=speed_score,
        games_played=int(pa / 4) if pa > 0 else 0,
        pa_total=int(total_pa),
    )


def build_pitcher_model(
    player_id: int,
    prior_season_games: list[dict] | None = None,
) -> PitcherModel:
    """Build a PitcherModel for a specific pitcher from API + cached data."""
    api_data = _fetch_player_stats(player_id, "pitching")
    stats = api_data.get("stats", {})
    name = api_data.get("fullName", f"Pitcher {player_id}")
    throws = api_data.get("throw_hand", "R")

    bf = stats.get("battersFaced", 0) or 0
    k = stats.get("strikeOuts", 0) or 0
    bb = stats.get("baseOnBalls", 0) or 0
    hr = stats.get("homeRuns", 0) or 0
    h = stats.get("hits", 0) or 0
    ip_str = stats.get("inningsPitched", "0")
    pitches = stats.get("numberOfPitches", 0) or 0
    strikes = stats.get("strikes", 0) or 0
    go = stats.get("groundOuts", 0) or 0
    ao = stats.get("airOuts", 0) or 0
    gs = stats.get("gamesStarted", 0) or 0

    try:
        ip = float(ip_str)
    except (ValueError, TypeError):
        ip = 0.0

    # Estimate BF if not provided
    if not bf and ip > 0:
        bf = int(3 * ip + h + bb)

    # Strike% from API (key command metric)
    strike_pct_str = stats.get("strikePercentage", "")
    if strike_pct_str:
        try:
            strike_pct = float(strike_pct_str)
        except (ValueError, TypeError):
            strike_pct = strikes / max(pitches, 1) if pitches > 0 else config.LEAGUE_AVG_STRIKE_PCT
    elif pitches > 0:
        strike_pct = strikes / pitches
    else:
        strike_pct = config.LEAGUE_AVG_STRIKE_PCT

    # Look up prior season
    prior = _lookup_prior_season_stats(player_id, name, prior_season_games, "pitching")
    prior_bf = int(prior.get("pa", 0) or prior.get("ab", 0) or 0)
    if not prior_bf and prior.get("ip", 0):
        prior_bf = int(3 * prior["ip"] + prior.get("h", 0) + prior.get("bb", 0))

    total_bf = bf + prior_bf * 0.5
    total_k = k + prior.get("k", 0) * 0.5
    total_bb = bb + prior.get("bb", 0) * 0.5
    total_hr = hr + prior.get("hr", 0) * 0.5

    if total_bf < 10:
        # imported at top level
        avg = build_league_average_pitcher(is_starter=gs > 0)
        avg.player_id = player_id
        avg.player_name = name
        avg.throws = throws
        avg.strike_pct = strike_pct
        return avg

    raw_k = total_k / total_bf
    raw_bb = total_bb / total_bf
    raw_hr_rate = total_hr / total_bf

    # Bayesian regression
    k_pct = (total_bf * raw_k + 60 * config.LEAGUE_AVG_K_PCT) / (total_bf + 60)
    bb_pct = (total_bf * raw_bb + 100 * config.LEAGUE_AVG_BB_PCT) / (total_bf + 100)
    hr_per_fb_est = raw_hr_rate / config.LEAGUE_AVG_FB_PCT
    hr_per_fb = (total_bf * hr_per_fb_est + 80 * config.LEAGUE_AVG_HR_PER_FB) / (total_bf + 80)

    # Batted ball profile from GO/AO
    total_batted = go + ao
    if total_batted > 15:
        gb_pct = go / total_batted
        fb_pct = ao / total_batted
        ld_pct = max(0.10, 1.0 - gb_pct - fb_pct)
        t = gb_pct + fb_pct + ld_pct
        gb_pct /= t; fb_pct /= t; ld_pct /= t
    else:
        gb_pct = config.LEAGUE_AVG_GB_PCT
        fb_pct = config.LEAGUE_AVG_FB_PCT
        ld_pct = config.LEAGUE_AVG_LD_PCT

    is_starter = gs > 0
    avg_pitches = pitches / max(gs, 1) if gs > 0 else 20
    avg_ip = ip / max(gs, 1) if gs > 0 else 1.0

    # Platoon splits (simplified)
    if throws == "L":
        k_vs_l = k_pct * 1.08; k_vs_r = k_pct * 0.92
        bb_vs_l = bb_pct * 0.90; bb_vs_r = bb_pct * 1.10
    else:
        k_vs_r = k_pct * 1.08; k_vs_l = k_pct * 0.92
        bb_vs_r = bb_pct * 0.90; bb_vs_l = bb_pct * 1.10

    return PitcherModel(
        player_id=player_id, player_name=name, team_id=0, throws=throws,
        k_pct=k_pct, bb_pct=bb_pct, hbp_pct=config.LEAGUE_AVG_HBP_PCT,
        gb_pct=gb_pct, fb_pct=fb_pct, ld_pct=ld_pct,
        hr_per_fb=hr_per_fb, babip_allowed=config.LEAGUE_AVG_BABIP,
        k_pct_vs_lhb=k_vs_l, bb_pct_vs_lhb=bb_vs_l,
        hr_per_fb_vs_lhb=hr_per_fb, babip_vs_lhb=config.LEAGUE_AVG_BABIP,
        k_pct_vs_rhb=k_vs_r, bb_pct_vs_rhb=bb_vs_r,
        hr_per_fb_vs_rhb=hr_per_fb, babip_vs_rhb=config.LEAGUE_AVG_BABIP,
        avg_pitches_per_start=avg_pitches,
        avg_innings_per_start=avg_ip,
        strike_pct=strike_pct,
        is_starter=is_starter,
        is_closer=False,
        games_played=int(total_bf / 20),
        batters_faced_total=int(total_bf),
    )


def build_game_models(
    game_id: str,
    lineup_data: dict,
    prior_season_games: list[dict] | None = None,
) -> dict:
    """Build complete models for a game: lineups, starters, bullpen.

    Returns {
        'home_lineup': [9 BatterModels],
        'away_lineup': [9 BatterModels],
        'home_bench': [BatterModels],
        'away_bench': [BatterModels],
        'home_starter': PitcherModel,
        'away_starter': PitcherModel,
        'home_bullpen': [PitcherModels],
        'away_bullpen': [PitcherModels],
    }
    """
    result = {}

    for side in ["home", "away"]:
        # Build lineup batters
        lineup_ids = lineup_data.get(f"{side}_lineup_ids", [])
        lineup = []
        for pid in lineup_ids[:9]:
            if pid:
                batter = build_batter_model(pid, prior_season_games)
                lineup.append(batter)

        # Pad to 9 if needed
        while len(lineup) < 9:
            # imported at top level
            avg = build_league_average_batter()
            avg.player_id = len(lineup) + 1
            lineup.append(avg)

        result[f"{side}_lineup"] = lineup

        # Build starter
        sp_id = lineup_data.get(f"{side}_starter_id", 0)
        if sp_id:
            result[f"{side}_starter"] = build_pitcher_model(sp_id, prior_season_games)
        else:
            # imported at top level
            result[f"{side}_starter"] = build_league_average_pitcher(is_starter=True)

        # Build bullpen (fetch from game data if available)
        result[f"{side}_bullpen"] = [
            build_league_average_pitcher(is_starter=False)
            for _ in range(5)
        ]
        result[f"{side}_bench"] = []

    return result
