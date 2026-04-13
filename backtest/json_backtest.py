"""JSON-backed backtest — no PostgreSQL required.

Reads season data from .context/season_YYYY.json and evaluates
predictions against actual outcomes. Builds team + pitcher models
in-memory from the JSON game history.

v2: Adds pitcher-specific models, dynamic regression, Pythagorean
    win%, and recent form/momentum signals.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, ".")

from models.pitcher_model import PitcherModel, build_league_average_pitcher
from models.team_model import TeamModel, ParkFactor
from sim.game import simulate_n_box_score_games
import config


# Map team names from statsapi to abbreviations
TEAM_ABBR = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "OAK", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

FANGRAPHS_TEAM_TO_ABBR = {
    "Angels": "LAA", "Orioles": "BAL", "Red Sox": "BOS", "White Sox": "CWS",
    "Guardians": "CLE", "Tigers": "DET", "Royals": "KC", "Twins": "MIN",
    "Yankees": "NYY", "Athletics": "OAK", "Mariners": "SEA", "Rays": "TB",
    "Rangers": "TEX", "Blue Jays": "TOR", "Diamondbacks": "AZ", "D-backs": "AZ",
    "Braves": "ATL", "Cubs": "CHC", "Reds": "CIN", "Rockies": "COL",
    "Marlins": "MIA", "Astros": "HOU", "Dodgers": "LAD", "Brewers": "MIL",
    "Nationals": "WSH", "Mets": "NYM", "Phillies": "PHI", "Pirates": "PIT",
    "Cardinals": "STL", "Padres": "SD", "Giants": "SF",
}


@dataclass
class BacktestResult:
    total_games: int
    correct_picks: int
    accuracy: float
    brier_score: float
    avg_margin_error: float
    avg_total_error: float
    rl_correct: int = 0
    rl_accuracy: float = 0.0
    ou_correct: int = 0
    ou_accuracy: float = 0.0
    calibration_by_decile: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)


def load_park_factors_from_json() -> dict[str, ParkFactor]:
    """Load park factors from .context/park_factors_2024.json."""
    pf_path = Path(".context/park_factors_2024.json")
    if not pf_path.exists():
        return {}

    with open(pf_path) as f:
        data = json.load(f)

    factors = {}
    for team in data.get("teams", []):
        abbr = FANGRAPHS_TEAM_TO_ABBR.get(team["team"], team["team"])
        factors[abbr] = ParkFactor(
            venue=team["team"],
            overall_factor=team["basic_5yr"] / 100.0,
            hr_factor=team["hr"] / 100.0,
            h_factor=team["single"] / 100.0,
            bb_factor=team["bb"] / 100.0,
        )
    return factors


# ---------------------------------------------------------------------------
# Pitcher model from JSON game logs
# ---------------------------------------------------------------------------

def _build_pitcher_index(all_games: list[dict]) -> dict[str, list[dict]]:
    """Pre-index pitcher appearances by name for fast lookup.

    Returns dict: pitcher_name -> list of {date, k, bb, hr, h, ip, pitches, is_start}
    """
    index = defaultdict(list)
    for g in all_games:
        game_date = g["game_date"]
        for side in ["home", "away"]:
            pitchers = g.get("pitching", {}).get(side, [])
            for i, p in enumerate(pitchers):
                name = p["name"]
                ip = 0.0
                try:
                    ip = float(p.get("ip", 0))
                except (ValueError, TypeError):
                    pass
                # Estimate batters faced when not available
                bf = p.get("batters_faced", 0)
                if not bf and ip > 0:
                    bf = int(3 * ip + p.get("h", 0) + p.get("bb", 0))
                index[name].append({
                    "date": game_date,
                    "k": p.get("k", 0),
                    "bb": p.get("bb", 0),
                    "hr": p.get("hr", 0),
                    "h": p.get("h", 0),
                    "ip": ip,
                    "bf": bf,
                    "pitches": p.get("pitches", 0),
                    "er": p.get("er", 0),
                    "is_start": i == 0,
                })
    return index


def build_pitcher_from_json(
    pitcher_name: str,
    pitcher_index: dict[str, list[dict]],
    as_of_date: str,
    prior_season_index: dict[str, list[dict]] | None = None,
) -> PitcherModel | None:
    """Build a pitcher model from accumulated JSON game logs.

    Uses prior-season data as the regression prior when available,
    falling back to league averages. This is critical early-season
    when a pitcher may only have 2-3 starts in the current year
    but 30+ from last season.
    """
    # Current season appearances before as_of_date
    appearances = pitcher_index.get(pitcher_name, [])
    current = [a for a in appearances if a["date"] < as_of_date]

    # Prior season appearances (all of them — already completed)
    prior = prior_season_index.get(pitcher_name, []) if prior_season_index else []

    if len(current) < 1 and len(prior) < 5:
        return None

    # Compute prior-season baselines (used as regression target)
    prior_k_pct = config.LEAGUE_AVG_K_PCT
    prior_bb_pct = config.LEAGUE_AVG_BB_PCT
    prior_hr_per_fb = config.LEAGUE_AVG_HR_PER_FB

    if prior:
        prior_bf = sum(a["bf"] for a in prior)
        if prior_bf >= 30:
            prior_k_pct = sum(a["k"] for a in prior) / prior_bf
            prior_bb_pct = sum(a["bb"] for a in prior) / prior_bf
            prior_hr_rate = sum(a["hr"] for a in prior) / prior_bf
            prior_hr_per_fb = prior_hr_rate / config.LEAGUE_AVG_FB_PCT if config.LEAGUE_AVG_FB_PCT > 0 else config.LEAGUE_AVG_HR_PER_FB

    # Blend current season + prior season stats (prior at 50% weight)
    # This was the optimal config from grid search: wt=0.5, reg=20
    PRIOR_WEIGHT = 0.5

    c_bf = sum(a["bf"] for a in current)
    c_k = sum(a["k"] for a in current)
    c_bb = sum(a["bb"] for a in current)
    c_hr = sum(a["hr"] for a in current)
    c_ip = sum(a["ip"] for a in current)
    c_pitches = sum(a["pitches"] for a in current)
    starts = sum(1 for a in current if a["is_start"])

    p_bf = sum(a["bf"] for a in prior)
    p_k = sum(a["k"] for a in prior)
    p_bb = sum(a["bb"] for a in prior)
    p_hr = sum(a["hr"] for a in prior)

    # Blend: current at full weight + prior at discounted weight
    total_bf = c_bf + p_bf * PRIOR_WEIGHT
    total_k = c_k + p_k * PRIOR_WEIGHT
    total_bb = c_bb + p_bb * PRIOR_WEIGHT
    total_hr = c_hr + p_hr * PRIOR_WEIGHT

    if total_bf < 10:
        return None

    raw_k_pct = total_k / total_bf
    raw_bb_pct = total_bb / total_bf
    raw_hr_rate = total_hr / total_bf

    # Light regression toward league average (reg=20 optimal from grid search)
    reg_n = 20
    k_pct = (total_bf * raw_k_pct + reg_n * config.LEAGUE_AVG_K_PCT) / (total_bf + reg_n)
    bb_pct = (total_bf * raw_bb_pct + reg_n * config.LEAGUE_AVG_BB_PCT) / (total_bf + reg_n)

    hr_per_fb_est = raw_hr_rate / config.LEAGUE_AVG_FB_PCT if config.LEAGUE_AVG_FB_PCT > 0 else config.LEAGUE_AVG_HR_PER_FB
    hr_per_fb = (total_bf * hr_per_fb_est + reg_n * config.LEAGUE_AVG_HR_PER_FB) / (total_bf + reg_n)

    # Estimate strike% from K%/BB% (correlates well)
    strike_pct = np.clip(0.55 + k_pct * 0.5 - bb_pct * 0.3, 0.52, 0.72)

    return PitcherModel(
        player_id=0,
        player_name=pitcher_name,
        team_id=0,
        throws="R",
        k_pct=k_pct,
        bb_pct=bb_pct,
        hbp_pct=config.LEAGUE_AVG_HBP_PCT,
        gb_pct=config.LEAGUE_AVG_GB_PCT,
        fb_pct=config.LEAGUE_AVG_FB_PCT,
        ld_pct=config.LEAGUE_AVG_LD_PCT,
        hr_per_fb=hr_per_fb,
        babip_allowed=config.LEAGUE_AVG_BABIP,
        k_pct_vs_lhb=k_pct, bb_pct_vs_lhb=bb_pct,
        hr_per_fb_vs_lhb=hr_per_fb, babip_vs_lhb=config.LEAGUE_AVG_BABIP,
        k_pct_vs_rhb=k_pct, bb_pct_vs_rhb=bb_pct,
        hr_per_fb_vs_rhb=hr_per_fb, babip_vs_rhb=config.LEAGUE_AVG_BABIP,
        avg_pitches_per_start=c_pitches / max(starts, 1) if starts > 0 else 90,
        avg_innings_per_start=c_ip / max(starts, 1) if starts > 0 else 5.5,
        is_starter=starts > max(len(current), 1) * 0.3,
        is_closer=False,
        games_played=len(current) + len(prior),
        batters_faced_total=total_bf,
    )


# ---------------------------------------------------------------------------
# Team model from JSON with dynamic regression + Pythagorean + momentum
# ---------------------------------------------------------------------------

def _compute_team_prior_from_season(
    team_id: int,
    prior_season_games: list[dict],
) -> tuple[float, float, float]:
    """Compute a team's RPG, RAPG, and Pythag win% from a full prior season.

    Returns (prior_rpg, prior_rapg, prior_wpct) or league averages if no data.
    """
    games = []
    for g in prior_season_games:
        if g.get("home_id") == team_id or g.get("away_id") == team_id:
            is_home = g.get("home_id") == team_id
            rf = g["home_score"] if is_home else g["away_score"]
            ra = g["away_score"] if is_home else g["home_score"]
            games.append((rf, ra))

    if len(games) < 20:
        return config.LEAGUE_AVG_RUNS_PER_GAME, config.LEAGUE_AVG_RUNS_PER_GAME, 0.500

    rf_arr = np.array([g[0] for g in games], dtype=float)
    ra_arr = np.array([g[1] for g in games], dtype=float)

    rpg = float(rf_arr.mean())
    rapg = float(ra_arr.mean())

    exp = config.PYTHAG_EXPONENT
    total_rf = float(rf_arr.sum())
    total_ra = float(ra_arr.sum())
    wpct = total_rf**exp / (total_rf**exp + total_ra**exp) if total_rf + total_ra > 0 else 0.500

    return rpg, rapg, wpct


def build_team_model_from_json(
    team_id: int,
    team_name: str,
    all_games: list[dict],
    as_of_date: str,
    prior_season_games: list[dict] | None = None,
) -> TeamModel:
    """Build a team model from JSON game history.

    Uses prior-season data as the regression target when available,
    so the Dodgers regress toward their 2025 RPG (~5.5) instead of
    league average (4.39). This is critical in early season.
    """
    abbr = TEAM_ABBR.get(team_name, team_name[:3].upper())

    # Compute prior-season baseline (used as regression target)
    if prior_season_games:
        prior_rpg, prior_rapg, prior_wpct = _compute_team_prior_from_season(
            team_id, prior_season_games
        )
    else:
        prior_rpg = config.LEAGUE_AVG_RUNS_PER_GAME
        prior_rapg = config.LEAGUE_AVG_RUNS_PER_GAME
        prior_wpct = 0.500

    # Current season games before as_of_date
    prior_games = []
    for g in all_games:
        if g["game_date"] >= as_of_date:
            continue
        if g.get("home_id") == team_id or g.get("away_id") == team_id:
            is_home = g.get("home_id") == team_id
            runs_for = g["home_score"] if is_home else g["away_score"]
            runs_against = g["away_score"] if is_home else g["home_score"]
            won = runs_for > runs_against
            prior_games.append({
                "date": g["game_date"],
                "runs_for": runs_for,
                "runs_against": runs_against,
                "won": won,
            })

    prior_games.sort(key=lambda x: x["date"], reverse=True)

    if not prior_games:
        # No current-season data: use prior-season baseline entirely
        return TeamModel(
            team_id=team_id, team_abbr=abbr, season=2026,
            wins=0, losses=0,
            runs_per_game=prior_rpg,
            runs_allowed_per_game=prior_rapg,
            wrc_plus=(prior_rpg / config.LEAGUE_AVG_RUNS_PER_GAME) * 100,
            fip_minus=(prior_rapg / config.LEAGUE_AVG_RUNS_PER_GAME) * 100,
            team_k_pct=config.LEAGUE_AVG_K_PCT,
            team_bb_pct=config.LEAGUE_AVG_BB_PCT,
            team_hr_per_fb=config.LEAGUE_AVG_HR_PER_FB,
            team_babip=config.LEAGUE_AVG_BABIP,
            win_pct=prior_wpct,
        )

    n = len(prior_games)
    weights = np.exp(-config.TEAM_DECAY_LAMBDA * np.arange(n))
    w_sum = weights.sum()

    runs_for = np.array([g["runs_for"] for g in prior_games], dtype=float)
    runs_against = np.array([g["runs_against"] for g in prior_games], dtype=float)

    wins = sum(1 for g in prior_games if g["won"])
    losses = n - wins

    raw_rpg = float(np.dot(weights, runs_for) / w_sum)
    raw_rapg = float(np.dot(weights, runs_against) / w_sum)

    # Dynamic regression toward PRIOR SEASON (not league avg)
    if n <= 5:
        reg_n = 15
    elif n <= 15:
        reg_n = 30
    elif n <= 30:
        reg_n = 50
    else:
        reg_n = 80

    rpg = (n * raw_rpg + reg_n * prior_rpg) / (n + reg_n)
    rapg = (n * raw_rapg + reg_n * prior_rapg) / (n + reg_n)

    # Pythagorean win% from current season
    exp = config.PYTHAG_EXPONENT
    total_rf = float(runs_for.sum())
    total_ra = float(runs_against.sum())
    if total_rf + total_ra > 0:
        pythag_wpct = total_rf**exp / (total_rf**exp + total_ra**exp)
    else:
        pythag_wpct = 0.500

    # Regress Pythagorean toward prior-season win% (not .500)
    pythag_reg = 30
    win_pct = (n * pythag_wpct + pythag_reg * prior_wpct) / (n + pythag_reg)

    # Recent form: momentum from last 5 games
    last_5 = prior_games[:min(5, n)]
    if len(last_5) >= 3:
        last_5_margin = np.mean([g["runs_for"] - g["runs_against"] for g in last_5])
        momentum = last_5_margin * config.MOMENTUM_FACTOR
    else:
        momentum = 0.0

    rpg += momentum
    rapg -= momentum  # good recent form implies offense up, defense better

    # Clamp
    rpg = max(2.0, min(8.0, rpg))
    rapg = max(2.0, min(8.0, rapg))

    wrc_plus = (rpg / config.LEAGUE_AVG_RUNS_PER_GAME) * 100
    fip_minus = (rapg / config.LEAGUE_AVG_RUNS_PER_GAME) * 100

    return TeamModel(
        team_id=team_id, team_abbr=abbr, season=2026,
        wins=wins, losses=losses,
        runs_per_game=rpg, runs_allowed_per_game=rapg,
        wrc_plus=wrc_plus, fip_minus=fip_minus,
        team_k_pct=config.LEAGUE_AVG_K_PCT,
        team_bb_pct=config.LEAGUE_AVG_BB_PCT,
        team_hr_per_fb=config.LEAGUE_AVG_HR_PER_FB,
        team_babip=config.LEAGUE_AVG_BABIP,
        win_pct=win_pct,
    )


# ---------------------------------------------------------------------------
# Backtest evaluation
# ---------------------------------------------------------------------------

def evaluate_from_json(
    season_file: str = ".context/season_2026.json",
    prior_season_file: str | None = ".context/season_2025.json",
    start_date: str | None = None,
    end_date: str | None = None,
    n_sims: int = 2500,
    min_team_games: int = 3,
) -> BacktestResult:
    """Evaluate predictions against actual outcomes using JSON data."""
    with open(season_file) as f:
        all_games = json.load(f)

    # Load prior season data (2025) for priors
    prior_season_games = None
    prior_pitcher_index = None
    if prior_season_file and Path(prior_season_file).exists():
        with open(prior_season_file) as f:
            prior_season_games = json.load(f)
        prior_pitcher_index = _build_pitcher_index(prior_season_games)
        print(f"Loaded {len(prior_season_games)} prior-season games, {len(prior_pitcher_index)} pitchers")

    # Also load summaries for starter names
    summary_path = season_file.replace(".json", "_summary.json")
    starter_lookup = {}  # game_id -> (home_starter, away_starter)
    if Path(summary_path).exists():
        with open(summary_path) as f:
            summaries = json.load(f)
        for s in summaries:
            starter_lookup[s["game_id"]] = (
                s.get("home_starter", ""),
                s.get("away_starter", ""),
            )

    print(f"Loaded {len(all_games)} current-season games from {season_file}")

    # Build pitcher index for fast lookup
    pitcher_index = _build_pitcher_index(all_games)
    print(f"Indexed {len(pitcher_index)} current-season pitchers")

    park_factors = load_park_factors_from_json()
    neutral_park = ParkFactor(venue="Neutral", overall_factor=1.0,
                              hr_factor=1.0, h_factor=1.0, bb_factor=1.0)
    league_avg_pitcher = build_league_average_pitcher(is_starter=True)

    eval_games = all_games
    if start_date:
        eval_games = [g for g in eval_games if g["game_date"] >= start_date]
    if end_date:
        eval_games = [g for g in eval_games if g["game_date"] <= end_date]
    eval_games.sort(key=lambda g: g["game_date"])
    print(f"Evaluating {len(eval_games)} games")

    rng = np.random.default_rng(42)

    correct = 0
    brier_sum = 0.0
    margin_errors = []
    total_errors = []
    rl_correct = 0
    ou_correct = 0
    cal_buckets = {i: {"predicted": [], "actual": []} for i in range(10)}
    predictions = []
    skipped = 0
    pitcher_used = 0

    for i, g in enumerate(eval_games):
        if (i + 1) % 20 == 0:
            print(f"  Game {i + 1}/{len(eval_games)}...")

        home_id = g.get("home_id")
        away_id = g.get("away_id")
        game_date = g["game_date"]

        if not home_id or not away_id:
            skipped += 1
            continue

        # Teams: use league-avg regression (prior-season team RPG doesn't carry over well due to roster turnover)
        home_team = build_team_model_from_json(home_id, g["home_team"], all_games, game_date, None)
        away_team = build_team_model_from_json(away_id, g["away_team"], all_games, game_date, None)

        if home_team.wins + home_team.losses < min_team_games:
            skipped += 1
            continue
        if away_team.wins + away_team.losses < min_team_games:
            skipped += 1
            continue

        # Look up actual starters from summary data
        home_starter_name, away_starter_name = starter_lookup.get(g["game_id"], ("", ""))

        # Build pitcher-specific models (fall back to league avg)
        home_starter = league_avg_pitcher
        away_starter = league_avg_pitcher

        if home_starter_name:
            sp = build_pitcher_from_json(home_starter_name, pitcher_index, game_date, prior_pitcher_index)
            if sp:
                home_starter = sp
                pitcher_used += 1

        if away_starter_name:
            sp = build_pitcher_from_json(away_starter_name, pitcher_index, game_date, prior_pitcher_index)
            if sp:
                away_starter = sp
                pitcher_used += 1

        park = park_factors.get(TEAM_ABBR.get(g["home_team"], ""), neutral_park)

        results = simulate_n_box_score_games(
            home_team, away_team, home_starter, away_starter, park,
            n_sims=n_sims, rng=rng,
        )

        pred_home_wp = results["home_win_pct"]
        pred_spread = results["spread_mean"]
        pred_total = results["total_mean"]
        pred_home_cover = results["home_cover_pct"]

        actual_home_win = g["home_score"] > g["away_score"]
        actual_margin = g["home_score"] - g["away_score"]
        actual_total = g["home_score"] + g["away_score"]

        if (pred_home_wp > 0.5) == actual_home_win:
            correct += 1

        brier_sum += (pred_home_wp - (1.0 if actual_home_win else 0.0)) ** 2
        margin_errors.append(abs(pred_spread - actual_margin))
        total_errors.append(abs(pred_total - actual_total))

        if (pred_home_cover > 0.5) == (actual_margin >= 2):
            rl_correct += 1

        default_line = round(pred_total * 2) / 2
        over_prob = results["over_pct_by_line"].get(default_line, 0.5)
        if (over_prob > 0.5) == (actual_total > default_line):
            ou_correct += 1

        bucket = min(9, int(pred_home_wp * 10))
        cal_buckets[bucket]["predicted"].append(pred_home_wp)
        cal_buckets[bucket]["actual"].append(1.0 if actual_home_win else 0.0)

        predictions.append({
            "game_id": g["game_id"],
            "date": game_date,
            "home": g["home_team"],
            "away": g["away_team"],
            "pred_wp": f"{pred_home_wp:.1%}",
            "pred_spread": f"{pred_spread:+.1f}",
            "pred_total": f"{pred_total:.1f}",
            "actual": f"{g['away_score']}-{g['home_score']}",
            "correct": (pred_home_wp > 0.5) == actual_home_win,
        })

    n = len(predictions)
    if skipped:
        print(f"  Skipped {skipped} games (insufficient history)")
    print(f"  Used pitcher-specific models {pitcher_used} times ({pitcher_used}/{n*2} starters)")

    calibration = {}
    for bucket_idx, bucket_data in cal_buckets.items():
        if bucket_data["predicted"]:
            calibration[f"{bucket_idx*10}-{(bucket_idx+1)*10}%"] = {
                "avg_predicted": float(np.mean(bucket_data["predicted"])),
                "avg_actual": float(np.mean(bucket_data["actual"])),
                "count": len(bucket_data["predicted"]),
            }

    return BacktestResult(
        total_games=n,
        correct_picks=correct,
        accuracy=correct / n if n > 0 else 0,
        brier_score=brier_sum / n if n > 0 else 0,
        avg_margin_error=float(np.mean(margin_errors)) if margin_errors else 0,
        avg_total_error=float(np.mean(total_errors)) if total_errors else 0,
        rl_correct=rl_correct,
        rl_accuracy=rl_correct / n if n > 0 else 0,
        ou_correct=ou_correct,
        ou_accuracy=ou_correct / n if n > 0 else 0,
        calibration_by_decile=calibration,
        predictions=predictions,
    )


def print_backtest_report(result: BacktestResult) -> None:
    """Print a formatted backtest report."""
    print("\n" + "=" * 60)
    print("MLB SIMULATION BACKTEST v2")
    print("=" * 60)
    print(f"Games evaluated:   {result.total_games}")
    print()

    print("--- Winner Prediction ---")
    print(f"Correct picks:     {result.correct_picks}/{result.total_games}")
    print(f"Accuracy:          {result.accuracy:.1%}")
    print(f"Brier score:       {result.brier_score:.4f}")
    print()

    print("--- Spread/Margin ---")
    print(f"Avg margin error:  {result.avg_margin_error:.2f} runs")
    print()

    print("--- Run Line (-1.5) ---")
    print(f"Correct picks:     {result.rl_correct}/{result.total_games}")
    print(f"Accuracy:          {result.rl_accuracy:.1%}")
    print()

    print("--- Over/Under ---")
    print(f"Correct picks:     {result.ou_correct}/{result.total_games}")
    print(f"Accuracy:          {result.ou_accuracy:.1%}")
    print(f"Avg total error:   {result.avg_total_error:.2f} runs")
    print()

    print("--- Baselines ---")
    print(f"  Always home:     ~54.0%")
    print(f"  Random:          ~50.0%")
    print(f"  Model:           {result.accuracy:.1%}")

    if result.calibration_by_decile:
        print()
        print("--- Calibration ---")
        for bucket, data in sorted(result.calibration_by_decile.items()):
            n = data["count"]
            pred = data["avg_predicted"]
            actual = data["avg_actual"]
            delta = actual - pred
            print(f"  {bucket:>10}: pred {pred:.1%} → actual {actual:.1%} ({delta:+.1%}) n={n}")

    if result.predictions:
        print()
        print("--- Sample Predictions ---")
        for p in result.predictions[-10:]:
            mark = "✓" if p["correct"] else "✗"
            print(f"  {mark} {p['date']} {p['away']:>25} @ {p['home']:<25} "
                  f"Pred:{p['pred_wp']} Actual:{p['actual']}")

    print("=" * 60)


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else None
    end = sys.argv[2] if len(sys.argv) > 2 else None
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 2500

    result = evaluate_from_json(start_date=start, end_date=end, n_sims=sims)
    print_backtest_report(result)
