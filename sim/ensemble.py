"""Ensemble model combining box score and detailed simulation engines.

Blends multiple signals to produce a final prediction:
1. Box score model (team-level stats, fast)
2. Detailed PA-by-PA model (player-level, slower)
3. Record-based anchor (season W-L%)
4. Weather adjustments
5. Schedule adjustments
6. Probability calibration

Outputs full betting predictions: moneyline, run line, over/under.
"""

import numpy as np
from models.batter_model import BatterModel
from models.pitcher_model import PitcherModel
from models.team_model import TeamModel, ParkFactor
from models.calibration import logistic_calibrate
from models.weather import WeatherAdjustment
from models.schedule_features import ScheduleAdjustment
from betting.confidence import compute_confidence
from betting.predictions import BettingPrediction
from sim.game import simulate_n_box_score_games
from sim.detailed_game import simulate_n_detailed_games
import config


def predict_game(
    home_team: TeamModel,
    away_team: TeamModel,
    home_lineup: list[BatterModel],
    away_lineup: list[BatterModel],
    home_starter: PitcherModel,
    away_starter: PitcherModel,
    home_bullpen: list[PitcherModel],
    away_bullpen: list[PitcherModel],
    park: ParkFactor,
    weather: WeatherAdjustment | None = None,
    schedule_home: ScheduleAdjustment | None = None,
    schedule_away: ScheduleAdjustment | None = None,
    market_odds: dict | None = None,
    n_sims: int = config.DEFAULT_NUM_SIMS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate ensemble prediction with full betting outputs.

    Returns dict with prediction details and a BettingPrediction object.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Apply weather to park factor
    adjusted_park = park
    if weather:
        adjusted_park = ParkFactor(
            venue=park.venue,
            overall_factor=park.overall_factor,
            hr_factor=park.hr_factor * weather.hr_multiplier,
            h_factor=park.h_factor,
            bb_factor=park.bb_factor,
        )

    # 1. Box score model
    box_results = simulate_n_box_score_games(
        home_team, away_team, home_starter, away_starter, adjusted_park,
        n_sims=n_sims, rng=rng,
    )

    # 2. Detailed model (fewer sims since it's slower)
    detailed_sims = max(500, n_sims // 10)
    detailed_results = simulate_n_detailed_games(
        home_lineup, away_lineup,
        home_starter, away_starter,
        home_bullpen, away_bullpen,
        adjusted_park, n_sims=detailed_sims, rng=rng,
    )

    # 3. Blend simulation engines
    w_box = config.TRAINED_WEIGHT_BOX_SCORE
    w_detail = config.TRAINED_WEIGHT_DETAILED

    sim_home_wp = w_box * box_results["home_win_pct"] + w_detail * detailed_results["home_win_pct"]
    sim_home_cover = w_box * box_results["home_cover_pct"] + w_detail * detailed_results["home_cover_pct"]

    # Blend O/U probabilities
    over_prob = {}
    for line in box_results["over_pct_by_line"]:
        box_over = box_results["over_pct_by_line"].get(line, 0.5)
        det_over = detailed_results["over_pct_by_line"].get(line, 0.5)
        over_prob[line] = w_box * box_over + w_detail * det_over

    # Score predictions
    home_score_mean = w_box * box_results["home_score_mean"] + w_detail * detailed_results["home_score_mean"]
    away_score_mean = w_box * box_results["away_score_mean"] + w_detail * detailed_results["away_score_mean"]
    total_mean = home_score_mean + away_score_mean
    total_std = w_box * box_results.get("total_std", 3.5) + w_detail * detailed_results.get("total_std", 3.5)

    # 4. Apply schedule adjustments to scores
    if schedule_home:
        home_score_mean += schedule_home.total_adjustment
    if schedule_away:
        away_score_mean += schedule_away.total_adjustment
    if weather:
        total_mean += weather.total_adjustment

    # Recalculate total after adjustments
    total_mean = home_score_mean + away_score_mean

    # 5. Record-based anchor
    record_home_wp = _record_based_prediction(home_team, away_team)
    w_record = config.TRAINED_RECORD_ANCHOR
    blended_wp = (1 - w_record) * sim_home_wp + w_record * record_home_wp

    # 6. Calibrate
    calibrated_wp = logistic_calibrate(
        blended_wp, config.TRAINED_CALIBRATION_A, config.TRAINED_CALIBRATION_B
    )

    # 7. Compute confidence
    confidence = compute_confidence(
        box_results["home_win_pct"], detailed_results["home_win_pct"]
    )

    # 8. Blend with market odds (Kalshi) as Bayesian prior
    market_weight_applied = 0.0
    if market_odds and config.TRAINED_MARKET_WEIGHT > 0:
        market_wp = market_odds.get("home_win_prob")
        if market_wp and 0.05 < market_wp < 0.95:
            # Confidence-adjusted weight: trust model more when engines agree
            adj_mkt_wt = config.TRAINED_MARKET_WEIGHT * (
                1.0 - config.MARKET_CONFIDENCE_DISCOUNT * confidence
            )
            calibrated_wp = (1 - adj_mkt_wt) * calibrated_wp + adj_mkt_wt * market_wp
            calibrated_wp = float(np.clip(calibrated_wp, 0.02, 0.98))
            market_weight_applied = adj_mkt_wt

            # Blend total toward market total if available
            market_total = market_odds.get("total")
            if market_total and market_total > 0:
                total_mean = (1 - adj_mkt_wt) * total_mean + adj_mkt_wt * market_total

    # Build betting prediction
    prediction = BettingPrediction(
        home_win_prob=calibrated_wp,
        away_win_prob=1.0 - calibrated_wp,
        home_cover_prob=sim_home_cover,
        away_cover_prob=1.0 - sim_home_cover,
        total_mean=total_mean,
        total_std=total_std,
        over_prob=over_prob,
        confidence=confidence,
        engine_agreement=1.0 - abs(box_results["home_win_pct"] - detailed_results["home_win_pct"]),
        home_score_mean=home_score_mean,
        away_score_mean=away_score_mean,
    )

    return {
        "prediction": prediction,
        "box_score_home_wp": box_results["home_win_pct"],
        "detailed_home_wp": detailed_results["home_win_pct"],
        "record_home_wp": record_home_wp,
        "raw_blended_wp": blended_wp,
        "calibrated_wp": calibrated_wp,
        "confidence": confidence,
        "market_odds": market_odds,
        "market_weight_applied": market_weight_applied,
        "box_results": box_results,
        "detailed_results": detailed_results,
    }


def _record_based_prediction(home_team: TeamModel, away_team: TeamModel) -> float:
    """Predict home win probability from season records using log5."""
    a = home_team.win_pct
    b = away_team.win_pct

    if a + b == 0 or a + b == 2:
        return 0.5

    # Log5 formula
    p = (a - a * b) / (a + b - 2 * a * b)

    # Small home field boost
    p += 0.02

    return float(np.clip(p, 0.05, 0.95))
