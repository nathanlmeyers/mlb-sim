"""Engine agreement scoring for bet confidence.

Measures how much the box score and detailed engines agree.
High agreement = higher confidence in the prediction.
"""

import numpy as np


def compute_confidence(
    box_score_wp: float,
    detailed_wp: float,
    agreement_threshold: float = 0.05,
    max_disagreement: float = 0.20,
    floor: float = 0.20,
) -> float:
    """Compute confidence score from engine agreement.

    Args:
        box_score_wp: Home win probability from box score model
        detailed_wp: Home win probability from detailed model
        agreement_threshold: Disagreement below this = full confidence
        max_disagreement: Disagreement above this = floor confidence
        floor: Minimum confidence score

    Returns:
        Confidence score between floor and 1.0
    """
    disagreement = abs(box_score_wp - detailed_wp)

    if disagreement <= agreement_threshold:
        return 1.0
    elif disagreement >= max_disagreement:
        return floor
    else:
        # Linear interpolation
        t = (disagreement - agreement_threshold) / (max_disagreement - agreement_threshold)
        return 1.0 - t * (1.0 - floor)
