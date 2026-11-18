"""RSS-gated terminal novelty reward for second-stage HRAMA fine-tuning."""

from __future__ import division

from .metrics import REFERENCE_AVOIDED


def compute_diversity_reward(record, archive, lambda_div,
                             adversary_count):
    """Return a bounded team bonus and per-adversary terminal bonus.

    Only a successful ``rss_reference_avoided`` record can receive a bonus.
    Other collisions remain part of the original CR but are not used as
    high-confidence ADS-attributable failures for archive optimization.
    """

    adversary_count = int(adversary_count)
    if adversary_count <= 0:
        raise ValueError("adversary_count must be positive")
    lambda_div = float(lambda_div)
    if lambda_div < 0.0:
        raise ValueError("lambda_div must be non-negative")

    label = record.get("counterfactual_label")
    result = {
        "eligible": False,
        "counterfactual_label": label,
        "novelty_raw": 0.0,
        "novelty_normalized": 0.0,
        "team_bonus": 0.0,
        "per_adversary_bonus": 0.0,
        "nearest_prototype_id": None,
        "nearest_distance": None,
        "is_duplicate": False,
        "archive_version": archive.version,
        "archive_size": len(archive.prototypes),
    }
    if label != REFERENCE_AVOIDED:
        return result
    summary = record.get("original_summary")
    if not isinstance(summary, dict):
        return result

    query = archive.query(summary)
    team_bonus = lambda_div * query["normalized_novelty"]
    result.update({
        "eligible": True,
        "novelty_raw": query["raw_distance"],
        "novelty_normalized": query["normalized_novelty"],
        "team_bonus": team_bonus,
        "per_adversary_bonus": team_bonus / float(adversary_count),
        "nearest_prototype_id": query["nearest_prototype_id"],
        "nearest_distance": query["nearest_distance"],
        "is_duplicate": query["is_duplicate"],
        "archive_version": query["archive_version"],
        "archive_size": query["archive_size"],
        "_query_result": query,
    })
    return result


def public_reward_result(result):
    """Remove the NumPy vector used internally before writing logs/info."""

    return {
        key: value for key, value in result.items()
        if key != "_query_result"
    }

