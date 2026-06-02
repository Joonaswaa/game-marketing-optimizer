"""SciPy linear programming wrapper for ad budget allocation."""

import logging
from typing import Any

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


def optimize_budget(
    total_budget: float,
    min_spends: dict[str, float],
    predicted_roas: dict[str, float],
) -> dict[str, Any]:
    """
    Allocate ad budget across channels to maximize expected ROAS.

    Uses scipy.optimize.linprog with method='highs' (NOT simplex).
    Minimizes negative ROAS (equivalent to maximizing ROAS).

    Args:
        total_budget: Total budget to allocate in dollars.
        min_spends: Dict of channel -> minimum required spend.
        predicted_roas: Dict of channel -> predicted ROAS multiplier from ML models.

    Returns:
        Dict containing allocations, percentages, expected_returns,
        and total_expected_return.

    Raises:
        ValueError: If SciPy optimization fails (infeasible constraints).
    """
    channels = list(predicted_roas.keys())
    n = len(channels)

    if total_budget <= 0:
        raise ValueError("Total budget must be positive")

    min_total = sum(min_spends.get(ch, 0.0) for ch in channels)
    if min_total > total_budget:
        raise ValueError(
            f"Minimum spend total (${min_total:,.0f}) exceeds budget (${total_budget:,.0f})"
        )

    c = [-predicted_roas[ch] for ch in channels]
    A_ub = [np.ones(n)]
    b_ub = [total_budget]
    bounds = [(min_spends.get(ch, 0.0), total_budget) for ch in channels]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise ValueError(f"Budget optimization failed: {result.message}")

    allocations = dict(zip(channels, result.x))
    percentages = {ch: (value / total_budget) * 100 for ch, value in allocations.items()}
    expected_returns = {
        ch: predicted_roas[ch] * allocations[ch] for ch in channels
    }
    total_expected_return = sum(expected_returns.values())

    logger.info(
        "Optimized budget $%s across %d channels — expected return $%.2f",
        f"{total_budget:,.0f}",
        n,
        total_expected_return,
    )

    return {
        "allocations": allocations,
        "percentages": percentages,
        "expected_returns": expected_returns,
        "total_expected_return": total_expected_return,
    }
