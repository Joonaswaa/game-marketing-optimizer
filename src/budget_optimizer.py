"""SciPy budget allocation with diminishing returns and channel caps."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHANNEL_SHARE = 0.40
DEFAULT_SATURATION = 25_000.0


def channel_return(spend: float, base_roas: float, saturation: float) -> float:
    """
    Expected IAP return for a channel at a given spend level.

    Uses a log response curve so marginal ROAS falls as spend increases
    (audience saturation), unlike a naive linear ROAS * spend model.

    Args:
        spend: Dollars allocated to the channel.
        base_roas: Historical or predicted ROAS at moderate spend.
        saturation: Spend scale where diminishing returns kick in.

    Returns:
        Expected dollar return from this spend level.
    """
    if spend <= 0 or base_roas <= 0 or saturation <= 0:
        return 0.0
    return float(base_roas * saturation * np.log1p(spend / saturation))


def marginal_roas(spend: float, base_roas: float, saturation: float) -> float:
    """
    Marginal ROAS at the current spend level (derivative of channel_return).

    Args:
        spend: Dollars allocated to the channel.
        base_roas: Base ROAS multiplier.
        saturation: Saturation scale for the response curve.

    Returns:
        Incremental return per additional dollar spent.
    """
    if spend <= 0 or saturation <= 0:
        return base_roas
    return float(base_roas / (1.0 + spend / saturation))


def optimize_budget(
    total_budget: float,
    min_spends: dict[str, float],
    predicted_roas: dict[str, float],
    saturation_scales: dict[str, float] | None = None,
    max_channel_share: float = DEFAULT_MAX_CHANNEL_SHARE,
) -> dict[str, Any]:
    """
    Allocate ad budget across channels to maximize expected return.

    Unlike linear programming on constant ROAS, this model applies:
    - Log diminishing returns per channel (saturation)
    - Maximum share cap per channel (diversification)
    - Minimum spend constraints per channel

    Args:
        total_budget: Total budget to allocate in dollars.
        min_spends: Dict of channel -> minimum required spend.
        predicted_roas: Dict of channel -> predicted ROAS at moderate spend.
        saturation_scales: Optional channel -> saturation scale for response curve.
        max_channel_share: Maximum fraction of budget any single channel may receive.

    Returns:
        Dict containing allocations, percentages, expected_returns,
        marginal_roas, and total_expected_return.

    Raises:
        ValueError: If constraints are infeasible or optimization fails.
    """
    channels = list(predicted_roas.keys())
    n = len(channels)

    if total_budget <= 0:
        raise ValueError("Total budget must be positive")

    if not 0 < max_channel_share <= 1:
        raise ValueError("max_channel_share must be between 0 and 1")

    min_total = sum(min_spends.get(ch, 0.0) for ch in channels)
    if min_total > total_budget:
        raise ValueError(
            f"Minimum spend total (${min_total:,.0f}) exceeds budget (${total_budget:,.0f})"
        )

    max_cap = max_channel_share * total_budget
    if min_total > n * max_cap + 1e-6:
        raise ValueError(
            f"Minimum spends require more than {max_channel_share:.0%} per channel. "
            "Lower min spends or raise max channel share."
        )

    scales = saturation_scales or {}
    sat_array = np.array(
        [max(scales.get(ch, DEFAULT_SATURATION), 1_000.0) for ch in channels]
    )
    roas_array = np.array([max(predicted_roas[ch], 0.01) for ch in channels])
    min_array = np.array([min_spends.get(ch, 0.0) for ch in channels])
    max_array = np.full(n, max_cap)

    def objective(x: np.ndarray) -> float:
        returns = roas_array * sat_array * np.log1p(x / sat_array)
        return -float(np.sum(returns))

    constraints = [{"type": "eq", "fun": lambda x: float(np.sum(x) - total_budget)}]
    bounds = [(float(min_array[i]), float(max_array[i])) for i in range(n)]

    x0 = np.clip(
        np.full(n, total_budget / n),
        min_array,
        max_array,
    )
    if not np.isclose(x0.sum(), total_budget):
        x0 = min_array + (total_budget - min_array.sum()) * (
            (max_array - min_array) / (max_array - min_array).sum()
        )

    from scipy.optimize import minimize

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        raise ValueError(f"Budget optimization failed: {result.message}")

    allocations_arr = np.clip(result.x, min_array, max_array)
    allocations = dict(zip(channels, allocations_arr))
    percentages = {ch: (value / total_budget) * 100 for ch, value in allocations.items()}
    expected_returns = {
        ch: channel_return(allocations[ch], predicted_roas[ch], float(sat_array[i]))
        for i, ch in enumerate(channels)
    }
    marginal = {
        ch: marginal_roas(allocations[ch], predicted_roas[ch], float(sat_array[i]))
        for i, ch in enumerate(channels)
    }
    total_expected_return = sum(expected_returns.values())

    logger.info(
        "Optimized budget $%s — expected return $%.2f (max share %.0f%%)",
        f"{total_budget:,.0f}",
        total_expected_return,
        max_channel_share * 100,
    )

    return {
        "allocations": allocations,
        "percentages": percentages,
        "expected_returns": expected_returns,
        "marginal_roas": marginal,
        "total_expected_return": total_expected_return,
        "max_channel_share": max_channel_share,
    }


def monte_carlo_return_distribution(
    allocations: dict[str, float],
    predicted_roas: dict[str, float],
    saturation_scales: dict[str, float] | None = None,
    roas_std: dict[str, float] | None = None,
    n_simulations: int = 2000,
    seed: int = 42,
    default_roas_cv: float = 0.15,
) -> dict[str, Any]:
    """
    Simulate uncertainty in total and per-channel returns via ROAS noise.

    Each simulation draws channel ROAS from a normal distribution centered on
    the point estimate, then recomputes diminishing-return revenue. This
    produces empirical confidence bands (P10 / P50 / P90).

    Args:
        allocations: Optimized spend per channel.
        predicted_roas: Point-estimate ROAS per channel.
        saturation_scales: Saturation scale per channel.
        roas_std: Optional observed ROAS standard deviation per channel.
        n_simulations: Number of Monte Carlo draws.
        seed: Random seed for reproducibility.
        default_roas_cv: Coefficient of variation when channel std is unknown.

    Returns:
        Dict with percentile bands, per-channel bands, and raw total simulations.
    """
    rng = np.random.default_rng(seed)
    channels = list(allocations.keys())
    scales = saturation_scales or {}

    totals = np.zeros(n_simulations)
    channel_sims: dict[str, np.ndarray] = {ch: np.zeros(n_simulations) for ch in channels}

    for sim_idx in range(n_simulations):
        sim_total = 0.0
        for ch in channels:
            mean_roas = max(predicted_roas[ch], 0.01)
            if roas_std and ch in roas_std and roas_std[ch] > 0:
                std = float(roas_std[ch])
            else:
                std = mean_roas * default_roas_cv
            std = max(std, mean_roas * 0.05)

            sampled_roas = max(float(rng.normal(mean_roas, std)), 0.01)
            sat = max(scales.get(ch, DEFAULT_SATURATION), 1_000.0)
            ret = channel_return(allocations[ch], sampled_roas, sat)
            channel_sims[ch][sim_idx] = ret
            sim_total += ret
        totals[sim_idx] = sim_total

    def _percentiles(arr: np.ndarray) -> dict[str, float]:
        return {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    total_stats = _percentiles(totals)
    channel_bands = {ch: _percentiles(channel_sims[ch]) for ch in channels}

    logger.info(
        "Monte Carlo (%d sims): total return P10=$%.0f P50=$%.0f P90=$%.0f",
        n_simulations,
        total_stats["p10"],
        total_stats["p50"],
        total_stats["p90"],
    )

    return {
        "n_simulations": n_simulations,
        "total": total_stats,
        "simulated_totals": totals,
        "channel_bands": channel_bands,
    }


def optimize_budget_with_uncertainty(
    total_budget: float,
    min_spends: dict[str, float],
    predicted_roas: dict[str, float],
    saturation_scales: dict[str, float] | None = None,
    roas_std: dict[str, float] | None = None,
    max_channel_share: float = DEFAULT_MAX_CHANNEL_SHARE,
    n_simulations: int = 2000,
    seed: int = 42,
    default_roas_cv: float = 0.15,
) -> dict[str, Any]:
    """
    Run budget optimization then Monte Carlo uncertainty on the allocation.

    Args:
        total_budget: Total budget to allocate.
        min_spends: Minimum spend per channel.
        predicted_roas: Point-estimate ROAS per channel.
        saturation_scales: Saturation scales for response curves.
        roas_std: Optional ROAS standard deviation per channel from data.
        max_channel_share: Max fraction of budget per channel.
        n_simulations: Monte Carlo simulation count.
        seed: Random seed.
        default_roas_cv: Default ROAS CV when std is unavailable.

    Returns:
        Optimization result dict plus a ``uncertainty`` key with MC bands.
    """
    result = optimize_budget(
        total_budget,
        min_spends,
        predicted_roas,
        saturation_scales=saturation_scales,
        max_channel_share=max_channel_share,
    )
    result["uncertainty"] = monte_carlo_return_distribution(
        allocations=result["allocations"],
        predicted_roas=predicted_roas,
        saturation_scales=saturation_scales,
        roas_std=roas_std,
        n_simulations=n_simulations,
        seed=seed,
        default_roas_cv=default_roas_cv,
    )
    return result
