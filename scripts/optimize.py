"""
Constrained Budget Optimizer with Hill Saturation

Solves: max Σᵢ fᵢ(bᵢ) s.t. Σbᵢ ≤ B, Lᵢ ≤ bᵢ ≤ Uᵢ
where fᵢ(b) = λᵢ · b^Kᵢ / (b^Kᵢ + EC50ᵢ^Kᵢ)

Outputs: optimal allocation, dual variables, sensitivity analysis.

Author: Fab Admasu
License: MIT
"""

import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


@dataclass
class ChannelParams:
    """Hill saturation parameters for a single channel."""
    name: str
    lam: float       # λ: maximum effect (ceiling)
    K: float          # Hill coefficient (steepness)
    ec50: float       # Half-maximal effective concentration
    floor: float      # Minimum spend constraint
    ceiling: float    # Maximum spend constraint
    adstock: float    # Geometric decay rate θ


@dataclass
class OptResult:
    """Optimization result with dual variables."""
    optimal_allocation: Dict[str, float]
    total_effect: float
    budget_shadow_price: float  # μ* = ∂f*/∂B
    marginal_rois: Dict[str, float]
    channel_effects: Dict[str, float]
    active_floors: List[str]
    active_ceilings: List[str]


def hill_function(b: float, lam: float, K: float, ec50: float) -> float:
    """Hill saturation: f(b) = λ · b^K / (b^K + EC50^K)"""
    if b <= 0:
        return 0.0
    return lam * (b ** K) / (b ** K + ec50 ** K)


def hill_gradient(b: float, lam: float, K: float, ec50: float) -> float:
    """Marginal ROI: ∂f/∂b = λK · EC50^K · b^(K-1) / (b^K + EC50^K)²"""
    if b <= 0:
        return lam * K / ec50  # limit as b→0
    num = lam * K * (ec50 ** K) * (b ** (K - 1))
    den = (b ** K + ec50 ** K) ** 2
    return num / den


def optimize_budget(
    channels: List[ChannelParams],
    total_budget: float,
    n_restarts: int = 10,
) -> OptResult:
    """
    Solve the constrained nonlinear budget optimization.
    Uses multi-start SLSQP for global search.
    """
    n = len(channels)

    def neg_total_effect(b):
        """Negative total effect (minimize = maximize effect)."""
        return -sum(
            hill_function(b[i], ch.lam, ch.K, ch.ec50)
            for i, ch in enumerate(channels)
        )

    def neg_total_gradient(b):
        """Gradient of negative total effect."""
        return np.array([
            -hill_gradient(b[i], ch.lam, ch.K, ch.ec50)
            for i, ch in enumerate(channels)
        ])

    # Bounds
    bounds = [(ch.floor, min(ch.ceiling, total_budget)) for ch in channels]

    # Budget constraint: Σbᵢ ≤ B
    constraints = [{
        'type': 'ineq',
        'fun': lambda b: total_budget - np.sum(b),
        'jac': lambda b: -np.ones(n),
    }]

    # Multi-start optimization
    best_result = None
    best_value = float('inf')

    for restart in range(n_restarts):
        # Random feasible starting point
        if restart == 0:
            # Equal allocation start
            x0 = np.full(n, total_budget / n)
        else:
            # Random Dirichlet start
            weights = np.random.dirichlet(np.ones(n) * 2)
            x0 = weights * total_budget

        # Clip to bounds
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        # Scale to budget
        if np.sum(x0) > total_budget:
            x0 = x0 * total_budget / np.sum(x0)

        result = minimize(
            neg_total_effect,
            x0,
            method='SLSQP',
            jac=neg_total_gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12},
        )

        if result.success and result.fun < best_value:
            best_value = result.fun
            best_result = result

    if best_result is None:
        raise RuntimeError("Optimization failed on all restarts")

    b_star = best_result.x

    # Extract dual variable (shadow price of budget)
    # Approximate via finite difference: μ* ≈ [f*(B+ε) - f*(B)] / ε
    eps = 0.01
    result_plus = minimize(
        neg_total_effect, b_star, method='SLSQP',
        jac=neg_total_gradient, bounds=bounds,
        constraints=[{'type': 'ineq', 'fun': lambda b: (total_budget + eps) - np.sum(b)}],
        options={'maxiter': 500, 'ftol': 1e-12},
    )
    shadow_price = (-result_plus.fun - (-best_result.fun)) / eps

    # Compute marginal ROIs
    marginal_rois = {
        ch.name: round(hill_gradient(b_star[i], ch.lam, ch.K, ch.ec50), 4)
        for i, ch in enumerate(channels)
    }

    # Channel effects
    effects = {
        ch.name: round(hill_function(b_star[i], ch.lam, ch.K, ch.ec50), 4)
        for i, ch in enumerate(channels)
    }

    # Active constraints
    active_floors = [ch.name for i, ch in enumerate(channels)
                     if abs(b_star[i] - ch.floor) < 1e-4]
    active_ceilings = [ch.name for i, ch in enumerate(channels)
                       if abs(b_star[i] - ch.ceiling) < 1e-4]

    return OptResult(
        optimal_allocation={ch.name: round(float(b_star[i]), 4)
                           for i, ch in enumerate(channels)},
        total_effect=round(-best_result.fun, 4),
        budget_shadow_price=round(shadow_price, 4),
        marginal_rois=marginal_rois,
        channel_effects=effects,
        active_floors=active_floors,
        active_ceilings=active_ceilings,
    )


def default_channels() -> List[ChannelParams]:
    """Default pharma promotional channels with realistic parameters."""
    return [
        ChannelParams("TV", lam=50.0, K=0.7, ec50=8.0, floor=2.0, ceiling=20.0, adstock=0.7),
        ChannelParams("Digital", lam=30.0, K=0.9, ec50=3.0, floor=1.0, ceiling=15.0, adstock=0.2),
        ChannelParams("Print", lam=15.0, K=0.5, ec50=2.0, floor=0.5, ceiling=8.0, adstock=0.4),
        ChannelParams("Email", lam=20.0, K=1.0, ec50=1.0, floor=0.3, ceiling=5.0, adstock=0.1),
        ChannelParams("Rep_Visits", lam=40.0, K=0.8, ec50=5.0, floor=3.0, ceiling=18.0, adstock=0.5),
        ChannelParams("Webinars", lam=12.0, K=0.6, ec50=1.5, floor=0.2, ceiling=4.0, adstock=0.3),
    ]


def main():
    parser = argparse.ArgumentParser(description="Constrained budget optimization")
    parser.add_argument("--budget", type=float, default=50.0, help="Total budget in $M")
    parser.add_argument("--channels", type=int, default=6, help="Number of channels")
    parser.add_argument("--output", type=str, default="outputs/optimization_results.json")
    args = parser.parse_args()

    channels = default_channels()[:args.channels]
    print(f"🎯 Optimizing ${args.budget}M across {len(channels)} channels\n")

    result = optimize_budget(channels, args.budget)

    print("═" * 60)
    print("OPTIMAL BUDGET ALLOCATION")
    print("═" * 60)
    print(f"\n{'Channel':<15} {'Allocation ($M)':>15} {'Effect':>10} {'Marginal ROI':>12}")
    print("-" * 55)
    for ch in channels:
        alloc = result.optimal_allocation[ch.name]
        effect = result.channel_effects[ch.name]
        mroi = result.marginal_rois[ch.name]
        floor_flag = " [floor]" if ch.name in result.active_floors else ""
        ceil_flag = " [ceiling]" if ch.name in result.active_ceilings else ""
        print(f"{ch.name:<15} ${alloc:>13.2f} {effect:>10.2f} {mroi:>12.4f}{floor_flag}{ceil_flag}")

    print(f"\n{'Total Effect:':<30} {result.total_effect:.4f}")
    print(f"{'Budget Shadow Price (μ*):':<30} {result.budget_shadow_price:.4f}")
    print(f"\n💡 μ* = {result.budget_shadow_price:.2f} means each additional $1M of budget")
    print(f"   yields ${result.budget_shadow_price:.2f}M in additional effect at the optimum.")

    if result.active_floors:
        print(f"\n⚠️  Floor-constrained: {', '.join(result.active_floors)}")
    if result.active_ceilings:
        print(f"⚠️  Ceiling-constrained: {', '.join(result.active_ceilings)}")

    # Save results
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\n📁 Results saved to {args.output}")


if __name__ == "__main__":
    main()
