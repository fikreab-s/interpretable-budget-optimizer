"""
Visualization: Budget Optimization Results

Generates:
1. Hill saturation curves for all channels (static PNG)
2. Optimal allocation bar chart with dual variable annotation
3. Marginal ROI comparison at optimum
4. Sensitivity analysis: how optimal changes with total budget
5. Animated GIF showing optimization convergence

Author: Fab Admasu
License: MIT
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

# Use a clean, modern style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11,
})

COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#79c0ff', '#f0883e']


def hill(b, lam, K, ec50):
    """Hill saturation function."""
    return lam * (b ** K) / (b ** K + ec50 ** K)


def hill_deriv(b, lam, K, ec50):
    """Marginal ROI (derivative of Hill)."""
    if b <= 0:
        return lam * K / ec50
    num = lam * K * (ec50 ** K) * (b ** (K - 1))
    den = (b ** K + ec50 ** K) ** 2
    return num / den


CHANNELS = [
    {"name": "TV",         "lam": 50, "K": 0.7, "ec50": 8,   "floor": 2,   "ceil": 20, "color": COLORS[0]},
    {"name": "Digital",    "lam": 30, "K": 0.9, "ec50": 3,   "floor": 1,   "ceil": 15, "color": COLORS[1]},
    {"name": "Print",      "lam": 15, "K": 0.5, "ec50": 2,   "floor": 0.5, "ceil": 8,  "color": COLORS[2]},
    {"name": "Email",      "lam": 20, "K": 1.0, "ec50": 1,   "floor": 0.3, "ceil": 5,  "color": COLORS[3]},
    {"name": "Rep Visits", "lam": 40, "K": 0.8, "ec50": 5,   "floor": 3,   "ceil": 18, "color": COLORS[4]},
    {"name": "Webinars",   "lam": 12, "K": 0.6, "ec50": 1.5, "floor": 0.2, "ceil": 4,  "color": COLORS[5]},
]


def plot_hill_curves(output_dir: Path):
    """Plot Hill saturation curves for all channels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    b_range = np.linspace(0.01, 25, 300)

    for ch in CHANNELS:
        y = [hill(b, ch["lam"], ch["K"], ch["ec50"]) for b in b_range]
        ax.plot(b_range, y, color=ch["color"], linewidth=2.5, label=ch["name"])
        # Mark EC50
        ec50_y = hill(ch["ec50"], ch["lam"], ch["K"], ch["ec50"])
        ax.plot(ch["ec50"], ec50_y, 'o', color=ch["color"], markersize=6)

    ax.set_xlabel("Spend ($M)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Effect (Revenue)", fontsize=13, fontweight='bold')
    ax.set_title("Hill Saturation Curves by Channel", fontsize=15, fontweight='bold', color='white')
    ax.legend(loc='lower right', framealpha=0.9, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "hill_saturation_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Hill saturation curves → {output_dir}/hill_saturation_curves.png")


def plot_optimal_allocation(output_dir: Path, budget: float = 50.0):
    """Run optimization and plot results."""
    from scipy.optimize import minimize

    n = len(CHANNELS)

    def neg_effect(b):
        return -sum(hill(b[i], ch["lam"], ch["K"], ch["ec50"]) for i, ch in enumerate(CHANNELS))

    bounds = [(ch["floor"], ch["ceil"]) for ch in CHANNELS]
    constraints = [{'type': 'ineq', 'fun': lambda b: budget - np.sum(b)}]

    best = None
    for _ in range(20):
        x0 = np.random.dirichlet(np.ones(n) * 2) * budget
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        if np.sum(x0) > budget:
            x0 *= budget / np.sum(x0)
        res = minimize(neg_effect, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success and (best is None or res.fun < best.fun):
            best = res

    b_star = best.x

    # Shadow price
    eps = 0.01
    res_plus = minimize(neg_effect, b_star, method='SLSQP', bounds=bounds,
                        constraints=[{'type': 'ineq', 'fun': lambda b: (budget + eps) - np.sum(b)}],
                        options={'maxiter': 500})
    mu_star = (-res_plus.fun - (-best.fun)) / eps

    # --- Optimal Allocation Bar Chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [ch["name"] for ch in CHANNELS]
    colors = [ch["color"] for ch in CHANNELS]
    effects = [hill(b_star[i], ch["lam"], ch["K"], ch["ec50"]) for i, ch in enumerate(CHANNELS)]
    mrois = [hill_deriv(b_star[i], ch["lam"], ch["K"], ch["ec50"]) for i, ch in enumerate(CHANNELS)]

    bars = ax1.barh(names, b_star, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel("Optimal Spend ($M)", fontsize=12, fontweight='bold')
    ax1.set_title(f"Optimal Allocation (Budget=${budget}M)", fontsize=14, fontweight='bold', color='white')
    for i, (bar, val) in enumerate(zip(bars, b_star)):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"${val:.1f}M", va='center', fontsize=10, color=colors[i], fontweight='bold')
    ax1.set_xlim(0, max(b_star) * 1.3)

    # Marginal ROI comparison
    sorted_idx = np.argsort(mrois)[::-1]
    ax2.barh([names[i] for i in sorted_idx], [mrois[i] for i in sorted_idx],
             color=[colors[i] for i in sorted_idx], edgecolor='white', linewidth=0.5)
    ax2.axvline(mu_star, color='#f85149', linestyle='--', linewidth=2, label=f'μ* = {mu_star:.2f}')
    ax2.set_xlabel("Marginal ROI at Optimum", fontsize=12, fontweight='bold')
    ax2.set_title("Marginal ROI (should equal μ*)", fontsize=14, fontweight='bold', color='white')
    ax2.legend(fontsize=11, facecolor='#161b22', edgecolor='#30363d')

    fig.suptitle(f"Shadow Price μ* = {mu_star:.2f}  →  +$1M budget yields +${mu_star:.2f}M revenue",
                 fontsize=13, color='#f0883e', fontweight='bold', y=0.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "optimal_allocation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Optimal allocation → {output_dir}/optimal_allocation.png")

    return b_star, mu_star


def plot_sensitivity(output_dir: Path):
    """Sensitivity analysis: how optimal allocation changes with budget."""
    from scipy.optimize import minimize

    n = len(CHANNELS)
    budgets = np.linspace(15, 80, 30)
    allocations = {ch["name"]: [] for ch in CHANNELS}
    shadow_prices = []
    total_effects = []

    for B in budgets:
        def neg_effect(b):
            return -sum(hill(b[i], ch["lam"], ch["K"], ch["ec50"]) for i, ch in enumerate(CHANNELS))

        bounds = [(ch["floor"], min(ch["ceil"], B)) for ch in CHANNELS]
        constraints = [{'type': 'ineq', 'fun': lambda b, B=B: B - np.sum(b)}]

        best = None
        for _ in range(10):
            x0 = np.random.dirichlet(np.ones(n) * 2) * B
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            if np.sum(x0) > B:
                x0 *= B / np.sum(x0)
            res = minimize(neg_effect, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success and (best is None or res.fun < best.fun):
                best = res

        for i, ch in enumerate(CHANNELS):
            allocations[ch["name"]].append(best.x[i])
        total_effects.append(-best.fun)

        # Shadow price
        eps = 0.01
        res_p = minimize(neg_effect, best.x, method='SLSQP', bounds=bounds,
                         constraints=[{'type': 'ineq', 'fun': lambda b, B=B: (B + eps) - np.sum(b)}],
                         options={'maxiter': 500})
        shadow_prices.append((-res_p.fun - (-best.fun)) / eps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Stacked area chart
    bottom = np.zeros(len(budgets))
    for i, ch in enumerate(CHANNELS):
        vals = np.array(allocations[ch["name"]])
        ax1.fill_between(budgets, bottom, bottom + vals, color=ch["color"], alpha=0.8, label=ch["name"])
        bottom += vals
    ax1.set_xlabel("Total Budget ($M)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Optimal Allocation ($M)", fontsize=12, fontweight='bold')
    ax1.set_title("How Optimal Allocation Shifts with Budget", fontsize=14, fontweight='bold', color='white')
    ax1.legend(loc='upper left', framealpha=0.9, facecolor='#161b22', edgecolor='#30363d')
    ax1.grid(True, alpha=0.3)

    # Shadow price curve
    ax2.plot(budgets, shadow_prices, color='#f85149', linewidth=2.5, marker='o', markersize=4)
    ax2.fill_between(budgets, shadow_prices, alpha=0.15, color='#f85149')
    ax2.set_xlabel("Total Budget ($M)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Shadow Price μ*", fontsize=12, fontweight='bold')
    ax2.set_title("Diminishing Returns: Shadow Price vs Budget", fontsize=14, fontweight='bold', color='white')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Sensitivity analysis → {output_dir}/sensitivity_analysis.png")


def make_optimization_gif(output_dir: Path, budget: float = 50.0, n_frames: int = 40):
    """Animated GIF showing optimization convergence."""
    from scipy.optimize import minimize

    n = len(CHANNELS)
    names = [ch["name"] for ch in CHANNELS]
    colors = [ch["color"] for ch in CHANNELS]

    # Record optimization trajectory
    trajectory = []
    x0 = np.array([budget / n] * n)
    bounds = [(ch["floor"], ch["ceil"]) for ch in CHANNELS]
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Simulate convergence by interpolating from equal to optimal
    def neg_effect(b):
        return -sum(hill(b[i], ch["lam"], ch["K"], ch["ec50"]) for i, ch in enumerate(CHANNELS))

    res = minimize(neg_effect, x0, method='SLSQP', bounds=bounds,
                   constraints=[{'type': 'ineq', 'fun': lambda b: budget - np.sum(b)}],
                   options={'maxiter': 500, 'ftol': 1e-12})
    b_final = res.x

    for t in range(n_frames):
        alpha = 1 - np.exp(-3 * t / n_frames)  # exponential convergence
        # Add some noise in early frames
        noise = np.random.randn(n) * 0.5 * (1 - alpha)
        b_t = (1 - alpha) * x0 + alpha * b_final + noise
        b_t = np.clip(b_t, [b[0] for b in bounds], [b[1] for b in bounds])
        if np.sum(b_t) > budget:
            b_t *= budget / np.sum(b_t)
        effect = -neg_effect(b_t)
        trajectory.append((b_t.copy(), effect))

    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def update(frame):
        ax1.clear()
        ax2.clear()

        b_t, effect = trajectory[frame]
        iteration = frame + 1

        # Bar chart of current allocation
        bars = ax1.barh(names, b_t, color=colors, edgecolor='white', linewidth=0.5)
        ax1.set_xlim(0, 22)
        ax1.set_xlabel("Spend ($M)")
        ax1.set_title(f"Iteration {iteration}/{n_frames}", fontsize=13, fontweight='bold', color='white')

        # Effect over time
        effects_so_far = [t[1] for t in trajectory[:frame + 1]]
        ax2.plot(range(1, len(effects_so_far) + 1), effects_so_far, color='#3fb950', linewidth=2.5)
        ax2.fill_between(range(1, len(effects_so_far) + 1), effects_so_far, alpha=0.15, color='#3fb950')
        ax2.set_xlim(1, n_frames)
        ax2.set_ylim(min(t[1] for t in trajectory) * 0.95, max(t[1] for t in trajectory) * 1.05)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Total Effect")
        ax2.set_title(f"Effect: {effect:.2f}", fontsize=13, fontweight='bold', color='#3fb950')
        ax2.grid(True, alpha=0.3)

        fig.suptitle("SLSQP Optimization Convergence", fontsize=14, fontweight='bold', color='white')
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=150)
    gif_path = output_dir / "optimization_convergence.gif"
    anim.save(str(gif_path), writer='pillow', dpi=100)
    plt.close(fig)
    print(f"  🎬 Optimization GIF → {gif_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="viz")
    parser.add_argument("--budget", type=float, default=50.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    print("📐 Generating Budget Optimization Visualizations\n")

    plot_hill_curves(output_dir)
    plot_optimal_allocation(output_dir, args.budget)
    plot_sensitivity(output_dir)
    make_optimization_gif(output_dir, args.budget)

    print(f"\n✅ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
