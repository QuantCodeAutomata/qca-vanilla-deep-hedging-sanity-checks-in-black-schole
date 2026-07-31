"""Delta surface comparison utilities: learned vs analytical hedge."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


def compute_learned_hedge_surface(
    hedger,
    S_grid: np.ndarray,
    time_grid: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Evaluate the learned hedge at a grid of (spot, time-to-maturity) states.

    The state passed to the policy is [ttm, log(S), 0.0] where 0.0 is a
    placeholder for the previous hedge position.

    Args:
        hedger: DeepHedger (or any module with a `.policy` attribute) in eval mode.
        S_grid: 1-D array of spot values.
        time_grid: 1-D array of time-to-maturity values (normalised, 0..1).
        device: PyTorch device string.

    Returns:
        Hedge surface array of shape [len(time_grid), len(S_grid)].
    """
    hedger.eval()
    surface = np.zeros((len(time_grid), len(S_grid)))

    with torch.no_grad():
        for i, ttm in enumerate(time_grid):
            for j, s in enumerate(S_grid):
                state = torch.tensor(
                    [[ttm, np.log(s), 0.0]], dtype=torch.float32
                ).to(device)
                action = hedger.policy(state)
                surface[i, j] = action.item()

    return surface


def plot_delta_comparison(
    learned_surface: np.ndarray,
    analytical_surface: np.ndarray,
    S_grid: np.ndarray,
    time_grid: np.ndarray,
    title: str = "Hedge Surface Comparison",
    save_path: Optional[str] = None,
):
    """Plot learned vs analytical delta surface side by side with difference panel.

    Produces a three-panel figure:
        1. Analytical delta surface.
        2. Learned hedge surface.
        3. Difference (learned − analytical).

    Args:
        learned_surface: Hedge values from the neural policy [len(time_grid), len(S_grid)].
        analytical_surface: Reference delta values [len(time_grid), len(S_grid)].
        S_grid: 1-D spot grid (x-axis).
        time_grid: 1-D time-to-maturity grid (y-axis).
        title: Figure super-title.
        save_path: Optional file path; figure is saved as PNG when provided.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    S_mesh, T_mesh = np.meshgrid(S_grid, time_grid)

    vmin = min(learned_surface.min(), analytical_surface.min())
    vmax = max(learned_surface.max(), analytical_surface.max())

    im0 = axes[0].pcolormesh(S_mesh, T_mesh, analytical_surface,
                              vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[0].set_title("Analytical Delta")
    axes[0].set_xlabel("Spot S")
    axes[0].set_ylabel("Time to Maturity")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(S_mesh, T_mesh, learned_surface,
                              vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title("Learned Hedge")
    axes[1].set_xlabel("Spot S")
    axes[1].set_ylabel("Time to Maturity")
    plt.colorbar(im1, ax=axes[1])

    diff = learned_surface - analytical_surface
    im2 = axes[2].pcolormesh(S_mesh, T_mesh, diff, cmap="RdBu_r")
    axes[2].set_title("Difference (Learned − Analytical)")
    axes[2].set_xlabel("Spot S")
    axes[2].set_ylabel("Time to Maturity")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    return fig
