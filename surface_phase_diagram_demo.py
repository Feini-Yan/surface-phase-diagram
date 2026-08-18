#!/usr/bin/env python3
"""
surface_phase_diagram_demo.py

Illustrative workflow for constructing a two-dimensional surface phase diagram
from first-principles total energies.

All numerical values are synthetic and are included only to demonstrate the
analysis workflow. They do not correspond to any unpublished dataset.

Workflow:
1. Define candidate surface structures and total energies.
2. Evaluate surface energies over a chemical-potential grid.
3. Identify the lowest-energy phase at each grid point.
4. Plot the resulting stability map.

Requirements:
    numpy
    matplotlib
"""

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


@dataclass(frozen=True)
class SurfaceStructure:
    """Candidate surface structure used in the phase-stability analysis."""
    name: str
    n_substrate: int
    n_x: int
    n_y: int
    total_energy: float
    area: float


# Synthetic bulk chemical potentials (eV/atom)
MU_SUBSTRATE_BULK = -5.80
MU_X_BULK = -4.10
MU_Y_BULK = -3.50


# Synthetic candidate structures.
# The values are chosen only to generate a visually clear example phase map.
STRUCTURES: List[SurfaceStructure] = [
    SurfaceStructure("Phase A", 40, 2, 2, -212.283738, 100.0),
    SurfaceStructure("Phase B", 40, 5, 2, -232.692608, 100.0),
    SurfaceStructure("Phase C", 40, 2, 5, -229.986594, 100.0),
    SurfaceStructure("Phase D", 40, 6, 5, -254.764069, 100.0),
    SurfaceStructure("Phase E", 40, 5, 7, -257.292190, 100.0),
    SurfaceStructure("Phase F", 40, 8, 3, -251.442829, 100.0),
]


# Light macaron palette
PHASE_COLORS = [
    "#B9DDF2",  # powder blue
    "#C9E7C4",  # mint
    "#F7D8A8",  # apricot
    "#DCC8F0",  # lavender
    "#F6C6CF",  # blush pink
    "#C7E6E2",  # pale aqua
]


def surface_energy(
    structure: SurfaceStructure,
    delta_mu_x: np.ndarray,
    delta_mu_y: np.ndarray,
) -> np.ndarray:
    """
    Calculate the surface energy:

        gamma = [E_tot - sum_i(N_i * mu_i)] / A

    with
        mu_X = mu_X_bulk + delta_mu_x
        mu_Y = mu_Y_bulk + delta_mu_y
    """
    mu_x = MU_X_BULK + delta_mu_x
    mu_y = MU_Y_BULK + delta_mu_y

    return (
        structure.total_energy
        - structure.n_substrate * MU_SUBSTRATE_BULK
        - structure.n_x * mu_x
        - structure.n_y * mu_y
    ) / structure.area


def compute_stability_map(
    structures: List[SurfaceStructure],
    delta_mu_x: np.ndarray,
    delta_mu_y: np.ndarray,
) -> np.ndarray:
    """Return the index of the lowest-energy phase at every grid point."""
    x_grid, y_grid = np.meshgrid(delta_mu_x, delta_mu_y, indexing="xy")

    energies = np.stack(
        [surface_energy(s, x_grid, y_grid) for s in structures],
        axis=0,
    )

    return np.argmin(energies, axis=0)


def plot_phase_diagram(
    structures: List[SurfaceStructure],
    delta_mu_x: np.ndarray,
    delta_mu_y: np.ndarray,
    stability_map: np.ndarray,
    output_file: str = "surface_phase_diagram_demo.png",
) -> None:
    """Plot the phase map using a soft pastel palette."""
    present_indices = np.unique(stability_map)
    cmap = ListedColormap(PHASE_COLORS[: len(structures)])

    fig, ax = plt.subplots(figsize=(7.4, 5.5))

    fill_levels = np.arange(len(structures) + 1) - 0.5
    ax.contourf(
        delta_mu_x,
        delta_mu_y,
        stability_map,
        levels=fill_levels,
        cmap=cmap,
        alpha=0.96,
    )

    boundary_levels = np.arange(len(structures) - 1) + 0.5
    ax.contour(
        delta_mu_x,
        delta_mu_y,
        stability_map,
        levels=boundary_levels,
        colors="#666666",
        linewidths=0.75,
    )

    ax.set_xlabel(r"$\Delta\mu_X$ (eV)", fontsize=12)
    ax.set_ylabel(r"$\Delta\mu_Y$ (eV)", fontsize=12)
    ax.set_title("Illustrative Surface Phase Diagram", fontsize=13)

    legend_handles = [
        Patch(
            facecolor=PHASE_COLORS[i],
            edgecolor="#888888",
            linewidth=0.6,
            label=structures[i].name,
        )
        for i in present_indices
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.tick_params(width=0.8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the phase-stability workflow."""
    delta_mu_x = np.linspace(-3.0, 0.0, 600)
    delta_mu_y = np.linspace(-3.0, 0.0, 600)

    stability_map = compute_stability_map(
        STRUCTURES,
        delta_mu_x,
        delta_mu_y,
    )

    present_indices = np.unique(stability_map)
    print("Stable phases in the sampled chemical-potential range:")
    for idx in present_indices:
        print(f"  - {STRUCTURES[idx].name}")

    plot_phase_diagram(
        STRUCTURES,
        delta_mu_x,
        delta_mu_y,
        stability_map,
    )


if __name__ == "__main__":
    main()
