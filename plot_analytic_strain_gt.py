"""Plot ground-truth cardiac strain directly from the analytical deformation.

No DENSE encoding, voxel averaging, phase wrapping, or noise is used here. The
six entries are components of the Green--Lagrange strain tensor in the local
(circumferential, longitudinal, radial) basis.  Shear entries are tensor shear
components (for example ECL), not engineering shear (2 * ECL).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from config import (
    AFINAL,
    ALPHA_TIME,
    R_ENDO,
    R_EPI,
    TIME_STEPS,
    Z_BOTTOM,
    Z_TOP,
    ZCONST,
)
from models.deformation_model import AnalyticalF


COMPONENT_NAMES = ("ECC", "ELL", "ERR", "ECL", "ECR", "ELR")


def local_green_lagrange_strain(phi, radius, z, coefficients):
    """Return [ECC, ELL, ERR, ECL, ECR, ELR] at one material point."""
    deformation_gradient = AnalyticalF(
        phi, radius, z, Z_BOTTOM, Z_TOP, coefficients
    )
    strain_cartesian = 0.5 * (
        deformation_gradient.T @ deformation_gradient - np.eye(3)
    )

    radial = np.array([np.cos(phi), np.sin(phi), 0.0])
    circumferential = np.array([-np.sin(phi), np.cos(phi), 0.0])
    longitudinal = np.array([0.0, 0.0, 1.0])
    local_basis = np.column_stack((circumferential, longitudinal, radial))
    strain_local = local_basis.T @ strain_cartesian @ local_basis

    components = np.array(
        [
            strain_local[0, 0],  # ECC
            strain_local[1, 1],  # ELL
            strain_local[2, 2],  # ERR
            strain_local[0, 1],  # ECL
            strain_local[0, 2],  # ECR
            strain_local[1, 2],  # ELR
        ]
    )
    components[np.abs(components) < 1.0e-14] = 0.0
    return components


def load_coefficients(path):
    if path is None:
        return np.asarray(AFINAL, dtype=float), "config.AFINAL"
    with open(path, "r", encoding="utf8") as coefficient_file:
        data = json.load(coefficient_file)
    return np.asarray(data["a_final"], dtype=float), path


def compute_peak_maps(coefficients, z, grid_size):
    coordinates = np.linspace(-R_EPI, R_EPI, grid_size)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radii = np.hypot(x_grid, y_grid)
    myocardium = (radii >= R_ENDO) & (radii <= R_EPI)
    peak_frame = int(np.argmax(ALPHA_TIME))
    peak_coefficients = coefficients * ALPHA_TIME[peak_frame]

    maps = np.full((*x_grid.shape, len(COMPONENT_NAMES)), np.nan)
    for row, column in np.argwhere(myocardium):
        phi = np.arctan2(y_grid[row, column], x_grid[row, column])
        maps[row, column] = local_green_lagrange_strain(
            phi, radii[row, column], z, peak_coefficients
        )
    return x_grid, y_grid, maps, peak_frame


def compute_transmural_curves(coefficients, z, radial_samples):
    radii = np.linspace(R_ENDO, R_EPI, radial_samples)
    strains = np.empty((TIME_STEPS, radial_samples, len(COMPONENT_NAMES)))
    for frame, alpha in enumerate(ALPHA_TIME):
        frame_coefficients = coefficients * alpha
        for radial_index, radius in enumerate(radii):
            # The analytical phantom is axisymmetric, so phi=0 is sufficient.
            strains[frame, radial_index] = local_green_lagrange_strain(
                0.0, radius, z, frame_coefficients
            )
    return radii, strains


def plot_peak_maps(x_grid, y_grid, maps, peak_frame, z, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for component_index, (axis, name) in enumerate(
        zip(axes.flat, COMPONENT_NAMES)
    ):
        values = maps[..., component_index]
        max_abs = np.nanmax(np.abs(values))
        is_zero = max_abs < 1.0e-14
        color_limit = 1.0e-6 if is_zero else max_abs
        image = axis.pcolormesh(
            x_grid,
            y_grid,
            values,
            shading="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.set_xlabel("X [mm]")
        axis.set_ylabel("Y [mm]")
        if is_zero:
            axis.text(
                0.5,
                0.06,
                "identically zero",
                transform=axis.transAxes,
                ha="center",
                fontsize=8,
            )
        fig.colorbar(image, ax=axis, shrink=0.82, label="strain")
    fig.suptitle(
        f"Analytical GT Green–Lagrange strain at peak frame {peak_frame} "
        f"(z = {z:g} mm)"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_curves(strains, z, output_path):
    frames = np.arange(TIME_STEPS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for component_index, (axis, name) in enumerate(
        zip(axes.flat, COMPONENT_NAMES)
    ):
        values = strains[..., component_index]
        mean = values.mean(axis=1)
        axis.fill_between(
            frames,
            values.min(axis=1),
            values.max(axis=1),
            color="0.78",
            label="transmural range",
        )
        axis.plot(frames, mean, color="black", linewidth=1.8, label="wall mean")
        axis.axhline(0.0, color="0.45", linewidth=0.7)
        axis.set_title(name)
        axis.set_xlabel("cardiac frame")
        axis.set_ylabel("strain")
        axis.grid(alpha=0.2)
        if np.max(np.abs(values)) < 1.0e-14:
            axis.set_ylim(-0.01, 0.01)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Analytical GT strain over time (z = {z:g} mm; shaded = endo-to-epi range)"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_peak_transmural(radii, strains, z, output_path):
    peak_frame = int(np.argmax(ALPHA_TIME))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for component_index, (axis, name) in enumerate(
        zip(axes.flat, COMPONENT_NAMES)
    ):
        values = strains[peak_frame, :, component_index]
        axis.plot(radii, values, color="black", linewidth=2.0)
        axis.axhline(0.0, color="0.45", linewidth=0.7)
        axis.set_title(name)
        axis.set_xlabel("radius [mm]")
        axis.set_ylabel("strain")
        axis.grid(alpha=0.2)
        axis.set_xlim(radii[0], radii[-1])
        if np.max(np.abs(values)) < 1.0e-14:
            axis.set_ylim(-0.01, 0.01)
            axis.text(
                0.5,
                0.9,
                "identically zero",
                transform=axis.transAxes,
                ha="center",
                fontsize=8,
            )
    fig.suptitle(
        f"Analytical GT strain across the wall at peak frame {peak_frame} "
        f"(z = {z:g} mm)"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_peak_grouped_strains(radii, strains, z, output_path):
    peak_frame = int(np.argmax(ALPHA_TIME))
    peak_strains = strains[peak_frame]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    paired = plt.get_cmap("Paired").colors

    normal_components = (
        (0, "ECC", paired[3]),  # dark green
        (1, "ELL", paired[7]),  # dark orange
        (2, "ERR", paired[9]),  # dark purple
    )
    for component_index, label, color in normal_components:
        axes[0].plot(
            radii,
            peak_strains[:, component_index],
            color=color,
            linewidth=2.2,
            label=label,
        )
    axes[0].set_title("Normal strains")
    axes[0].set_ylim(-0.3, 0.6)

    shear_components = (
        (3, "ECL", paired[5]),  # dark red
        (4, "ECR", paired[1]),  # dark blue
        (5, "ELR", paired[11]), # dark brown
    )
    for component_index, label, color in shear_components:
        axes[1].plot(
            radii,
            peak_strains[:, component_index],
            color=color,
            linewidth=2.2,
            label=label,
        )
    axes[1].set_title("Shear strains")
    axes[1].set_ylim(-0.05, 0.1)

    for axis in axes:
        axis.axhline(0.0, color="0.45", linewidth=0.7)
        axis.set_xlim(radii[0], radii[-1])
        axis.set_box_aspect(1)
        axis.set_xlabel("radius [mm]")
        axis.set_ylabel("strain")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)

    fig.suptitle(
        f"Analytical GT strain across the wall at peak frame {peak_frame} "
        f"(z = {z:g} mm)"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficients",
        help="Optional a_final.json; defaults to AFINAL in config.py.",
    )
    parser.add_argument("--z", type=float, default=ZCONST, help="Slice Z in mm.")
    parser.add_argument("--grid-size", type=int, default=241)
    parser.add_argument("--radial-samples", type=int, default=101)
    parser.add_argument("--output-dir", default="results/analytic_strain_gt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Z_BOTTOM <= args.z <= Z_TOP:
        raise ValueError(f"--z must be between {Z_BOTTOM:g} and {Z_TOP:g} mm")
    if args.grid_size < 3 or args.radial_samples < 2:
        raise ValueError("--grid-size must be >= 3 and --radial-samples must be >= 2")

    coefficients, coefficient_source = load_coefficients(args.coefficients)
    os.makedirs(args.output_dir, exist_ok=True)

    x_grid, y_grid, maps, peak_frame = compute_peak_maps(
        coefficients, args.z, args.grid_size
    )
    radii, strains = compute_transmural_curves(
        coefficients, args.z, args.radial_samples
    )

    peak_plot = os.path.join(args.output_dir, "strain_components_peak.png")
    time_plot = os.path.join(args.output_dir, "strain_components_over_time.png")
    transmural_plot = os.path.join(
        args.output_dir, "strain_components_peak_transmural.png"
    )
    grouped_plot = os.path.join(
        args.output_dir, "strain_components_peak_grouped.png"
    )
    data_path = os.path.join(args.output_dir, "analytic_strain_gt.npz")
    plot_peak_maps(x_grid, y_grid, maps, peak_frame, args.z, peak_plot)
    plot_time_curves(strains, args.z, time_plot)
    plot_peak_transmural(radii, strains, args.z, transmural_plot)
    plot_peak_grouped_strains(radii, strains, args.z, grouped_plot)
    np.savez_compressed(
        data_path,
        component_names=np.asarray(COMPONENT_NAMES),
        coefficients=coefficients,
        coefficient_source=coefficient_source,
        alpha_time=ALPHA_TIME,
        z=args.z,
        radii=radii,
        strains=strains,
        x=x_grid,
        y=y_grid,
        peak_frame=peak_frame,
        peak_maps=maps,
    )

    print(f"Saved {peak_plot}")
    print(f"Saved {time_plot}")
    print(f"Saved {transmural_plot}")
    print(f"Saved {grouped_plot}")
    print(f"Saved {data_path}")


if __name__ == "__main__":
    main()
