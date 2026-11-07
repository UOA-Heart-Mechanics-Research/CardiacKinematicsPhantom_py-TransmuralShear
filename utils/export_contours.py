"""Export analytic endo/epi contours of the phantom for DENSE analysis tools (e.g. mobstr3D).

The deformed radius r(R) = a0 + (1 + a1) R + a2 R^2 depends only on the reference radius and the
time scaling, not on z, so the endocardium (R_ENDO) and epicardium (R_EPI) are circles about the
phantom axis, identical on every slice. Contours are written in image coordinates matching the
DICOMs from export_to_dicom.py: x = column index, y = row index, zero-based, pixel centres at
integers.

Every contour vertex lies exactly on the circle, at the angle of a pixel centre near the 
boundary (see circle_ics). The resulting mask is exactly the set of pixels whose centre lies 
within the true wall, for any grid, coefficients or frame.

Output format (mobstr3D "DICOM_and_contour" input):
    {"endo": {"Slice_s": {"Frame_k": [[col, row], ...]}}, "epi": {...}}

Usage (from the repo root):
    python -m utils.export_contours --z 4 12 20 28 [--check-snr 10000]
"""

import argparse
import json
import os

import numpy as np

from config import AFINAL, ALPHA_TIME, HX, HY, R_ENDO, R_EPI, XLIM, YLIM
from utils.export_to_dicom import COLS, FIRST_PIXEL_XY, ROWS, snr_label

# Pixel centres of the image grid (mm), identical to the DICOM geometry in export_to_dicom.py
_COL_IDX, _ROW_IDX = np.meshgrid(np.arange(COLS), np.arange(ROWS))
PIXEL_X = FIRST_PIXEL_XY[0] + HX * _COL_IDX.ravel()
PIXEL_Y = FIRST_PIXEL_XY[1] + HY * _ROW_IDX.ravel()
PIXEL_R = np.hypot(PIXEL_X, PIXEL_Y)
PIXEL_THETA = np.arctan2(PIXEL_Y, PIXEL_X)
# Half a pixel diagonal: pixels with a centre within this distance of a boundary get their own vertex
EDGE_BAND = 0.5 * np.hypot(HX, HY)


def deformed_radius(R, a):
    """Deformed radius of reference radius R (see models.deformation_model.compute_deformation_mapping)."""
    return a[0] + (1.0 + a[1]) * R + a[2] * R**2


def max_sliver(radius, theta):
    """Largest gap between the circle and a polygon with vertices on the circle at the sorted angles theta."""
    gaps = np.diff(np.concatenate((theta, [theta[0] + 2.0 * np.pi])))
    return radius * (1.0 - np.cos(gaps.max() / 2.0)), gaps


def circle_ics(radius):
    """
    Closed polygon for a circle of the given radius (mm) about the phantom axis, in image coordinates
    [col, row], such that a pixel-centre-in-polygon test is exact.

    Vertices lie exactly on the circle, one on the radial line through every pixel centre within
    EDGE_BAND (half a pixel diagonal) of the boundary. The polygon is convex and star-shaped about
    the axis, and along the ray at a vertex's angle its boundary is exactly at `radius`, so each of
    those pixel centres is inside if r < radius and outside if r > radius - however close to the
    boundary it lies. Every other pixel centre is at least EDGE_BAND from the boundary, so it is
    classified correctly as long as the polygon's largest sliver off the circle (between neighbouring
    vertices) is smaller than EDGE_BAND; if a gap is ever too wide (very small radius or coarse grid)
    its midpoint angle is added until it is not.

    A pixel centre exactly on the circle (|r - radius| ~ 0) has no defined side and is left to the
    point-in-polygon test.
    """
    theta = np.unique(PIXEL_THETA[np.abs(PIXEL_R - radius) <= EDGE_BAND])
    if theta.size < 3:
        theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    sliver, gaps = max_sliver(radius, theta)
    while sliver >= EDGE_BAND:
        wide = gaps / 2.0 >= np.arccos(max(1.0 - EDGE_BAND / radius, -1.0))
        theta = np.unique(np.concatenate((theta, theta[wide] + gaps[wide] / 2.0)))
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        theta = np.unique(theta)
        sliver, gaps = max_sliver(radius, theta)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    col = (x - FIRST_PIXEL_XY[0]) / HX
    row = (y - FIRST_PIXEL_XY[1]) / HY
    pts = np.column_stack((col, row))
    return np.vstack((pts, pts[:1]))  # exactly closed


def analytic_contours(z_positions, coefficients=AFINAL):
    """Return {"endo": ..., "epi": ...} for all frames, with Slice_0 = highest z."""
    z_sorted = sorted((float(z) for z in z_positions), reverse=True)
    contours = {"endo": {}, "epi": {}}
    for s, _ in enumerate(z_sorted):
        slice_key = f"Slice_{s}"
        contours["endo"][slice_key] = {}
        contours["epi"][slice_key] = {}
        for k, alpha in enumerate(ALPHA_TIME):
            a = np.asarray(coefficients, dtype=float) * alpha
            frame_key = f"Frame_{k}"
            contours["endo"][slice_key][frame_key] = circle_ics(deformed_radius(R_ENDO, a)).tolist()
            contours["epi"][slice_key][frame_key] = circle_ics(deformed_radius(R_EPI, a)).tolist()
    return contours, z_sorted


def save_analytic_contours(z_positions, out_file="results/contours/contours_ics.json"):
    contours, z_sorted = analytic_contours(z_positions)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(contours, f)
    print(f"Saved {out_file}  (Slice_0..{len(z_sorted) - 1} = z {z_sorted} mm, {len(ALPHA_TIME)} frames)")
    return out_file, z_sorted


def plot_contour_check(contours, z_sorted, snr, out_file, frames=(0, int(np.argmax(ALPHA_TIME))), rep=0):
    """Overlay the contours on the exported magnitude (.mat) for a visual check."""
    import matplotlib.pyplot as plt
    import scipy.io as sio

    fig, axes = plt.subplots(len(frames), len(z_sorted), figsize=(3.2 * len(z_sorted), 3.2 * len(frames)),
                             squeeze=False)
    for s, z in enumerate(z_sorted):
        z_label = int(z) if float(z).is_integer() else z
        mat_dir = f"results/phasedata/matlab/Zpos_{z_label}/SNR_{snr_label(snr)}/Rep_{rep}"
        for r, k in enumerate(frames):
            ax = axes[r, s]
            mag = sio.loadmat(os.path.join(mat_dir, f"mag_{k + 1:02d}.mat"))["magnitude_wN"]
            ax.imshow(mag, cmap="gray", origin="upper")
            for name, colour in (("endo", "tab:red"), ("epi", "tab:cyan")):
                pts = np.asarray(contours[name][f"Slice_{s}"][f"Frame_{k}"])
                # line = polygon used for masking, markers = its vertices (one per edge-pixel angle)
                ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=0.8,
                        marker="o", ms=1.6, mfc=colour, mec="none")
            c = (np.mean(XLIM) - FIRST_PIXEL_XY[0]) / HX
            half = 1.25 * R_EPI / HX
            ax.set_xlim(c - half, c + half)
            ax.set_ylim(c + half, c - half)
            ax.set_title(f"Slice_{s} (z={z:g}) frame {k}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--z", type=float, nargs="+", required=True, help="Slice z positions (mm) in the stack.")
    parser.add_argument("--out", default="results/contours/contours_ics.json")
    parser.add_argument("--check-snr", default=None,
                        help="If given, overlay the contours on this SNR's magnitude and save a PNG next to the JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_file, z_sorted = save_analytic_contours(args.z, args.out)
    if args.check_snr is not None:
        snr = args.check_snr if args.check_snr == "inf" else float(args.check_snr)
        contours, _ = analytic_contours(args.z)
        png = os.path.splitext(out_file)[0] + f"_check_SNR{snr_label(snr)}.png"
        plot_contour_check(contours, z_sorted, snr, png)


if __name__ == "__main__":
    main()
