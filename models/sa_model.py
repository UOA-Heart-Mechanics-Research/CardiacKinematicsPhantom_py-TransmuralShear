# deformation_model.py

import numpy as np
import os
import pyvista as pv
import matplotlib.pyplot as plt
from ..utils.displacement_utils import  is_myocardium
from ..utils.coordinate_utils import cart2pol
from ..config import *
from .deformation_model import AnalyticalF, compute_deformation_mapping              

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_with_colored_grid(deformed_points, z_slice, t, output_path):
    """
    Plots a scatter plot of points and colors grid cells that contain at least one point.

    Parameters:
    - deformed_points: numpy array of shape (N, 2) containing the points to plot.
    - z_slice: int or float, Z coordinate slice for the title.
    - t: int, Time step for the title and file name.
    - output_path: str, Directory to save the output plot.
    """
   
    
    # Create the grid
    x_edges = np.arange(XLIM[0], XLIM[1] + HX, HX)
    y_edges = np.arange(YLIM[0], YLIM[1] + HY, HY)
    NvoxelX, NvoxelY = len(x_edges) - 1, len(y_edges) - 1

    # Create a 2D array to store whether a grid cell is occupied
    grid_occupied = np.zeros((NvoxelX, NvoxelY), dtype=bool)
    
    # Check each grid cell for points
    for i in range(NvoxelX):
        for j in range(NvoxelY):
            # Define the cell boundaries
            x_start, x_end = x_edges[i], x_edges[i + 1]
            y_start, y_end = y_edges[j], y_edges[j + 1]
            
            # Check if any point falls inside the current cell
            grid_occupied[i, j] = np.any(
                (deformed_points[:, 0] >= x_start) & (deformed_points[:, 0] < x_end) &
                (deformed_points[:, 1] >= y_start) & (deformed_points[:, 1] < y_end)
            )
    
    # Plot the scatter points
    plt.figure(figsize=(10, 10))
    #plt.scatter(deformed_points[:, 0], deformed_points[:, 1], color="gray")
    
    # Color the grid cells that are occupied
    for i in range(NvoxelX):
        for j in range(NvoxelY):
            if grid_occupied[i, j]:
                x_start, x_end = x_edges[i], x_edges[i + 1]
                y_start, y_end = y_edges[j], y_edges[j + 1]
                # Adjust cell boundaries to center points
                plt.gca().add_patch(
                    plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, color='gray', alpha=0.3)
                )
            else:
                x_start, x_end = x_edges[i], x_edges[i + 1]
                y_start, y_end = y_edges[j], y_edges[j + 1]
                #Edge only rectangles
                plt.gca().add_patch(
                    plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, fill=False, edgecolor='gray', alpha = 0.3)
                )
    
    # Add grid lines
    plt.grid(False)
    
    # Add labels and title
    plt.title(f"Short-Axis Slice at Z={z_slice}, Time Step {t}")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    
    # Set limits and save the figure
    plt.xlim(XLIM[0], XLIM[1])
    plt.ylim(YLIM[0], YLIM[1])
    plt.axis("equal")
    #Turn off the bounding box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    output_file = os.path.join(output_path, f"short_axis_grid_t{t:02d}.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def generate_short_axis_slice(z_slice, gridX, gridY, output_path):
    """
    Generates a short-axis slice of the phantom (containing only myocardium voxels)
    in the reference and deformed configurations for all time steps.
    In this function, the voxel positions mimic the DENSE image acquisition.
    Parameters:
        z_slice (float): Z-coordinate of the short-axis slice.
        gridX (np.ndarray): X-coordinates of the grid points.
        gridY (np.ndarray): Y-coordinates of the grid points.
        output_path (str): Directory path to save the results.

    Returns:
        deformed_points_all_times (np.ndarray): Deformed coordinates of the short-axis slice for all time steps. Shape (Nvoxels, Ntime, 3).
        None: Saves the short-axis slices in reference and deformed configurations to files.
    """
   
    # Prepare output directories
    output_path = os.path.join(output_path, f"Zpos_{z_slice}")
    os.makedirs(output_path, exist_ok=True)
    impages_output_path = os.path.join(output_path, "images")
    vtk_output_path = os.path.join(output_path, "vtk_files")
    np_output_path = os.path.join(output_path, "numpy_files")
    os.makedirs(impages_output_path, exist_ok=True)
    os.makedirs(vtk_output_path, exist_ok=True)
    os.makedirs(np_output_path, exist_ok=True)

    # Generate the reference grid (short-axis slice)
    gridX_ref, gridY_ref = np.meshgrid(gridX, gridY)
    gridZ_ref = np.full_like(gridX_ref, z_slice)  # All points lie in the same Z-plane

    # Flatten grid for easier computation
    ref_points = np.column_stack((gridX_ref.ravel(), gridY_ref.ravel(), gridZ_ref.ravel()))

    # Filter points to include only those in the myocardium
    myocardium_points = []
    for point in ref_points:
        if is_myocardium(point, R_ENDO, R_EPI, Z_BOTTOM, Z_TOP):
            myocardium_points.append(point)

    myocardium_points = np.array(myocardium_points)

    # Save reference configuration
    np.save(os.path.join(np_output_path, "reference_configuration.npy"), myocardium_points)

    x_min, x_max = np.min(myocardium_points[:, 0]), np.max(myocardium_points[:, 0])
    y_min, y_max = np.min(myocardium_points[:, 1]), np.max(myocardium_points[:, 1])

    deformed_points_all_times = []
    F_analytical = []
    
    # Loop through time steps and generate deformed configuration
    for t, alpha in enumerate(ALPHA_TIME):
        a_t = AFINAL * alpha
        deformed_points = []
        F_points = []
        for point in myocardium_points:
            Phi, R, Z = cart2pol(point[0], point[1], point[2])
            xa, ya, za = compute_deformation_mapping(Phi, R, Z, Z_BOTTOM, Z_TOP, a_t, cartesian=True)
            deformed_points.append([xa, ya, za])
            F_i = AnalyticalF(Phi, R, Z, Z_BOTTOM, Z_TOP, a_t)
            F_points.append(F_i)

        # Save deformed configuration for this time step
        deformed_points = np.array(deformed_points)
        deformed_points_all_times.append(deformed_points)
        F_points = np.array(F_points)
        F_analytical.append(F_points)
        displacement = deformed_points - myocardium_points
        np.save(os.path.join(np_output_path, f"sa_points_time_{t:02d}.npy"), deformed_points)
        np.save(os.path.join(np_output_path, f"displacement_time_{t:02d}.npy"), displacement)
        np.save(os.path.join(np_output_path, f"F_time_{t:02d}.npy"), F_points)

        plt.figure(figsize=(10, 10))
        plt.scatter(deformed_points[:, 0], deformed_points[:, 1], color="black")
        plt.title(f"Short-Axis Slice at Z={z_slice}, Time Step {t}")
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.xlim(x_min, x_max)  
        plt.ylim(y_min, y_max)  
        plt.axis("equal")
        plt.savefig(os.path.join(impages_output_path, f"short_axis_t{t:02d}.png"))
        plt.close()
        plot_with_colored_grid(deformed_points, z_slice, t, impages_output_path)

        # Save VTK file for visualization
        deformed_points_vtk = pv.PolyData(deformed_points)
        deformed_points_vtk.save(os.path.join(vtk_output_path, f"short_axis_t{t:02d}.vtk"))

    return np.array(deformed_points_all_times).transpose(1, 0, 2), np.array(F_analytical).transpose(1, 0, 2, 3)


def generate_multi_short_axis_slices(z_slices, N_radial, N_circumferential, output_path):
    """
    Generates multiple short-axis slices of the phantom (containing only myocardium voxels)
    with specified points in radial, circumferential, and Z-directions.

    Parameters:
        afinal (np.ndarray): Optimized deformation coefficients.
        alphaTime (np.ndarray): Array of time-dependent scaling factors for deformation.
        z_slice (np.ndarray): Z-coordinates for the short-axis slices.
        N_radial (int): Number of points in the radial direction.
        N_circumferential (int): Number of points in the circumferential direction.
        output_path (str): Directory path to save the results.

    Returns:
        None: Saves the short-axis slices in reference and deformed configurations to files.
    """

   # Prepare output directories
    output_path = os.path.join(output_path, f"Zpos_{z_slices}")
    os.makedirs(output_path, exist_ok=True)
    impages_output_path = os.path.join(output_path, "images")
    vtk_output_path = os.path.join(output_path, "vtk_files")
    np_output_path = os.path.join(output_path, "numpy_files")
    os.makedirs(impages_output_path, exist_ok=True)
    os.makedirs(vtk_output_path, exist_ok=True)
    os.makedirs(np_output_path, exist_ok=True)

    # Initialize storage for reference configurations
    reference_slices = []

    for z_slice in z_slices:
        # Generate the polar grid for the current slice
        radii = np.linspace(R_ENDO, R_EPI, N_radial)
        angles = np.linspace(0, 2 * np.pi, N_circumferential, endpoint=False)
        radii, angles = np.meshgrid(radii, angles)

        # Convert polar grid to Cartesian coordinates
        gridX_ref = radii * np.cos(angles)
        gridY_ref = radii * np.sin(angles)
        gridZ_ref = np.full_like(gridX_ref, z_slice)  # Fixed Z-coordinate

        # Flatten the grid for easier computation
        ref_points = np.column_stack((gridX_ref.ravel(), gridY_ref.ravel(), gridZ_ref.ravel()))

        # Append reference points for this slice
        reference_slices.append(ref_points)

    # Combine all slices into a single array and save
    reference_slices_combined = np.vstack(reference_slices)
    np.save(os.path.join(np_output_path, "reference_configuration.npy"), reference_slices_combined)

    all_F = []
    all_points = []
    # Loop through time steps and generate deformed configuration
    for t, alpha in enumerate(ALPHA_TIME):
        a_t = AFINAL * alpha
        deformed_slices = []
        F_slices = []
        for slice_points in reference_slices:
            deformed_points = []
            F_points = []
            for point in slice_points:
                Phi, R, Z = cart2pol(point[0], point[1], point[2])
                xa, ya, za = compute_deformation_mapping(Phi, R, Z, Z_BOTTOM, Z_TOP, a_t, cartesian=True)
                deformed_points.append([xa, ya, za])
                F_i = AnalyticalF(Phi, R, Z, Z_BOTTOM, Z_TOP, a_t)
                F_points.append(F_i)
            deformed_slices.append(deformed_points)
            F_slices.append(F_points)
            
        # Combine all deformed slices and save for this time step
        deformed_slices_combined = np.vstack(deformed_slices)
        F_slices = np.array(F_slices)
        displacement = deformed_slices_combined - reference_slices_combined
        np.save(os.path.join(np_output_path, f"deformed_configuration_t_{t:02d}.npy"), deformed_slices_combined)
        np.save(os.path.join(np_output_path, f"displacement_t_{t:02d}.npy"), displacement)
        np.save(os.path.join(np_output_path, f"F_t_{t:02d}.npy"), F_slices)
        # Save VTK file for visualization
        deformed_points_vtk = pv.PolyData(deformed_slices_combined)
        deformed_points_vtk.save(os.path.join(vtk_output_path, f"short_axis_t{t:02d}.vtk"))
        all_F.append(F_slices)
        all_points.append(deformed_slices_combined)

        plt.figure(figsize=(10, 10))
        plt.scatter(reference_slices_combined[:, 0], reference_slices_combined[:, 1], label="Reference", alpha=0.5)
        plt.scatter(deformed_slices_combined[:, 0], deformed_slices_combined[:, 1], label=f"Deformed t={t}", alpha=0.5)
        plt.legend()
        plt.title(f"Multi Short-Axis Slices, Time Step {t}")
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(os.path.join(impages_output_path, f"multi_short_axis_t{t:02d}.png"))
        plt.close()

    all_points = np.array(all_points)
    all_F = np.array(all_F)
    return all_points, all_F.transpose(1, 0, 2, 3, 4)

   
