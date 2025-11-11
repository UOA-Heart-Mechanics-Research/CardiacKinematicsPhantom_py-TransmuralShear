
import numpy as np
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import PIL.Image as Image

# Import configuration and utility functions
from config import *
from utils.mesh_utils import phantom_hexa_mesh, phase_mesh
from utils.displacement_utils import eulerian_disp, is_myocardium
from utils.vtk_utils import plot_to_vtk, rescale_image
from models.deformation_model import AnalyticalPhantom, Microstructure
from models.optimization_model import OptimizeDisp, target_strain_error, EulerianMap
from models.noise_model import add_noise_to_data_dense, check_snr
import time
from scipy.optimize import fsolve, root
import json
import scipy.io as sio
from utils.export_to_dicom import save_dense_dicoms


################################################### Main function
def main():
    # Start timer
    StartTime = time.time()
    # Ensure output directories exist
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    os.makedirs(VTK_PATH, exist_ok=True)
    os.makedirs(PHASE_FILE_PATH, exist_ok=True)
    os.makedirs(IMAGE_PATH, exist_ok=True)
    # os.makedirs(SA_SLICE_PATH, exist_ok=True)

    # Clear any previous VTK files
    if Path(VTK_PATH).exists():
        shutil.rmtree(VTK_PATH)
    os.makedirs(VTK_PATH)

    phantom_data = {}
    
    # Compute optimized displacement parameters
    if COMPUTE_PHANTOM:
        a_final = OptimizeDisp(R_ENDO, R_EPI, Z_BOTTOM, Z_TOP, THETA_ENDO, THETA_MID, THETA_EPI, TARGET_STRAINS, WEIGHTS, DEBUG_FLAG)
        print(f'a_final={a_final}')
        with open(os.path.join(BASE_OUTPUT_PATH, 'a_final.json'), 'w') as f:
            json.dump({'a_final': a_final.tolist()}, f)
    else:
        a_final = AFINAL

    # Generate grid points
    gridX = np.arange(XLIM[0], XLIM[1] + HX, HX)
    gridY = np.arange(YLIM[0], YLIM[1] + HY, HY)
    NvoxelX, NvoxelY = len(gridX) - 1, len(gridY) - 1
    xPhase, connPhase = phase_mesh(gridX, gridY, ZCONST)

    phantom_data['NvoxelX'] = NvoxelX
    phantom_data['NvoxelY'] = NvoxelY
    phantom_data['TimeSteps'] = TIME_STEPS
    phantom_data['hx'] = HX
    phantom_data['hy'] = HY
    phantom_data['ke_x'] = KE_X
    phantom_data['ke_y'] = KE_Y
    phantom_data['ke_z'] = KE_Z
    phantom_data['SNR'] = SNR
    phantom_data['reps'] = REPS
    
    # Save phantom data as JSON file
    with open(os.path.join(PHASE_FILE_PATH, 'phantom_data.json'), 'w') as f:
        json.dump(phantom_data, f)
    
    sio.savemat(os.path.join(PHASE_FILE_PATH, 'phantom_data.mat'), phantom_data)
    
    if SAVE_PHANTOM_VTK:
        # Visualize displacement in VTK format
        AnalyticalPhantom(a_final, R_ENDO, R_EPI, Z_BOTTOM, Z_TOP, PHANTOM_MESH_SIZE, NPOINT_Z, NPOINT_R, NPOINT_C, THETA_ENDO, THETA_MID, THETA_EPI, ALPHA_TIME, VTK_PATH, DEBUG_FLAG)

    # Parameters
    tol = 5.0e-4
    options = {'maxiter': 10000, 'maxfev': 10000, 'xtol': tol}
    Zg = np.linspace(ZCONST - 0.5 * HZ, ZCONST + 0.5 * HZ, SAMPLE_Z * 2 + 1)[1::2]

    # Initialize storage for magnitude and phase data
    mag_phase = np.zeros((NvoxelY, NvoxelX, 4, len(ALPHA_TIME)))

    # Loop over time steps
    for s, alpha in enumerate(ALPHA_TIME):
        # print(f'\n\nTime step {s + 1}/{TIME_STEPS}\n')
        dispAllX = np.zeros((NvoxelY * SAMPLE_Y, NvoxelX * SAMPLE_X, SAMPLE_Z)) # Row (first array) is the vertical direction (y direction), column (second array) is the horizontal direction (x direction)
        dispAllY = np.zeros((NvoxelY * SAMPLE_Y, NvoxelX * SAMPLE_X, SAMPLE_Z)) # Row (first array) is the vertical direction (y direction), column (second array) is the horizontal direction (x direction)
        dispAllZ = np.zeros((NvoxelY * SAMPLE_Y, NvoxelX * SAMPLE_X, SAMPLE_Z)) # Row (first array) is the vertical direction (y direction), column (second array) is the horizontal direction (x direction)
        # print(f'dispAllX.shape={dispAllX.shape}, dispAllY.shape={dispAllY.shape}, dispAllZ.shape={dispAllZ.shape}')

        magnitude = np.zeros((NvoxelY, NvoxelX)) # Row (first array) is the vertical direction (y direction), column (second array) is the horizontal direction (x direction)
        phaseX = np.zeros((NvoxelY, NvoxelX)) # Row (first array) is the vertical direction (y direction), column (second array) is the horizontal direction (x direction)
        phaseY = np.zeros((NvoxelY, NvoxelX)) # Row (first array) is the vertical direction (y direction), column (second array) is the horizontal direction (x direction)
        phaseZ = np.zeros((NvoxelY, NvoxelX)) # Row (first array) is the vertical direction (y direction), column (second array) is the horizontal direction (x direction)
        # print(f'magnitude.shape={magnitude.shape}, phaseX.shape={phaseX.shape}, phaseY.shape={phaseY.shape}, phaseZ.shape={phaseZ.shape}')

        for column in range(NvoxelX): # Loop over columns (horizontal direction)
            Xg = np.linspace(gridX[column], gridX[column + 1], SAMPLE_X * 2 + 1)[1::2] #Sample points per voxel
            for row in range(NvoxelY): # Loop over rows (vertical direction)
                Yg = np.linspace(gridY[row], gridY[row + 1], SAMPLE_Y * 2 + 1)[1::2] #Sample points per voxel
                NphaseToAverage = 0

                for is_ in range(SAMPLE_X): # Loop over sample points in x direction
                        for js in range(SAMPLE_Y): # Loop over sample points in y direction
                            for ks in range(SAMPLE_Z): # Loop over sample points in z direction
                                
                                xgiven_coordinate = np.array([Xg[is_], Yg[js], Zg[ks]]) # As this one stands for real coordinates, Xg[is_] is the x coordinate of the sample point, Yg[js] is the y coordinate of the sample point, and Zg[ks] is the z coordinate of the sample point
                                
                                if np.linalg.norm(xgiven_coordinate[:2]) > RMIN:
                                    indRow = row * SAMPLE_Y + js
                                    indColumn = column * SAMPLE_X + is_
                                    indDepth = ks

                                    X0 = xgiven_coordinate - np.array([
                                        dispAllX[indRow, indColumn, indDepth],
                                        dispAllY[indRow, indColumn, indDepth],
                                        dispAllZ[indRow, indColumn, indDepth]
                                    ])

                                    a_s = a_final * alpha
                                    beta_s = BETA_MAX * alpha

                                    def eulerian_wrapper(X):
                                        return EulerianMap(X, xgiven_coordinate, Z_BOTTOM, Z_TOP, a_s)
                                    
                                    #Xval = fsolve(eulerian_wrapper, X0, xtol=tol)
                                    sol = root(eulerian_wrapper, X0, method ='hybr') #Find the coordinates of the point in the reference configuration
                                    # Check if the solver was successful
                                    if sol.success:
                                        Xval = sol.x  # Solution
                                    else:
                                        # print("Root finding did not converge:", sol.message)
                                        pass
                                    
                                    if np.linalg.norm(eulerian_wrapper(Xval)) < tol and is_myocardium(Xval, R_ENDO, R_EPI, Z_BOTTOM, Z_TOP):
                                        
                                        # Compute displacement
                                        dispXsingle = xgiven_coordinate[0] - Xval[0]
                                        dispYsingle = xgiven_coordinate[1] - Xval[1]
                                        dispZsingle = xgiven_coordinate[2] - Xval[2]

                                        # Store displacement
                                        dispAllX[indRow, indColumn, indDepth] = dispXsingle
                                        dispAllY[indRow, indColumn, indDepth] = dispYsingle
                                        dispAllZ[indRow, indColumn, indDepth] = dispZsingle

                                        # Accumulate phase and magnitude
                                        phaseX[row, column] += dispXsingle
                                        phaseY[row, column] += dispYsingle
                                        phaseZ[row, column] += dispZsingle
                                        NphaseToAverage += 1

                                        magnitude[row, column] += 1.0
                                    else:
                                        dispAllX[indRow, indColumn, indDepth] = 0
                                        dispAllY[indRow, indColumn, indDepth] = 0
                                        dispAllZ[indRow, indColumn, indDepth] = 0

                magnitude[row, column] /= N_POINTS_PER_VOXEL
                if NphaseToAverage == 0:
                    NphaseToAverage = 1

                phaseX[row, column] *= KE_X / NphaseToAverage
                phaseY[row, column] *= KE_Y / NphaseToAverage
                phaseZ[row, column] *= KE_Z / NphaseToAverage

                # mag phase storage
                mag_phase[row, column, 0, s] = magnitude[row, column]
                mag_phase[row, column, 1, s] = phaseX[row, column]
                mag_phase[row, column, 2, s] = phaseY[row, column]
                mag_phase[row, column, 3, s] = phaseZ[row, column]

        for rep in range(REPS):
            for snr_i in SNR:
                # Wrap phase values to be within [-0.5, 0.5]
                phaseX = (phaseX + 0.5) % 1 - 0.5
                phaseY = (phaseY + 0.5) % 1 - 0.5
                phaseZ = (phaseZ + 0.5) % 1 - 0.5
                # Simple-Four-point
                magnitude_wN, phaseX_wN, phaseY_wN, phaseZ_wN = add_noise_to_data_dense(
                    magnitude, phaseX, phaseY, phaseZ, snr_i, enc_str = ENC_STR
                )
                
                phaseX_wN = (phaseX_wN + 0.5) % 1 - 0.5
                phaseY_wN = (phaseY_wN + 0.5) % 1 - 0.5
                phaseZ_wN = (phaseZ_wN + 0.5) % 1 - 0.5

                CellScalars = np.zeros((NvoxelX * NvoxelY, 4))
                CellScalars[:, 0] = magnitude_wN.flatten()
                CellScalars[:, 1] = phaseX_wN.flatten()
                CellScalars[:, 2] = phaseY_wN.flatten()
                CellScalars[:, 3] = phaseZ_wN.flatten()

                snr_label = "inf" if snr_i == np.inf else f"{int(snr_i):02d}"
                snr_img_folder = os.path.join(IMAGE_PATH, f"SNR_{snr_label}")
                os.makedirs(snr_img_folder, exist_ok=True)
                snr_img_folder = os.path.join(snr_img_folder, ENC_STR)
                os.makedirs(snr_img_folder, exist_ok=True)
                if SAVE_PHANTOM_VTK:
                    vtk_file_name = f"Phantom_Zpos_{ZCONST}_Rep_{rep}_SNR_{snr_label}_Time_{s:02d}"
                    vtk_file_name = os.path.join(VTK_PATH, vtk_file_name)
                    plot_to_vtk(
                        xt = xPhase, conn=connPhase - 1, el_type=PHASE_EL_TYPE, node_scalars=np.array([]), node_scalars_names=np.array([]),
                        cell_scalars=CellScalars, cell_scalars_names=CELL_SCALARS_NAME, output_file_name=vtk_file_name, visualize=False, 
                        binary=False, file_format='vtu'
                    )
                
                # Visualization (debugging)
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                for ax, data, title in zip(
                    axes.flat,
                    [magnitude_wN, phaseX_wN, phaseY_wN, phaseZ_wN],
                    ["Magnitude", "X [cycles]", "Y [cycles]", "Z [cycles]"]
                ):
                    ax.imshow(rescale_image(data), cmap='seismic')
                    ax.set_title(title)
                    ax.axis('off')
                fig.tight_layout()
                fig.savefig(f"{snr_img_folder}/Zpos_{ZCONST}_Rep_{rep}_SNR_{snr_label}_Time_{s:02d}.png", dpi=300)
                plt.close()
                #Save each phase and magnitude in a separate image using PIL
                rescaled_images = [rescale_image(data) for data in [magnitude_wN, phaseX_wN, phaseY_wN, phaseZ_wN]]
                for i, rescaled_data in enumerate(rescaled_images):
                    image = Image.fromarray((rescaled_data * 255).astype(np.uint8))
                    # Rescale the image to 300x300 pixels
                    image = image.resize((300, 300), Image.NEAREST)
                    image.save(f"{snr_img_folder}/Zpos_{ZCONST}_Rep_{rep}_SNR_{snr_label}_Time_{s:02d}_{CELL_SCALARS_NAME[i]}.png", dpi=(300, 300), cmap='seismic')
                
                if CHECK_SNR:
                    snr_values = check_snr(magnitude, phaseX, phaseY, phaseZ, snr_i, enc_str=ENC_STR)
                    # Visualize the SNR map
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(snr_values, cmap='cividis')
                    cbar = fig.colorbar(im, ax=ax, label='SNR')
                    cbar.ax.set_yticks([snr_values.min(), snr_values.max()])
                    cbar.ax.set_yticklabels([f"{snr_values.min():.2f}", f"{snr_values.max():.2f}"])
                    ax.set_title('Per-Pixel SNR Map')
                    fig.savefig(f"{snr_img_folder}/SNR_MAP_Zpos_{ZCONST}_Rep_{rep}_SNR_{snr_label}_Time_{s:02d}.png", dpi=300, bbox_inches='tight')
                    plt.close()
                # Save data
                output_path_temp = os.path.join(
                    PHASE_FILE_PATH, f"Rep_{rep}", f"Zpos_{ZCONST}", f"SNR_{snr_label}"
                )
                
                os.makedirs(output_path_temp, exist_ok=True)
                np.save(os.path.join(output_path_temp, f"phaseX_{s:02d}.npy"), phaseX_wN)
                np.save(os.path.join(output_path_temp, f"phaseY_{s:02d}.npy"), phaseY_wN)
                np.save(os.path.join(output_path_temp, f"phaseZ_{s:02d}.npy"), phaseZ_wN)
                np.save(os.path.join(output_path_temp, f"mag_{s:02d}.npy"), magnitude_wN)
                #Save as matfiles
                output_path_temp = os.path.join(
                    PHASE_FILE_PATH, "matlab", f"Zpos_{ZCONST}", f"SNR_{snr_label}", f"Rep_{rep}"
                )
                os.makedirs(output_path_temp, exist_ok=True)
                sio.savemat(os.path.join(output_path_temp, f"phaseX_{s+1:02d}.mat"), {'phaseX_wN': phaseX_wN})
                sio.savemat(os.path.join(output_path_temp, f"phaseY_{s+1:02d}.mat"), {'phaseY_wN': phaseY_wN})
                sio.savemat(os.path.join(output_path_temp, f"phaseZ_{s+1:02d}.mat"), {'phaseZ_wN': phaseZ_wN})
                sio.savemat(os.path.join(output_path_temp, f"mag_{s+1:02d}.mat"), {'magnitude_wN': magnitude_wN})

    # Save data as dicoms
    save_dense_dicoms()     # Call the function to save dense dicoms from export_to_dicom.py (all default parameters are set in config.py)
    print("DICOM files saved successfully.")

    # Print elapsed time
    print(f"Elapsed time = {time.time() - StartTime}")

if __name__ == "__main__":
    main()
