# noise_model.py

import numpy as np
import matplotlib.pyplot as plt

def dis_encoding_strategie(encoding_strategie):
    """
    Returns the transformation matrices for the displacement encoding strategies.
    Options are 'Balanced-Four-point' and 'Simple-Four-point'.
    """
    if encoding_strategie =='Balanced-Four-point':
        # Constants for phase encoding transformation
        sqrt3over3 = np.sqrt(3.0) / 3.0
        A = np.array([[-sqrt3over3, -sqrt3over3, -sqrt3over3, 1],
                    [ sqrt3over3,  sqrt3over3, -sqrt3over3, 1],
                    [ sqrt3over3, -sqrt3over3,  sqrt3over3, 1],
                    [-sqrt3over3,  sqrt3over3,  sqrt3over3, 1]])
        sqrt3over4 = np.sqrt(3.0) / 4.0
        Ainv = np.array([[-sqrt3over4,  sqrt3over4,  sqrt3over4, -sqrt3over4],
                        [-sqrt3over4,  sqrt3over4, -sqrt3over4,  sqrt3over4],
                        [-sqrt3over4, -sqrt3over4,  sqrt3over4,  sqrt3over4],
                        [ 0.25,  0.25,  0.25,  0.25]])
    elif encoding_strategie == 'Simple-Four-point':
        A = np.array([[0, 0, 0, 1],
                     [ 1,  0, 0, 1],
                     [ 0, 1,  0, 1],
                     [0,  0,  1, 1]])
        Ainv = np.array([[-1, 1, 0, 0],
                         [-1, 0, 1, 0],
                         [-1, 0, 0, 1],
                         [1, 0, 0, 0]])
       
    else:
        raise ValueError(f"Unknown Displacement Encoding Strategies: {encoding_strategie}")
    
    return A, Ainv

def add_noise_to_data_dense(magnitude, phaseX, phaseY, phaseZ, SNR, enc_str='Balanced-Four-point'):
    """
    Adds noise to magnitude and phase data to simulate different SNR levels.
    
    Parameters:
        magnitude (np.ndarray): Original magnitude data.
        phaseX, phaseY, phaseZ (np.ndarray): Original phase data for each axis.
        SNR (float): Desired signal-to-noise ratio. Use `np.inf` for no noise.
        enc_str (str): Encoding strategy for displacement encoding. Options are 'Balanced-Four-point' and 'Simple-Four-point'.
    Returns:
        tuple: Arrays with noise added - magnitude_wN, phaseX_wN, phaseY_wN, phaseZ_wN.
    """
    # Get dimensions of the input data
    Nx, Ny = magnitude.shape

    # Encoding strategy matrices
    A, Ainv = dis_encoding_strategie(enc_str)

    # Convert SNR to noise standard deviation
    NoiseStandardDeviation = 2.0 / SNR if SNR != np.inf else 0.0

    # Generate complex noise arrays for all phase components
    noise_shape = (Nx, Ny, 4)
    noise = NoiseStandardDeviation * (np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape))

    # Stack the input phases into an array
    phases = np.stack([phaseX, phaseY, phaseZ, np.zeros_like(phaseX)], axis=-1)

    # Calculate original phase encoded as complex numbers
    phi_ab = 2 * np.pi * np.einsum('ij,...j->...i', A, phases)  # Shape: (Nx, Ny, 4)
    S_ab = magnitude[..., None] * (np.cos(phi_ab) + 1j * np.sin(phi_ab))  # Shape: (Nx, Ny, 4)

    # Add complex noise
    Swn = S_ab + noise  # Shape: (Nx, Ny, 4)

    # Decode noisy phases
    phi_ab_wn = np.angle(Swn)  # Shape: (Nx, Ny, 4)
    phi_ab_wn = np.round((phi_ab - phi_ab_wn) / (2 * np.pi)) + phi_ab_wn / (2 * np.pi)

    # Apply inverse transformation to get noisy phases in X, Y, Z directions
    D_phi_b = np.einsum('ij,...j->...i', Ainv, phi_ab_wn)  # Shape: (Nx, Ny, 4)

    # Extract noisy phase components
    phaseX_wN = D_phi_b[..., 0]
    phaseY_wN = D_phi_b[..., 1]
    phaseZ_wN = D_phi_b[..., 2]

    # Compute magnitude with noise
    magnitude_wN = np.mean(np.abs(Swn), axis=-1)  # Shape: (Nx, Ny)

    return magnitude_wN, phaseX_wN, phaseY_wN, phaseZ_wN    



def rescale_image(img):
    """
    Rescales an image array to the range [0, 1] for visualization purposes.

    Parameters:
        img (np.ndarray): Input image array.
    
    Returns:
        np.ndarray: Rescaled image array with values between 0 and 1.
    """
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)

def check_snr(magnitude, phaseX, phaseY, phaseZ, SNR, enc_str='Balanced-Four-point', n_reps=1000):
    """
    Check the signal-to-noise ratio (SNR) in the generated data by computing the per-pixel SNR values 
    using n_reps repetitions to compute the meand and standard deviation of the magnitude data per pixel.

    Parameters:
        magnitude (np.ndarray): Original magnitude data.
        phaseX, phaseY, phaseZ (np.ndarray): Original phase data for each axis.
        SNR (float): Desired signal-to-noise ratio. Use `np.inf` for no noise.
        enc_str (str): Encoding strategy for displacement encoding. Options are 'Balanced-Four-point' and 'Simple-Four-point'.
        n_reps (int): Number of repetitions to compute the mean and standard deviation of the magnitude data.

    Returns:
        np.ndarray: Per-pixel SNR values for the generated data.
    """
    # Preallocate array for storing noisy magnitudes
    magnitude_list = np.empty((n_reps, *magnitude.shape), dtype=magnitude.dtype)
    # Generate noisy data and store magnitude
    for i in range(n_reps):
        magnitude_wN, _, _, _ = add_noise_to_data_dense(magnitude, phaseX, phaseY, phaseZ, SNR, enc_str)
        magnitude_list[i] = magnitude_wN

    # Compute per-pixel mean and standard deviation
    magnitude_mean = np.mean(magnitude_list, axis=0)
    magnitude_std = np.std(magnitude_list, axis=0)
    # Compute per-pixel SNR
    snr_map = np.divide(
        magnitude_mean, 
        magnitude_std, 
        out=np.zeros_like(magnitude_mean),  # Handle division by zero safely
        where=magnitude_std > 0             # Only compute SNR where noise_std > 0
    )

    return snr_map

   
    
