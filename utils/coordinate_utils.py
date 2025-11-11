# coordinate_utils.py

import numpy as np

def cart2pol(x, y, z):
    """
    Converts Cartesian coordinates (x, y, z) to cylindrical polar coordinates (theta, r, z).

    Parameters:
        x, y, z (float): Cartesian coordinates.
    
    Returns:
        tuple: (theta, r, z), where theta is the polar angle, r is the radial distance, and z remains the same.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, r, z
