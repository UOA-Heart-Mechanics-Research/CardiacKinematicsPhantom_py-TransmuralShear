# displacement_utils.py

import numpy as np

def eulerian_disp(x, Zbot, Ztop, a):
    """
    Converts a displacement from Lagrangian to Eulerian coordinates for a cylindrical phantom.
    
    Parameters:
        x (np.ndarray): Cartesian coordinates of the point (x, y, z).
        Zbot (float): Bottom Z-coordinate of the cylinder.
        Ztop (float): Top Z-coordinate of the cylinder.
        a (np.ndarray): Array of displacement coefficients.
    
    Returns:
        Xref (np.ndarray): Reference (Lagrangian) coordinates after transformation.
    """
    tol = 1.0e-8
    if np.linalg.norm(a) > tol:
        Xref = np.zeros(3)
        Xref[2] = x[2] / (1 + a[4])

        theta = np.arctan2(x[1], x[0])
        rho = np.sqrt(x[0]**2 + x[1]**2)

        Theta = theta - a[3] * (x[2] / (1 + a[4]) - Zbot) / (Ztop - Zbot)
        R = (-(1 + a[1]) + np.sqrt((1 + a[1])**2 - 4 * a[2] * (a[0] - rho))) / (2 * a[2])

        Xref[0] = R * np.cos(Theta)
        Xref[1] = R * np.sin(Theta)
    else:
        Xref = x
    return Xref

def is_myocardium(Xval, Rendo, Repi, Zbot, Ztop):
    """
    Checks if a point is within the myocardium bounds (cylinder shell between endocardial and epicardial surfaces).
    
    Parameters:
        Xval (np.ndarray): Cartesian coordinates of the point (x, y, z).
        Rendo (float): Endocardial radius.
        Repi (float): Epicardial radius.
        Zbot (float): Bottom Z-coordinate of the myocardium.
        Ztop (float): Top Z-coordinate of the myocardium.
    
    Returns:
        bool: True if the point is within the myocardium bounds, otherwise False.
    """
    Rval = np.linalg.norm(Xval[:2])
    tol = 1.0e-7

    # Check radial and Z bounds
    if Rendo - tol < Rval < Repi + tol and Zbot <= Xval[2] <= Ztop:
        return True
    else:
        if not (Zbot <= Xval[2] <= Ztop):
            print(f"Warning: Point {Xval} is outside Z bounds")  # Optional debug output
        return False

def ComputeInvariants(F, Fiber, Lvec, Cvec, Rvec):
    """
    Computes strain invariants based on the deformation gradient tensor F and various fiber directions.
    
    Parameters:
        F (np.ndarray): Deformation gradient tensor (3x3).
        Fiber (np.ndarray): Fiber direction vector.
        Lvec (np.ndarray): Longitudinal vector direction.
        Cvec (np.ndarray): Circumferential vector direction.
        Rvec (np.ndarray): Radial vector direction.
    
    Returns:
        Invariants (np.ndarray): Array of strain invariants.
        InvariantNames (list): List of names corresponding to each invariant.
    """
    fib = Fiber[:3].T
    C = F.T @ F  # Right Cauchy-Green deformation tensor
    E = 0.5 * (C - np.eye(3))  # Green-Lagrangian strain tensor

    # Initialize invariants and their names
    Invariants = np.zeros(8)
    InvariantNames = ['I1_', 'I2_', 'I3_', 'I4_', 'EFF', 'ELL', 'ECC', 'ERR']

    # Compute invariants
    Invariants[0] = np.trace(C)
    Invariants[1] = 0.5 * (np.trace(C)**2 - np.trace(C @ C))
    Invariants[2] = np.linalg.det(C)
    Invariants[3] = fib.T @ C @ fib
    Invariants[4] = fib.T @ E @ fib
    Invariants[5] = Lvec.T @ E @ Lvec
    Invariants[6] = Cvec.T @ E @ Cvec
    Invariants[7] = Rvec.T @ E @ Rvec

    return Invariants, InvariantNames

def Microstructure(ThetaEndo, ThetaMid, ThetaEpi, Rendo, Repi, R, eR, eT, DebugFlag=0):
    """
    Computes the microstructure (fiber orientation) in the myocardium for given parameters.
    
    Parameters:
        ThetaEndo (float): Fiber angle at the endocardium.
        ThetaMid (float): Fiber angle at the mid-wall.
        ThetaEpi (float): Fiber angle at the epicardium.
        Rendo (float): Endocardial radius.
        Repi (float): Epicardial radius.
        R (float): Current radial position.
        eR (np.ndarray): Radial direction unit vector.
        eT (np.ndarray): Tangential direction unit vector.
        DebugFlag (int): Debug flag for extra output.
    
    Returns:
        fiber (np.ndarray): Fiber orientation vector.
    """

    # Normalize alpha to [0, 1] from endo to epi
    r = (R - Rendo) / (Repi - Rendo)

    # Compute the fiber angle at this radial position
    theta = (
        2.0 * ThetaEndo * (r - 0.5) * (r - 1.0) -
        4.0 * ThetaMid * r * (r - 1) +
        2.0 * ThetaEpi * r * (r - 0.5)
    )

    # Compute fiber orientation vector at element position
    fiber = eT * np.cos(theta) + np.cross(eR, eT) * np.sin(theta)

    # Debug checks for perpendicularity and normalization
    tol = 1.0e-8
    if DebugFlag == 1:
        if np.dot(fiber, eR) > tol:
            print("Warning: fiber and radial vector are not perpendicular")
        if abs(np.linalg.norm(fiber) - 1) > tol:
            print("Warning: fiber vector is not normalized")

    # Compute the perpendicular direction to the fiber and radial vectors
    fperp = np.cross(fiber, eR)
    fiber = np.concatenate((fiber, fperp, eR))  # Return concatenated fiber, perpendicular vector, and radial vector

    return fiber
