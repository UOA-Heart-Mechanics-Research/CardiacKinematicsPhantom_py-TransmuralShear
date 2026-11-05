# optimize_model.py
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from models.deformation_model import AnalyticalF, compute_deformation_mapping
from utils.displacement_utils import Microstructure
import pandas as pd
from utils.coordinate_utils import cart2pol
from config import OPT_BETA, TARGET_J, WEIGHT_J, TARGET_STRAIN_NAMES
def target_strain_error(a, Rendo, Repi, Zbot, Ztop, thetaEndo, thetaMid, thetaEpi, TargetStrains, Weights, Phi, DebugFlag):
    """
    Computes the squared error between the calculated strain and target strain for given parameters.

    Parameters:
        a (np.ndarray): Array of deformation coefficients.
        Rendo, Repi (float): Endocardial and epicardial radii.
        Zbot, Ztop (float): Bottom and top Z-bounds of the phantom.
        thetaEndo, thetaMid, thetaEpi (float): Fiber angles for endocardium, mid-wall, and epicardium.
        TargetStrains (np.ndarray): Target strain values.
        Weights (np.ndarray): Weights for strain errors.
        Phi (float): Circumferential angle (constant for axial symmetry).
        DebugFlag (int): If set, enables additional debugging output.

    Returns:
        float: The total weighted error between computed and target strains.
    """
    if len(TargetStrains) != len(TARGET_STRAIN_NAMES):
        raise ValueError(
            f"Expected targets {TARGET_STRAIN_NAMES}, got {len(TargetStrains)} values."
        )
    if len(Weights) != len(TargetStrains):
        raise ValueError("Weights and target strains must have the same length.")

    Zmid = 0.5 * (Zbot + Ztop)
    eR = np.array([np.cos(Phi), np.sin(Phi), 0.0])
    eT = np.array([-np.sin(Phi), np.cos(Phi), 0.0])
    eZ = np.array([0.0, 0.0, 1.0])

    def strains_at_radius(radius):
        Fan = AnalyticalF(
            Phi, radius, Zmid, Zbot, Ztop, a,
            Rinner=Rendo, Router=Repi
        )
        Ean = 0.5 * (Fan.T @ Fan - np.eye(3))
        fiber = Microstructure(
            thetaEndo, thetaMid, thetaEpi,
            Rendo, Repi, radius, eR, eT, DebugFlag
        )[:3]
        return np.array([
            eT.T @ Ean @ eT,       # ECC
            eZ.T @ Ean @ eZ,       # ELL
            eR.T @ Ean @ eR,       # ERR
            eT.T @ Ean @ eZ,       # ECL
            eT.T @ Ean @ eR,       # ECR
            eZ.T @ Ean @ eR,       # ELR
            fiber @ Ean @ fiber.T, # EFF
            np.linalg.det(Fan),     # J
        ])

    inner = strains_at_radius(Rendo)
    outer = strains_at_radius(Repi)
    wall_mean = 0.5 * (inner + outer)

    # One residual per named target. Inner/outer normal strains use their
    # corresponding wall; shared targets use the two-wall mean.
    calculated = np.array([
        inner[0],     # ECCi
        outer[0],     # ECCo
        wall_mean[1], # ELL
        inner[2],     # ERRi
        outer[2],     # ERRo
        wall_mean[3], # ECL
        wall_mean[4], # ECR
        wall_mean[5], # ELR
        wall_mean[6], # EFF
    ])
    strain_error = np.sum(Weights * (calculated - TargetStrains)**2)
    jacobian_error = WEIGHT_J * (wall_mean[7] - TARGET_J)**2
    return strain_error + jacobian_error

def OptimizeDisp(Rendo, Repi, Zbot, Ztop, thetaEndo, thetaMid, thetaEpi, TargetStrains, Weights, DebugFlag):
    """
    Optimizes deformation coefficients to match a target strain distribution.

    Parameters:
        Rendo, Repi (float): Endocardial and epicardial radii.
        Zbot, Ztop (float): Bottom and top Z-bounds of the phantom.
        thetaEndo, thetaMid, thetaEpi (float): Fiber angles for endocardium, mid-wall, and epicardium.
        TargetStrains (np.ndarray): Target strain values.
        Weights (np.ndarray): Weights for strain error terms.
        DebugFlag (int): If set, enables additional debugging output.

    Returns:
        np.ndarray: Optimized deformation coefficients.
    """
    Phi = 0.0  # Circumferential angle (axially symmetric phantom)

    # Initial guess for deformation coefficients
    if OPT_BETA:
        a0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        a0 = np.array([0.0, 0.0, 0.0, 0.0])

    # Store the objective function value at each iteration for plotting
    y_values = []

    # Callback to record the error at each iteration
    def callback(xk):
        y_values.append(target_strain_error(xk, Rendo, Repi, Zbot, Ztop, thetaEndo, thetaMid, thetaEpi, TargetStrains, Weights, Phi, DebugFlag))

    # Define cost function for minimization
    def cost_function(a, *params):
        return target_strain_error(a, *params)

    # Optimization parameters
    options = {'maxiter': 1e5, 'maxfev': 1e5, 'disp': DebugFlag == 1}

    # Optimization arguments
    params = (Rendo, Repi, Zbot, Ztop, thetaEndo, thetaMid, thetaEpi, TargetStrains, Weights, Phi, DebugFlag)
    result = minimize(cost_function, a0, args=params, tol=1e-7, method='Nelder-Mead', options=options, callback=callback)
    a_final = result.x

    # Plot the error over iterations if DebugFlag is set
    if DebugFlag == 1:
        plt.figure()
        plt.plot(range(len(y_values)), y_values)
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Optimization Error vs Iterations")
        plt.savefig(f"results/images/error_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        print('Optimization Results:')
        print(f'Function Value: {result.fun}')
        print(f'Exit Flag: {result.status}')

        columns = ['J', 'ECC', 'ELL', 'ERR', 'ECL', 'ECR', 'ELR', 'EFF']
        df = pd.DataFrame(columns=columns)

        # Analyze strain values
        NsamplePointsR = 2
        NsamplePointsZ = 3
        Location = ['Endo - Zbot', 'Endo - Zmid', 'Endo - Ztop', 'Epi  - Zbot', 'Epi  - Zmid', 'Epi  - Ztop']
        R = np.linspace(Rendo, Repi, NsamplePointsR)
        Z = np.linspace(Zbot, Ztop, NsamplePointsZ)

        eR = np.array([np.cos(Phi), np.sin(Phi), 0.0])
        # eT = np.array([np.sin(Phi), -np.cos(Phi), 0.0])
        eT = np.array([-np.sin(Phi), np.cos(Phi), 0.0])
        eZ = np.array([0.0, 0.0, 1.0])
      
        for i in range(NsamplePointsR):
            for j in range(NsamplePointsZ):
                Fan = AnalyticalF(
                    Phi, R[i], Z[j], Zbot, Ztop, a_final,
                    Rinner=Rendo, Router=Repi
                )
                #print(f'Fan={Fan}')
                C = np.transpose(Fan) @ Fan

                J = np.linalg.det(Fan)
                #print(J)
                E = 0.5 * (C - np.eye(3))
                ECC = np.transpose(eT) @ E @ eT
                ELL = np.transpose(eZ) @ E @ eZ
                ERR = np.transpose(eR) @ E @ eR
                ECL = np.transpose(eT) @ E @ eZ
                ECR = np.transpose(eT) @ E @ eR
                ELR = np.transpose(eZ) @ E @ eR
                fiber = Microstructure(thetaEndo, thetaMid, thetaEpi, Rendo, Repi, R[i], eR, eT, DebugFlag)
                EFF = fiber[0:3] @ E @ fiber[0:3].T


                #add row to dataframe
                df.loc[Location[(i*NsamplePointsZ)+j]] = [
                    J, ECC, ELL, ERR, ECL, ECR, ELR, EFF
                ]
                
        print(df)
        l_over_L = 1+a_final[3]
        rendo = a_final[0] + Rendo*(1+a_final[1]) + a_final[2]*Rendo**2

        EF = 1.0 - l_over_L*(rendo/Rendo)**2
        print('')
        print(f'Ejection fraction = {EF}')
        print('')

    
    return a_final

def EulerianMap(X, xgiven, Zbot, Ztop, a_s):
    """
    Maps a point from the Lagrangian to the Eulerian frame by computing the
    difference between the given point and its deformation mapping.

    Parameters:
        X (np.ndarray): Original Cartesian coordinates of the point (x, y, z).
        xgiven (np.ndarray): Reference Cartesian coordinates of the point.
        Zbot (float): Bottom Z-bound of the cylinder.
        Ztop (float): Top Z-bound of the cylinder.
        a_s (np.ndarray): Deformation coefficients for the current state.
        

    Returns:
        np.ndarray: Mapping vector G = xgiven - deformed coordinates.
    """
    # Convert to cylindrical coordinates
    Phi, R, Z = cart2pol(X[0], X[1], X[2])

    # Compute deformation mapping with the given coefficients and options
    xa, ya, za = compute_deformation_mapping(Phi, R, Z, Zbot, Ztop, a_s, cartesian=True)

    # Calculate the Eulerian mapping difference
    G = xgiven - np.array([xa, ya, za])
    return G
