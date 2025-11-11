# optimize_model.py
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from models.deformation_model import AnalyticalF, compute_deformation_mapping
from utils.displacement_utils import Microstructure
import pandas as pd
from utils.coordinate_utils import cart2pol
from config import OPT_BETA
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
    error = 0.0
    NsamplePointsR = 2
    NsamplePointsZ = 3

    R = np.linspace(Rendo, Repi, NsamplePointsR)
    Z = np.linspace(Zbot, Ztop, NsamplePointsZ)
    eR = np.array([np.cos(Phi), np.sin(Phi), 0.0])
    eT = np.array([-np.sin(Phi), np.cos(Phi), 0.0])
    # eT = np.array([np.sin(Phi), -np.cos(Phi), 0.0])
    eZ = np.array([0.0, 0.0, 1.0])

    # Compute error over sample points
    for i in range(NsamplePointsR):
        for j in range(NsamplePointsZ):
            Fan = AnalyticalF(Phi, R[i], Z[j], Zbot, Ztop, a)
            Can = Fan.T @ Fan

            fiber = Microstructure(thetaEndo, thetaMid, thetaEpi, Rendo, Repi, R[i], eR, eT, DebugFlag)
            
            J = np.linalg.det(Fan)
            ELL = 0.5 * (eZ.T @ Can @ eZ - 1.0)
            ECC = 0.5 * (eT.T @ Can @ eT - 1.0)
            ERR = 0.5 * (eR.T @ Can @ eR - 1.0)
            EFF = 0.5 * (fiber[:3] @ Can @ fiber[:3].T - 1.0)
            # EFF = 0.5 * (fiber[:3].T @ Can @ fiber[:3] - 1.0)
            
            error += Weights[0] * (J - TargetStrains[0])**2 \
                     + Weights[1] * (ELL - TargetStrains[1])**2 \
                     + Weights[2 + i] * (ECC - TargetStrains[2 + i])**2 \
                     + Weights[4 + i] * (ERR - TargetStrains[4 + i])**2 \
                     + Weights[6] * (EFF - TargetStrains[6])**2

    return error

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
        a0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
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

        # print('                 J        ELL       ECC       ERR       EFF')
        columns = ['J', 'ELL', 'ECC', 'ERR', 'EFF']
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
                Fan = AnalyticalF(Phi, R[i], Z[j], Zbot, Ztop, a_final)
                #print(f'Fan={Fan}')
                C = np.transpose(Fan) @ Fan

                J = np.linalg.det(Fan)
                #print(J)
                ELL = 0.5*(np.transpose(eZ) @ C @ eZ - 1.0)
                ECC = 0.5*(np.transpose(eT) @ C @ eT - 1.0)
                ERR = 0.5*(np.transpose(eR) @ C @ eR - 1.0)
                fiber = Microstructure(thetaEndo, thetaMid, thetaEpi, Rendo, Repi, R[i], eR, eT, DebugFlag)
                EFF = 0.5*(np.dot(fiber[0:3], np.dot(C, fiber[0:3].T)) - 1.0)
                # EFF = 0.5*(np.dot(fiber[0:3].T, np.dot(C, fiber[0:3])) - 1.0)


                #add row to dataframe
                df.loc[Location[(i*NsamplePointsZ)+j]] = [J, ELL, ECC, ERR, EFF]
                
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