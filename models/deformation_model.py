# deformation_model.py
import numpy as np
import matplotlib.pyplot as plt
try:
    from CardiacKinematicsPhantom_python.utils.displacement_utils import ComputeInvariants, Microstructure
    from CardiacKinematicsPhantom_python.utils.coordinate_utils import cart2pol
    from CardiacKinematicsPhantom_python.config import OPT_BETA, BETA_MAX, R_ENDO, R_EPI
except:
    from utils.displacement_utils import ComputeInvariants, Microstructure
    from utils.coordinate_utils import cart2pol
    from config import OPT_BETA, BETA_MAX, R_ENDO, R_EPI
import os 

def compute_deformation_mapping(
    Phi, R, Z, Zbot, Ztop, a, cartesian=False,
    Rinner=R_ENDO, Router=R_EPI, beta=BETA_MAX
):
    """
    Computes the deformation mapping for a cylindrical point based on given coefficients.

    Parameters:
        Phi (float): Circumferential angle in radians.
        R (float): Radial position.
        Z (float): Axial position.
        Zbot (float): Bottom Z-bound of the cylinder.
        Ztop (float): Top Z-bound of the cylinder.
        a (np.ndarray): Array of deformation coefficients.
        cartesian (bool): If True, returns Cartesian coordinates instead of cylindrical.
    
    Returns:
        tuple: Deformed coordinates (r, phi, z) in cylindrical coordinates or (x, y, z) in Cartesian.

    For OPT_BETA=1, the deformed angle is
        t = Phi + a4*(Z-Zbot)/(Ztop-Zbot)
                + a5*(R-Router)/(Rinner-Router).
    """
    r = a[0] + (1 + a[1]) * R + a[2] * R**2
    if OPT_BETA:
        if len(a) < 6:
            raise ValueError("The radial-torsion model requires six coefficients (a0...a5).")
        axial_position = (Z - Zbot) / (Ztop - Zbot)
        radial_position = (R - Router) / (Rinner - Router)
        t = Phi + a[4] * axial_position + a[5] * radial_position
    else:
        t = Phi + beta * (Z - Zbot) / (Ztop - Zbot)

    z = Z + a[3] * Z

    if cartesian:
        xa = r * np.cos(t)
        ya = r * np.sin(t)
        za = z
        return xa, ya, za
    else:   
        return r, t, z

def compute_Fp(
    R, Z, Zbot, Ztop, a, Rinner=R_ENDO, Router=R_EPI, beta=BETA_MAX
):
    """
    Constructs the in-plane deformation gradient matrix Fp.

    Parameters:
        R (float): Radial position.
        Theta (float): Circumferential angle in radians.
        partials (dict): Partial derivatives from compute_partial_derivatives.
    
    Returns:
        np.ndarray: In-plane deformation gradient tensor Fp.
    """
    r = a[0] + (1+a[1])*R + a[2]*R**2

    if OPT_BETA:
        if len(a) < 6:
            raise ValueError("The radial-torsion model requires six coefficients (a0...a5).")
        dt_dR = a[5] / (Rinner - Router)
        dt_dZ = a[4] / (Ztop-Zbot)
    else:
        dt_dR = 0.0
        dt_dZ = beta / (Ztop-Zbot)

    dr_dR = 1 + a[1] + 2*a[2]*R
    dz_dZ = 1.0 + a[3] 

    # Rows use the deformed (r, theta, z) basis and columns use the reference
    # (R, Theta, Z) basis. Angular derivatives acquire a factor of r.
    Fp = np.array([[dr_dR,       0.0,       0.0],
                   [r * dt_dR,   r / R, r * dt_dZ],
                   [0.0,         0.0,     dz_dZ]])
    return Fp

def compute_Fan(Fp, Phi, theta):
    """
    Computes the full deformation gradient tensor in Cartesian coordinates.

    Parameters:
        Fp (np.ndarray): In-plane deformation gradient tensor.
        Phi (float): Original circumferential angle.
    Returns:
        np.ndarray: Full deformation gradient tensor Fan.
    """
    reference_basis = np.array([
        [np.cos(Phi), -np.sin(Phi), 0.0],
        [np.sin(Phi),  np.cos(Phi), 0.0],
        [0.0,          0.0,         1.0],
    ])
    deformed_basis = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    return deformed_basis @ Fp @ reference_basis.T

def AnalyticalF(
    Phi, R, Z, Zbot, Ztop, a, beta=BETA_MAX,
    Rinner=R_ENDO, Router=R_EPI
):
    """
    High-level function that calculates the full gradient tensor.

    Parameters:
        Phi, R, Z (float): Cylindrical coordinates of the point.
        Zbot, Ztop (float): Z-bounds of the cylinder.
        a (np.ndarray): Deformation coefficients.

    Returns:
        tuple: Deformation gradient tensor Fan.
    """
    Fp = compute_Fp(R, Z, Zbot, Ztop, a, Rinner, Router, beta)
    _, theta, _ = compute_deformation_mapping(
        Phi, R, Z, Zbot, Ztop, a, cartesian=False,
        Rinner=Rinner, Router=Router, beta=beta
    )
    Fan = compute_Fan(Fp, Phi, theta)

    return Fan

def AnalyticalPhantom(afinal, Rendo, Repi, Zbot, Ztop, h, NpointZ, NpointR, NpointC, thetaEndo, thetaMid, thetaEpi, alphaTime, output_path, DebugFlag=0):
    """
    Generates and visualizes the deformation of a cylindrical phantom over time.

    Parameters:
        afinal (np.ndarray): Final deformation coefficients.
        Rendo, Repi, Zbot, Ztop (float): Geometry of the phantom.
        h (float): Mesh size.
        NpointZ, NpointR, NpointC (int): Mesh resolution in Z, radial, and circumferential directions.
        thetaEndo, thetaMid, thetaEpi (float): Fiber orientation angles.
        alphaTime (np.ndarray): Time-dependent scaling factors.
        output_path (str): Path for saving VTK files.
        DebugFlag (int): Enables debugging.
    """
    from utils.mesh_utils import phantom_hexa_mesh
    from utils.vtk_utils import plot_to_vtk

    # Generate phantom mesh
    nodesCart, nodesPolar, conn = phantom_hexa_mesh(Rendo, Repi, Zbot, Ztop, h, NpointZ, NpointR, NpointC)
    NumInv = 8

    # Arrays for strain invariants over time
    Eff_an = np.empty((len(alphaTime), conn.shape[0]))
    J_an = np.empty((len(alphaTime), conn.shape[0]))
    Eff_num = np.empty((len(alphaTime), conn.shape[0]))
    J_num = np.empty((len(alphaTime), conn.shape[0]))
    FibersUpdated = np.empty((conn.shape[0], 3))

    for s, alpha in enumerate(alphaTime):
        a_s = afinal * alpha
        xt = np.empty((nodesPolar.shape[0], 3))

        # Calculate deformed coordinates at each time step
        for i, node in enumerate(nodesPolar):
            Fan = AnalyticalF(
                node[0], node[1], node[2], Zbot, Ztop, a_s,
                Rinner=Rendo, Router=Repi
            )
            xa, ya, za = compute_deformation_mapping(
                node[0], node[1], node[2], Zbot, Ztop, a_s,
                cartesian=True, Rinner=Rendo, Router=Repi
            )
            xt[i, :] = [xa, ya, za]

        # Calculate strain invariants and update fiber directions
        CellScalars = np.empty((conn.shape[0], NumInv * 2))
        for e, elem in enumerate(conn):
            Xel = nodesCart[elem - 1, :]
            ElBar = np.mean(Xel, axis=0)
            rqp = np.sqrt(ElBar[0] ** 2 + ElBar[1] ** 2)
            ThetaPol, _, Zpol = cart2pol(ElBar[0], ElBar[1], ElBar[2])

            Lvec = np.array([0, 0, 1])
            Cvec = np.array([ElBar[1], -ElBar[0], 0.0]) / np.linalg.norm([ElBar[1], -ElBar[0], 0.0])
            # Cvec = np.array([-ElBar[1], ElBar[0], 0.0]) / np.linalg.norm([-ElBar[1], ElBar[0], 0.0])
            Rvec = np.array([ElBar[0], ElBar[1], 0.0]) / np.linalg.norm([ElBar[0], ElBar[1], 0.0])

            fiber = Microstructure(thetaEndo, thetaMid, thetaEpi, Rendo, Repi, rqp, Rvec, Cvec, DebugFlag)
            Fan = AnalyticalF(
                ThetaPol, rqp, Zpol, Zbot, Ztop, afinal * alpha,
                Rinner=Rendo, Router=Repi
            )
            CellScalars[e, :NumInv], _ = ComputeInvariants(Fan, fiber, Lvec, Cvec, Rvec)
            Eff_an[s, e] = CellScalars[e, 4]
            J_an[s, e] = np.sqrt(CellScalars[e, 2])

            FibersUpdated[e, :] = fiber[:3] @ Fan.T
            FibersUpdated[e, :] = FibersUpdated[e, :] / np.linalg.norm(FibersUpdated[e, :])

            phiEl = xt[elem - 1, :]
            numQP = 0

            Fnum = LinHexa(Xel, phiEl, numQP)
      
            CellScalars[e, NumInv:NumInv * 2], InvariantNames = ComputeInvariants(Fnum, fiber, Lvec, Cvec, Rvec)
            Eff_num[s, e] = CellScalars[e, NumInv + 4]
            J_num[s, e] = np.sqrt(CellScalars[e, NumInv + 2])

        # Save VTK file at each time step
        el_type = 32
        scalar_names = ["I1", "I2", "I3", "I4", "EFF", "ELL", "ECC", "ERR"]
        output_file_name = os.path.join(output_path,f"Phantom_Time_{s:02d}")
        plot_to_vtk(xt, conn - 1, el_type, np.array([]), np.array([]), CellScalars, scalar_names, output_file_name)

    # Plot median EFF and J over time
    EffAnMedian = np.median(Eff_an, axis=1)
    plt.figure(10)
    plt.plot(EffAnMedian, '-or', label='Analytical')
    plt.plot(np.median(Eff_num, axis=1), '-ok', label='Numerical')
    plt.legend()
    plt.xlabel('Time [ms]')
    plt.ylabel('E_{ff}')
    plt.savefig(f"{output_path}/Eff_Eff.png", dpi=300, bbox_inches='tight')

    plt.figure(11)
    plt.plot(np.median(J_an, axis=1), '-or', label='Analytical')
    plt.plot(np.median(J_num, axis=1), '-ok', label='Numerical')
    plt.legend()
    plt.xlabel('Time [ms]')
    plt.ylabel('J')
    plt.savefig(f"{output_path}/J_Eff.png", dpi=300, bbox_inches='tight')

def LinHexa(Xel, phiEl, numQP):
    #Checked
    if numQP == 0:
        QP = np.array([[0.0, 0.0, 0.0]])
    elif numQP == 1:
        qp = 1.0 / np.sqrt(3)
        QP = np.array([[-qp, -qp, -qp],
                       [ qp, -qp, -qp],
                       [ qp,  qp, -qp],
                       [-qp,  qp, -qp],
                       [-qp, -qp,  qp],
                       [ qp, -qp,  qp],
                       [ qp,  qp,  qp],
                       [-qp,  qp,  qp]])

    for q in range(QP.shape[0]):
        # derivative of shape functions in isoparametric space
        DNr = 0.125 * np.array([[(QP[q, 1] - 1) * (1 - QP[q, 2]), (1 - QP[q, 1]) * (1 - QP[q, 2]), (1 + QP[q, 1]) * (1 - QP[q, 2]), (1 + QP[q, 1]) * (QP[q, 2] - 1),
                                 (QP[q, 1] - 1) * (1 + QP[q, 2]), (1 - QP[q, 1]) * (1 + QP[q, 2]), (1 + QP[q, 1]) * (1 + QP[q, 2]), -(1 + QP[q, 1]) * (1 + QP[q, 2])],
                                [(QP[q, 0] - 1) * (1 - QP[q, 2]), (1 + QP[q, 0]) * (QP[q, 2] - 1), (1 + QP[q, 0]) * (1 - QP[q, 2]), (1 - QP[q, 0]) * (1 - QP[q, 2]),
                                 (QP[q, 0] - 1) * (1 + QP[q, 2]), -(QP[q, 0] + 1) * (1 + QP[q, 2]), (1 + QP[q, 0]) * (1 + QP[q, 2]), (1 - QP[q, 0]) * (1 + QP[q, 2])],
                                [(QP[q, 0] - 1) * (1 - QP[q, 1]), (1 + QP[q, 0]) * (QP[q, 1] - 1), -(1 + QP[q, 0]) * (1 + QP[q, 1]), (QP[q, 0] - 1) * (1 + QP[q, 1]),
                                 (1 - QP[q, 0]) * (1 - QP[q, 1]), (1 + QP[q, 0]) * (1 - QP[q, 1]), (1 + QP[q, 0]) * (1 + QP[q, 1]), (1 - QP[q, 0]) * (1 + QP[q, 1])]])

        # Jacobian transformation matrix
        JT = DNr @ Xel
        detJ = np.linalg.det(JT)  # Element volume

        if detJ < 0:
            print("ERROR: element with negative Jacobian")

        JTinv = np.linalg.inv(JT)

        DNx = JTinv @ DNr  # Shape functions derivatives in XYZ domain

        # Compute deformation gradient F
        Fnum = np.zeros((3, 3))
        for i in range(8):
            Fnum[0, 0] += phiEl[i, 0] * DNx[0, i]
            Fnum[0, 1] += phiEl[i, 0] * DNx[1, i]
            Fnum[0, 2] += phiEl[i, 0] * DNx[2, i]

            Fnum[1, 0] += phiEl[i, 1] * DNx[0, i]
            Fnum[1, 1] += phiEl[i, 1] * DNx[1, i]
            Fnum[1, 2] += phiEl[i, 1] * DNx[2, i]

            Fnum[2, 0] += phiEl[i, 2] * DNx[0, i]
            Fnum[2, 1] += phiEl[i, 2] * DNx[1, i]
            Fnum[2, 2] += phiEl[i, 2] * DNx[2, i]

    return Fnum
