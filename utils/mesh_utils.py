# mesh_utils.py

import numpy as np

def phantom_hexa_mesh(Rendo, Repi, Zbot, Ztop, h, NpointZ, NpointR, NpointC):
    """
    Generates a hexahedral mesh for the cylindrical phantom.
    
    Parameters:
        Rendo (float): Endocardial radius.
        Repi (float): Epicardial radius.
        Zbot (float): Bottom Z-coordinate of the cylinder.
        Ztop (float): Top Z-coordinate of the cylinder.
        h (float): Mesh spacing.
        NpointZ (int): Number of elements in the Z direction.
        NpointR (int): Number of elements in the radial direction.
        NpointC (int): Number of elements in the circumferential direction.
    
    Returns:
        nodesCart (np.ndarray): Cartesian coordinates of mesh nodes.
        nodesPolar (np.ndarray): Polar coordinates of mesh nodes.
        conn (np.ndarray): Element connectivity for the mesh.
    """
    # Calculate the number of points in each direction based on spacing if not specified
    DeltaH = Ztop - Zbot
    if NpointZ < 1:
        NpointZ = int(np.ceil(DeltaH / h))
    if NpointR < 1:
        NpointR = int(np.ceil((Repi - Rendo) / h))
    if NpointC < 1:
        NpointC = int(np.ceil(np.pi * (Rendo + Repi) / h))

    # Initialize arrays for node coordinates
    total_nodes = (NpointZ + 1) * (NpointR + 1) * NpointC
    nodesCart = np.empty((total_nodes, 3))
    nodesPolar = np.empty((total_nodes, 3))

    # Generate nodes in cylindrical coordinates
    ind = 0
    for k in range(NpointZ + 1):
        z = Zbot + k * DeltaH / NpointZ
        for j in range(NpointR + 1):
            r = Rendo + (Repi - Rendo) * j / NpointR
            for i in range(1, NpointC + 1):
                theta = 2 * np.pi * i / NpointC
                nodesCart[ind] = [r * np.cos(theta), r * np.sin(theta), z]
                nodesPolar[ind] = [theta, r, z]
                ind += 1

    # Initialize array for element connectivity
    total_elements = NpointZ * NpointR * NpointC
    conn = np.empty((total_elements, 8), dtype=int)

    # Define connectivity for each element in the mesh
    ind = 0
    for k in range(NpointZ):
        upZ = NpointC * (NpointR + 1)
        for j in range(NpointR):
            upR = NpointC
            for i in range(1, NpointC):
                conn[ind] = np.array([i, i + upR, i + 1 + upR, i + 1,
                                      i + upZ, i + upR + upZ, i + 1 + upR + upZ, i + 1 + upZ]) + upR * j + upZ * k
                ind += 1

            # Wrap around for the last circumferential element
            conn[ind] = np.array([NpointC, NpointC + upR, 1 + upR, 1,
                                  NpointC + upZ, NpointC + upR + upZ, 1 + upR + upZ, 1 + upZ]) + upR * j + upZ * k
            ind += 1

    return nodesCart, nodesPolar, conn

def phase_mesh(gridX, gridY, zconst):
    """
    Generates a 2D phase mesh in the X-Y plane at a constant Z value.
    
    Parameters:
        gridX (np.ndarray): Array of X coordinates for the grid.
        gridY (np.ndarray): Array of Y coordinates for the grid.
        zconst (float): Z-value (height) at which the phase mesh is generated.
    
    Returns:
        xPhase (np.ndarray): Phase mesh points with shape (Nx * Ny, 3).
        connPhase (np.ndarray): Element connectivity array for phase mesh.
    """
    Nx = len(gridX)
    Ny = len(gridY)
    xPhase = np.zeros((Nx * Ny, 3))
    
    # Assign X, Y, and Z coordinates to each mesh point
    ind = 0
    for i in range(Nx):
        for j in range(Ny):
            xPhase[ind, 0:3] = [gridX[i], gridY[j], zconst]
            ind += 1

    # Define connectivity for quadrilateral elements
    connPhase = np.zeros(((Nx - 1) * (Ny - 1), 4), dtype=int)
    ind = 0
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            connPhase[ind, 0:4] = [(i * Ny) + j, (i + 1) * Ny + j, (i + 1) * Ny + j + 1, (i * Ny) + j + 1]
            ind += 1

    return xPhase, connPhase
