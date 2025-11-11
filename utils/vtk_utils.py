# vtk_utils.py

import numpy as np
import pyvista as pv

def plot_to_vtk(
    xt, conn, el_type, node_scalars=None, node_scalars_names=None, 
    cell_scalars=None, cell_scalars_names=None, output_file_name="output", 
    visualize=False, binary=False, file_format="vtk"
):
    """
    Print connectivity, invariants, and nodal positions to VTK in various formats.
    """
    # Validate inputs
    if xt.shape[1] != 3:
        raise ValueError("The `xt` array must have shape (N, 3), where N is the number of nodes.")
    if conn.ndim != 2:
        raise ValueError("The `conn` array must be a 2D array with element connectivity.")

    # Determine cell types and build connectivity
    element_type_map = {
        11: (pv.CellType.TRIANGLE, 3),
        12: (pv.CellType.QUADRATIC_TRIANGLE, 6),
        21: (pv.CellType.QUAD, 4),
        22: (pv.CellType.QUADRATIC_QUAD, 8),
        31: (pv.CellType.TETRA, 4),
        32: (pv.CellType.HEXAHEDRON, 8),
    }

    if el_type not in element_type_map:
        raise ValueError(f"Element type {el_type} not implemented.")

    cell_type, nodes_per_element = element_type_map[el_type]

    if conn.shape[1] != nodes_per_element:
        raise ValueError(f"Each row in `conn` must have {nodes_per_element} elements for element type {el_type}.")

    # Prepare VTK cell structure
    n_cells = conn.shape[0]
    cell_size = conn.shape[1]
    cells = np.hstack((np.full((n_cells, 1), cell_size), conn)).ravel()
    celltypes = np.full(n_cells, cell_type)
    
    # Create PyVista UnstructuredGrid object
    grid = pv.UnstructuredGrid(cells, celltypes, xt)

    # Add node scalars
    if node_scalars is not None and node_scalars_names is not None and len(node_scalars_names) > 0:
        if node_scalars.shape[0] != xt.shape[0]:
            raise ValueError("The number of rows in `node_scalars` must match the number of nodes in `xt`.")
        for i, scalar_name in enumerate(node_scalars_names):
            grid.point_data[scalar_name] = node_scalars[:, i]

    # Add cell scalars
    if cell_scalars is not None and cell_scalars_names is not None and len(cell_scalars_names) > 0:
        if cell_scalars.shape[0] != conn.shape[0]:
            raise ValueError("The number of rows in `cell_scalars` must match the number of elements in `conn`.")
        for i, scalar_name in enumerate(cell_scalars_names):
            grid.cell_data[scalar_name] = cell_scalars[:, i]

    # Define output file path with the chosen format
    supported_formats = {"vtk", "vtu", "vtp"}
    if file_format not in supported_formats:
        raise ValueError(f"Unsupported file format '{file_format}'. Choose from {supported_formats}.")
        
    # Save the file
    if file_format == "vtk":
        if 'vtk' not in output_file_name[-4:]:
            output_file_name += ".vtk"
        grid.save(output_file_name, binary=binary)
    elif file_format == "vtu":
        if 'vtu' not in output_file_name[-4:]:
            output_file_name += ".vtu"
        grid.save(output_file_name, binary=binary)
    elif file_format == "vtp":
        if 'vtp' not in output_file_name[-4:]:
            output_file_name += ".vtp"
        grid.cast_to_polydata().save(output_file_name, binary=binary)

    print(f"File saved in {file_format} format to: {output_file_name}")

    # Visualize if requested
    if visualize:
        grid.plot(show_edges=True)


def rescale_image(img):
    """
    Rescales an image array to the range [0, 1] for visualization.
    
    Parameters:
        img (np.ndarray): The input image array.
    
    Returns:
        np.ndarray: Rescaled image array with values in the range [0, 1].
    """
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)
    

