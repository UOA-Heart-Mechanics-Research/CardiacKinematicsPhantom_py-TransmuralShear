############################### Library Imports
###############################################
import os
import numpy as np

############################### Directory Paths
###############################################

BASE_OUTPUT_PATH = 'results'
VTK_PATH = os.path.join(BASE_OUTPUT_PATH, 'vtk')
PHASE_FILE_PATH = os.path.join(BASE_OUTPUT_PATH, 'phasedata')
IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, 'images')
SA_SLICE_PATH = os.path.join(BASE_OUTPUT_PATH, 'short_axis')

############################### Configuration Parameters
########################################################
# Debug Flag
DEBUG_FLAG = 1  # Set to 1 to enable debug output

# Geometry of Analytical (Cylindrical) Phantom
R_ENDO = 25.0  # Endocardial radius in mm
R_EPI = 35.0   # Epicardial radius in mm
RMIN = 5.0     # Minimum radius to avoid singularities in displacement field

Z_BOTTOM = 0.0 # Bottom of the cylindrical phantom in mm
Z_TOP = 38.0   # Top of the cylindrical phantom in mm (shortens to ~32 mm at peak, filling four 8 mm slices at z = 4, 12, 20, 28)

# Model-based Myofiber Orientation Angles (in radians)
THETA_ENDO = 37.0 * np.pi / 180.0
THETA_MID = -9.0 * np.pi / 180.0
THETA_EPI = -45.0 * np.pi / 180.0

OPT_BETA = 1            # 1 -> minimize parameters to compute torsion; 0 -> fixed torsion; 
BETA_MAX = 0*np.pi/180  # [rad] Max torsion - Used if OptBeta = 0


# Cardiac Motion Parameters (Displacement Coefficients)
COMPUTE_PHANTOM = 1     # Set COMPUTE_PHANTOM to 1 if you want to recompute the parameters, otherwise set to 0 to use predefined values
SAVE_PHANTOM_VTK = 1    # Set to 1 to save the phantom geometry as a VTK and VTU file
AFINAL = np.array([-19.660282302570206, 0.6673134857325834, -0.005894384160182399, -0.15734976241731824, 0.12142141492532116, -0.03634028399959273])  # Predefined values if COMPUTE_PHANTOM is 0


# Target strains and weights for optimization. Shears are tensor shear strains,
# not engineering shear strains. "i" and "o" denote inner and outer wall.
# Shared targets (ELL, ECL, ECR, ELR, EFF) are matched at both walls, so they
# carry about twice the weight of the wall-specific ones.
TARGET_STRAIN_NAMES = ('ECCi', 'ECCo', 'ELL', 'ERRi', 'ERRo', 'ECL', 'ECR', 'ELR', 'EFF')
TARGET_STRAINS = np.array([-0.20, -0.16, -0.15, 0.45, 0.30, 0.035, 0.03875, 0.03, -0.15])
WEIGHTS = np.ones(len(TARGET_STRAINS))

# J is retained as an optional objective term, but a zero weight removes the
# incompressibility constraint from the optimization.
TARGET_J = 1.0
WEIGHT_J = 0.0


# Time Steps for Cardiac Cycle (Defines the variation of displacement magnitude over time)
TIME_STEPS = 21
ALPHA_TIME = np.sin(np.linspace(0, np.pi, TIME_STEPS))
ENC_STR='Simple-Four-point' # Encoding strategy

# Physical Field of View limits
XLIM = np.array([-100, 100])   # X-axis limits for Field of View in mm
YLIM = np.array([-100, 100])   # Y-axis limits for Field of View in mm

###################### Change as needed for different slice locations ######################
ZCONST = 8                 # Z-slice location for imaging in mm
############################################################################################

# Voxel Size
HX = 2.5                     # X-grid size in mm
HY = 2.5                     # Y-grid size in mm
HZ = 8.0                     # Z-grid size (through-plane voxel size) in mm

# Sampling points per Voxel
SAMPLE_X = 2                 # Sample points per voxel in X
SAMPLE_Y = 2                 # Sample points per voxel in Y
SAMPLE_Z = 3                 # Sample points per voxel in Z

# Encoding Frequency 
KE_X = 0.1
KE_Y = 0.1
KE_Z = 0.08

# Noise Parameters
SNR =  [40]               # [10, 20, 40, np.inf ~ 1000]   # Signal-to-Noise Ratios for different noise levels
REPS = 1                    # Number of repetitions per SNR level
CHECK_SNR = True            # Set to True to check SNR levels in the generated data

# Visualization Parameters
NPOINT_Z = 6                # Number of elements in Z direction for VTK representation
NPOINT_R = 3                # Number of elements in radial direction for VTK representation
NPOINT_C = 65               # Number of elements in circumferential direction for VTK representation
PHASE_EL_TYPE = 21          # Element type for phase images (linear quadrilateral elements)
CELL_SCALARS_NAME = ['Magnitude', 'Phase_X', 'Phase_Y', 'Phase_Z']  # Scalar names for VTK output

# Mesh and Phantom Parameters
PHANTOM_MESH_SIZE = 1.0    # Phantom mesh size for VTK representation (unused if NPOINT_Z, NPOINT_R, NPOINT_C are set)
N_POINTS_PER_VOXEL = SAMPLE_X * SAMPLE_Y * SAMPLE_Z     # Total sample points per voxel for averaging phase data
