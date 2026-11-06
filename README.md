# CardiacKinematicsPhantom_py-TransmuralShear

This is an extended version of the original CardiacKinematicsPhantom_py, adding non-zero transmural shear strain components ($E_{CR}$ and $E_{LR}$).

The extension includes:
* A change was made to the analytical equation 'theta' (the circumferential direction), adding radial dependence, and hence a sixth coefficient ($a_5$).
* The additional coefficient was added in to the optimisation to target strain code.
* Target strains were added for $E_{CL}$, $E_{CR}$ and $E_{LR}$.
* Other target strains were restored to the original MATLAB phantom values.
* All target strain weights were set to 1. Targets shared by both walls ($E_{LL}$, $E_{FF}$ and the shears) are matched at both the inner and outer wall, as in the original optimisation.
* The Jacobian objective weight was set to zero by default, so incompressibility is not imposed.
* The fibre angles were restored to the original MATLAB phantom values of 37°, −9° and −45° (endocardium, mid-wall, epicardium). 
* The phantom height was changed from 40 mm to 38 mm, so that four 8 mm short-axis slices fit within the phantom at peak contraction.
* A new script, `plot_analytic_strain_gt.py`, plots the analytical ground-truth strains without DENSE encoding, voxel averaging or noise.

This results in a deformation field that gives rise to non-zero strains for all six Green Lagrange strain components.

## Analytical deformation model

The phantom is a thick-walled cylinder with reference (undeformed) cylindrical
coordinates $(R, \Theta, Z)$, $R_{endo} \le R \le R_{epi}$ and
$Z_{bot} \le Z \le Z_{top}$. Each material point maps to deformed coordinates
$(r, \theta, z)$:

$$
\begin{aligned}
r &= a_0 + (1 + a_1)\,R + a_2\,R^2 \\
\theta &= \Theta + a_4\,\frac{Z - Z_{bot}}{Z_{top} - Z_{bot}} + a_5\,\frac{R - R_{epi}}{R_{endo} - R_{epi}} \\
z &= (1 + a_3)\,Z
\end{aligned}
$$

## Target strains

| Target | Value | Weight | Source | Achieved at peak (endo / epi) |
| --- | --- | --- | --- | --- |
| $E_{CC,i}$ | −0.20 | 1 | Reverted to original MATLAB phantom; matches Zhong et al. (2010), mid-ventricular subendocardial layer (−0.20 ± 0.02) | −0.231 |
| $E_{CC,o}$ | −0.16 | 1 | Reverted to original MATLAB phantom; matches Zhong et al. (2010), mid-ventricular subepicardial layer (−0.16 ± 0.02) | −0.096 |
| $E_{LL}$ | −0.15 | 1 | Unchanged. Moore et al. (2000) equatorial mean −0.15 | −0.143 / −0.140 |
| $E_{RR,i}$ | 0.45 | 1 | Reverted to original MATLAB phantom | 0.444 |
| $E_{RR,o}$ | 0.30 | 1 | Reverted to original MATLAB phantom | 0.294 |
| $E_{CL}$ | 0.035 | 1 | Mean of Moore et al. (2000), equatorial (0.035), and Young et al. (1994), mid-ventricle (0.035) | 0.022 / 0.045 |
| $E_{CR}$ | 0.03875 | 1 | Mean of Moore et al. (2000), equatorial (0.0525), and Young et al. (1994), mid-ventricle (0.025) | 0.024 / 0.051 |
| $E_{LR}$ | 0.03 | 1 | Mean of Moore et al. (2000), equatorial (0.0475), and Young et al. (1994), mid-ventricle (0.0125) | 0.002 / 0.006 |
| $E_{FF}$ | −0.15 | 1 | Unchanged | −0.179 / −0.163 |
| $J$ | 1.0 | 0 | Not imposed | 0.89 |

Each shear value from a paper is the mean over its four regions (Moore:
septal, anterior, lateral, inferior; Young: septal, posterior, lateral,
anterior). Moore et al. report $E_{CR}$ and $E_{LR}$ as $E_{RC}$ and $E_{RL}$.

Fibre angles are 37° (endocardium), −9° (mid-wall) and −45° (epicardium), reverted to the original MATLAB phantom.

The undeformed height is 38 mm (`Z_TOP`). At peak contraction the phantom shortens to about 32 mm, so
four 8 mm short-axis slices centred at $z$ = 4, 12, 20 and 28 mm lie within
the myocardium throughout the cycle.

The default coefficients (`AFINAL`) are:

| $a_0$ | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |
| --- | --- | --- | --- | --- | --- |
| −19.6603 | 0.66731 | −0.0058944 | −0.15735 | 0.12142 | −0.036340 |

### References

* Moore, C. C., Lugo-Olivieri, C. H., McVeigh, E. R., & Zerhouni, E. A. (2000). Three-dimensional systolic strain patterns in the normal human left ventricle: characterization with tagged MR imaging. *Radiology*, 214(2), 453–466.
* Young, A. A., Kramer, C. M., Ferrari, V. A., Axel, L., & Reichek, N. (1994). Three-dimensional left ventricular deformation in hypertrophic cardiomyopathy. *Circulation*, 90(2), 854–867.
* Zhong, X., Spottiswoode, B. S., Meyer, C. H., Kramer, C. M., & Epstein, F. H. (2010). Imaging three-dimensional myocardial mechanics using navigator-gated volumetric spiral cine DENSE MRI. *Magnetic Resonance in Medicine*, 64(4), 1089–1097. [doi: 10.1002/mrm.22503](https://doi.org/10.1002/mrm.22503)
* Wang, V. Y., Casta, C., Zhu, Y.-M., Cowan, B. R., Croisille, P., Young, A. A., et al. (2016). Image-based investigation of human in vivo myofibre strain. *IEEE Transactions on Medical Imaging*, 35(11), 2486–2496.

### For citation

#### Results of the extension are published in: 
<blockquote>WIP</blockquote>

#### The original phantom was published in:
<blockquote>Verzhbinsky, I. A., Perotti, L. E., Moulin, K., Cork, T. E., Loecher, M., & Ennis, D. B. (2020). Estimating Aggregate Cardiomyocyte Strain Using In Vivo Diffusion and Displacement Encoded MRI. IEEE Transactions on Medical Imaging, 39(3), 656–667. 

[doi: 10.1109/TMI.2019.2933813](https://doi.org/10.1109/TMI.2019.2933813). </blockquote>

---
$\\[2cm]$

# CardiacKinematicsPhantom_py : Synthetic DENSE MRI Generator (Analytical Phantom)
Python and extended version of the mid-ventricular cardiac kinematics phantom available at: https://github.com/luigiemp/CardiacKinematicsPhantom

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a tool for generating and exporting synthetic cardiac DENSE (Displacement Encoding with Stimulated Echoes) MRI phantoms. It is highly configurable and features a robust, built-in DICOM exporter that produces ready-to-use DENSE series (AveMag, x/y/z phases), making it ideal for testing DENSE analysis pipelines and DICOM viewers.

The optimized analytical motion uses six coefficients. The sixth coefficient
adds an independent transmural contribution to the deformed cylindrical angle,
so all three analytical shear strains can be nonzero. The Jacobian objective
weight is zero by default, so incompressibility is not imposed.

---

## ✨ Features

* **Analytical Phantom Model**: Generates a synthetic left-ventricular phantom with configurable geometry, motion, and torsion.
* **Realistic DENSE Simulation**: Simulates DENSE imaging physics, including configurable encoding frequencies and signal-to-noise ratios (SNR).
* **Built-in DICOM Exporter**: The primary feature is its ability to export the generated data as a complete, multi-series DICOM set that is compatible with standard medical imaging software.
* **Multiple Output Formats**: In addition to DICOM, it saves visualizations (PNG), raw phase data (`.npy`, `.mat`), and geometry/displacement data for advanced visualization (VTK/VTU).

---

## 🚀 Quickstart Guide

### 1. Setup Environment

First, clone the repository, create a Python virtual environment, and install the required dependencies.

```bash
# Clone the repository
git clone <https://github.com/CBL-UCF/CardiacKinematicsPhantom_py>
cd AnalyticalPhantom

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Phantom

Open **`config.py`** in a text editor or python. All parameters for geometry, motion, imaging, and output are located here. Adjust them as needed for your simulation.

### 3. Run the Generator

Execute the main script to generate the phantom and all associated output files.

```bash
python main.py
```
After the script finishes, all outputs will be located in the `results/` directory.

### Plot analytical ground-truth strain only

To plot the six Green--Lagrange strain components directly from the analytical
deformation gradient, without DENSE encoding, voxel averaging, or noise, run:

```bash
python plot_analytic_strain_gt.py
```

This writes peak-systolic maps, temporal plots, and peak-systolic transmural
profiles for `ECC`, `ELL`, `ERR`, `ECL`, `ECR`, and `ELR` to
`results/analytic_strain_gt/`. The shear values are tensor components, not
engineering shear. By default the script uses `AFINAL` from `config.py`; pass
`--coefficients results/a_final.json` to use parameters previously produced by
the optimizer.

---

## ⚙️ Configuration (`config.py`)

All runtime parameters are centralized in `config.py`. Key settings are summarized below.

| Category             | Parameters                                           | Description                                      |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------ |
| **Paths** | `BASE_OUTPUT_PATH`                                    | The root directory for all generated files.      |
| **Geometry** | `R_ENDO`, `R_EPI`, `Z_BOTTOM`, `Z_TOP`             | Sets the endocardial/epicardial radii and axial extent. |
| **Motion** | `COMPUTE_PHANTOM`, `AFINAL`, `OPT_BETA`              | Toggles motion optimization and sets torsion parameters. |
| **Imaging** | `XLIM`, `YLIM`, `HX`, `HY`, `HZ`                    | Defines the Field of View (FOV) and voxel resolution. |
| **DENSE Encoding** | `KE_X`, `KE_Y`, `KE_Z`                       | Sets the DENSE encoding frequency (cycles/mm).   |
| **Noise** | `SNR`, `REPS`                                         | A list of SNRs and repetitions to simulate.      |
| **Outputs** | `SAVE_PHANTOM_VTK`, `TIME_STEPS`                    | Toggles VTK output and sets temporal sampling.   |

---

## 📁 Generated Outputs

The `main.py` script populates the `results/` directory with the following outputs:

* `results/dicom/`: **Primary Output**. Contains the generated DICOM series (`AveMag`, `x-encPha`, `y-encPha`, `z-encPha`).
* `results/images/`: PNG visualizations of the magnitude and wrapped phase images for each time step.
* `results/phasedata/`: Raw phase data saved as `.npy` and `.mat` files, along with a `phantom_data.json` summary.
* `results/vtk/`: VTK (`.vtu`) files containing the phantom geometry and displacement vectors for visualization in software like ParaView.

---

## 🏗️ Project Structure

```
AnalyticalPhantom/
├── main.py                   # Main executable script to run the generator
├── config.py                 # All runtime parameters and configuration
├── export_to_dicom.py        # DICOM writer and export logic
├── requirements.txt          # List of Python dependencies
├── templates/                # JSON templates for DICOM headers (required)
├── models/                   # Analytical, optimization, and noise models
├── utils/                    # Helper functions for mesh, displacement, etc.
└── results/                  # (It will be generated once the main.py is run)
```

---

## 📦 Dependencies

The core dependencies required to run the simulation are:

* `numpy`
* `scipy`
* `matplotlib`
* `Pillow`
* `pydicom` (Crucial for the DICOM export feature)

To install Python Dependencies, run:
```bash
pip install -r requirements.txt
```

## 📖 Citation

*(                                 )*

## 🔑 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
