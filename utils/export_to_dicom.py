# save_dense_dicoms.py
import os, glob
import numpy as np
import scipy.io as sio
import pydicom
import json
from pathlib import Path
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from config import HX, HY, HZ, XLIM, YLIM, ZCONST, TIME_STEPS, KE_X, KE_Y, KE_Z, ZCONST, SNR

# Build a pydicom.Dataset from a JSON header
def build_dataset_from_header_json(json_path, generate_new_uids=True):
    """
    Load a small header JSON and return a pydicom.Dataset with file_meta filled.
    json_path: path to JSON produced by export_templates.py
    If generate_new_uids is True the function will create fresh Study/Series/SOP UIDs.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf8") as f:
        hdr = json.load(f)

    ds = Dataset()
    # copy simple attributes back to dataset
    for k, v in hdr.items():
        if k == "file_meta":
            continue
        setattr(ds, k, v)

    fm = FileMetaDataset()
    tsuid = hdr.get("file_meta", {}).get("TransferSyntaxUID")
    fm.TransferSyntaxUID = tsuid or ExplicitVRLittleEndian
    ds.file_meta = fm

    if generate_new_uids:
        # use fresh stable UIDs so repo JSONs don't leak original IDs
        ds.StudyInstanceUID = hdr.get("StudyInstanceUID") or generate_uid()
        ds.SeriesInstanceUID = hdr.get("SeriesInstanceUID") or generate_uid()
        ds.SOPInstanceUID = hdr.get("SOPInstanceUID") or generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    return ds

# Load a template (JSON)
def load_template(path_or_json):
    """
    Load a template which can be either:
      - a JSON header template (templates/*.json) produced by export_templates.py
      - a real DICOM file (.dcm)
    Returns a pydicom.Dataset.
    """
    p = Path(path_or_json)
    if p.suffix.lower() == ".json":
        if not p.exists():
            raise FileNotFoundError(f"JSON template not found: {p}")
        return build_dataset_from_header_json(p, generate_new_uids=True)
    else:
        # real DICOM fallback - keep behavior unchanged
        return pydicom.dcmread(str(p))


# Image geometry parameters
ROWS = int(round((YLIM[1] - YLIM[0]) / HY))
COLS = int(round((XLIM[1] - XLIM[0]) / HX))
PIXEL_SPACING_RC = [float(HY), float(HX)]     # [row, col]
SLICE_THICKNESS = float(HZ)
SLICE_LOCATION  = float(ZCONST)
NFRAMES = int(TIME_STEPS)

# Quantization functions
def quantize_phase_uint12(phase_wrapped):
    # ensure in [-0.5, 0.5) then map -> [0..4095] uint16
    a = (phase_wrapped + 0.5) % 1.0 - 0.5
    q = np.round((a + 0.5) * 4095.0)
    return np.clip(q, 0, 4095).astype(np.uint16)

def quantize_mag_uint12(mag_0_1, max_mag=4000):
    q = np.round(np.clip(mag_0_1, 0.0, 1.0) * max_mag)
    return np.clip(q, 0, 4095).astype(np.uint16)


def load_frames(mat_dir, stem):  # stem in ['mag','phaseX','phaseY','phaseZ']
    key = {'mag':'magnitude_wN','phaseX':'phaseX_wN','phaseY':'phaseY_wN','phaseZ':'phaseZ_wN'}[stem]
    files = sorted(glob.glob(os.path.join(mat_dir, f"{stem}_*.mat")))
    frames = [np.array(sio.loadmat(fp)[key], dtype=np.float32) for fp in files]
    return frames

# --- core DICOM writer using a template ---
def write_series_from_template(
    frames_float, template_path, out_dir,
    series_number, series_description, *,
    is_phase, enc_freq=None,
    swap_xy=False, negate_xyz=(0,0,0),
    largest_mag=4000, study_uid=None
):
    os.makedirs(out_dir, exist_ok=True)
    # tpl = pydicom.dcmread(template_path)
    tpl = load_template(template_path)
    # Fix StudyInstanceUID if it ends with a dot (invalid DICOM UID)
    if str(tpl.StudyInstanceUID).endswith('.'):
        tpl.StudyInstanceUID = generate_uid()
    if study_uid is None:
        study_uid = tpl.StudyInstanceUID

    series_uid = generate_uid()
    
    neg_x, neg_y, neg_z = map(int, negate_xyz)

    for i, arr in enumerate(frames_float, start=1):
        A = arr.copy()
        if is_phase:
            # Choose the correct axis negation at call time (x/y/z series)
            # Here we only apply sign once (caller decides which index is relevant)
            pass

        # quantize
        if is_phase:
            pix = quantize_phase_uint12(A)
            lpv = 4095
        else:
            pix = quantize_mag_uint12(A, max_mag=largest_mag)
            lpv = largest_mag

        # swap XY if requested (transpose data and swap pixel spacing order)
        if swap_xy:
            pix = pix.T
            pixel_spacing = [PIXEL_SPACING_RC[1], PIXEL_SPACING_RC[0]]
            rows, cols = COLS, ROWS
        else:
            pixel_spacing = PIXEL_SPACING_RC
            rows, cols = ROWS, COLS

        ds = tpl.copy()
        
        # geometry & matrix
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelSpacing = pixel_spacing
        ds.SliceThickness = str(SLICE_THICKNESS)
        ds.SliceLocation  = str(SLICE_LOCATION)

        # timing / counts
        ds.CardiacNumberOfImages = str(NFRAMES)

        # pixel format: 12-bit in 16 container
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SmallestImagePixelValue = int(pix.min())
        ds.LargestImagePixelValue  = int(lpv)

        # series identity
        ds.StudyInstanceUID  = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber      = str(series_number)
        ds.SeriesDescription = series_description
        ds.InstanceNumber    = str(i)

        # comments (EncFreq / swap / negate flags)
        if is_phase:
            # RCSflip:x/y/z encode your negate flags; RCswap encodes swap_xy
            ds.ImageComments = (
                f"DENSE {series_description} - Scale:1.000000 "
                f"EncFreq:{enc_freq:.2f} RCswap:{int(swap_xy)} "
                f"RCSflip:{neg_x}/{neg_y}/{neg_z} Phs:{i-1}/{NFRAMES}"
            )
        else:
            ds.ImageComments = f"DENSE overall mag - RCswap:{int(swap_xy)} RCSflip:{neg_x}/{neg_y}/{neg_z} Phs:{i-1}/{NFRAMES}"

        # fresh SOP Instance
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # pixels
        ds.PixelData = pix.tobytes(order="C")

        ds.save_as(os.path.join(out_dir, f"Image_{i:04d}.dcm"), enforce_file_format=True)

#  Save all four series: mag, x-phase, y-phase, z-phase
def save_all_series(mat_dir, out_root, tpl_mag, tpl_x, tpl_y, tpl_z,
                    swap_xy=False, neg_x=False, neg_y=False, neg_z=False):
    mag_frames  = load_frames(mat_dir, 'mag')
    x_frames    = load_frames(mat_dir, 'phaseX')  # already wrapped [-0.5,0.5]
    y_frames    = load_frames(mat_dir, 'phaseY')
    z_frames    = load_frames(mat_dir, 'phaseZ')

    # Prefer the StudyInstanceUID from the JSON template (or DICOM fallback)
    study_ds = load_template(tpl_mag)
    study_uid = getattr(study_ds, "StudyInstanceUID", None) or generate_uid()

    # Write each series
    write_series_from_template(mag_frames, tpl_mag, os.path.join(out_root, "AveMag"),
        series_number=103001, series_description="AveMag",
        is_phase=False, enc_freq=None, swap_xy=swap_xy,
        negate_xyz=(0,0,0), largest_mag=4000, study_uid=study_uid)

    write_series_from_template(x_frames, tpl_x, os.path.join(out_root, "x-encPha"),
        series_number=104001, series_description="x-encPha",
        is_phase=True, enc_freq=float(KE_X), swap_xy=swap_xy,
        negate_xyz=(int(neg_x), int(neg_y), int(neg_z)), study_uid=study_uid)

    write_series_from_template(y_frames, tpl_y, os.path.join(out_root, "y-encPha"),
        series_number=105001, series_description="y-encPha",
        is_phase=True, enc_freq=float(KE_Y), swap_xy=swap_xy,
        negate_xyz=(int(neg_x), int(neg_y), int(neg_z)), study_uid=study_uid)

    write_series_from_template(z_frames, tpl_z, os.path.join(out_root, "z-encPha"),
        series_number=106001, series_description="z-encPha",
        is_phase=True, enc_freq=float(KE_Z), swap_xy=swap_xy,
        negate_xyz=(int(neg_x), int(neg_y), int(neg_z)), study_uid=study_uid)


def save_dense_dicoms(
    mat_dir=None,
    out_root=None,
    tpl_mag="templates/AveMag_template.json",
    tpl_x="templates/x-encPha_template.json",
    tpl_y="templates/y-encPha_template.json",
    tpl_z="templates/z-encPha_template.json",
    swap_xy: bool = False,
    neg_x: bool = False,
    neg_y: bool = False,
    neg_z: bool = False,
    snr_value=None,
    z_position=None,
    rep: int = 0
):
    from config import SNR, ZCONST
    if snr_value is None:
        snr_value = SNR[0] if isinstance(SNR, (list, tuple)) else SNR
    if z_position is None:
        z_position = ZCONST
    if mat_dir is None:
        mat_dir = f"results/phasedata/matlab/Zpos_{z_position}/SNR_{snr_value}/Rep_{rep}"
    if out_root is None:
        out_root = f"results/dicom/SNR{snr_value}_Rep{rep}/Zpos_{z_position}"

    save_all_series(mat_dir, out_root, tpl_mag, tpl_x, tpl_y, tpl_z,
                    swap_xy=swap_xy, neg_x=neg_x, neg_y=neg_y, neg_z=neg_z)


