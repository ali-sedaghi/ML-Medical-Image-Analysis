import os

import nibabel as nib


def load_patient_data(patient_path):
    """
    Loads the 4 modalities and the segmentation mask for a single patient.
    """
    # Extract patient ID from path
    patient_id = os.path.basename(patient_path)

    t1_path = os.path.join(patient_path, f"{patient_id}_t1.nii")
    t1ce_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii")
    t2_path = os.path.join(patient_path, f"{patient_id}_t2.nii")
    flair_path = os.path.join(patient_path, f"{patient_id}_flair.nii")
    seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")

    t1 = nib.load(t1_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    t2 = nib.load(t2_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    return t1, t1ce, t2, flair, seg
