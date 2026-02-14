import os

import numpy as np
import pandas as pd
from loader import load_patient_data
from tqdm.notebook import tqdm


def extract_patient_stats(train_dirs, num_samples=None):
    """
    Calculates shape and mean intensity per channel for a list of patients.
    If num_samples is None, it runs on all patients.
    """
    stats_list = []

    # Use a subset if specified, otherwise use all
    dirs_to_process = train_dirs[:num_samples] if num_samples else train_dirs

    print(f"Processing {len(dirs_to_process)} patients...")

    for patient_path in tqdm(dirs_to_process):
        patient_id = os.path.basename(patient_path)

        t1, t1ce, t2, flair, seg = load_patient_data(patient_path)

        # 1. Get Shape (All channels usually have the same shape in BraTS)
        # We check T1 as representative
        vol_shape = t1.shape

        # 2. Calculate Mean per Channel
        # Note: We often care about the mean of the 'brain' area, excluding the
        # black background (0). Here we calculate the mean of non-zero pixels
        # to get a more accurate representation of the tissue intensity.

        stats = {
            "Patient_ID": patient_id,
            "Shape": str(vol_shape),
            "T1_mean": np.mean(t1[t1 > 0]),
            "T1ce_mean": np.mean(t1ce[t1ce > 0]),
            "T2_mean": np.mean(t2[t2 > 0]),
            "FLAIR_mean": np.mean(flair[flair > 0]),
        }
        stats_list.append(stats)

    return pd.DataFrame(stats_list)
