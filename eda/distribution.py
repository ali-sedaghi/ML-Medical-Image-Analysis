import os

import nibabel as nib
import numpy as np
from tqdm.notebook import tqdm


def get_class_distribution(train_dirs, num_samples=20):
    class_counts = {0: 0, 1: 0, 2: 0, 4: 0}

    print(f"Analyzing {num_samples} patients for class distribution...")

    for i in tqdm(range(num_samples)):
        path = train_dirs[i]
        try:
            patient_id = os.path.basename(path)
            seg_path = os.path.join(path, f"{patient_id}_seg.nii")
            seg = nib.load(seg_path).get_fdata()

            unique, counts = np.unique(seg, return_counts=True)
            for u, c in zip(unique, counts):
                if u in class_counts:
                    class_counts[u] += c
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

    return class_counts
