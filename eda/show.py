import matplotlib.pyplot as plt
import numpy as np


def show_slice(t1, t1ce, t2, flair, seg, slice_num):
    """
    Visualizes a specific slice index across all modalities.
    """
    _, ax = plt.subplots(1, 5, figsize=(20, 5))

    ax[0].imshow(t1[:, :, slice_num], cmap="gray")
    ax[0].set_title("T1")

    ax[1].imshow(t1ce[:, :, slice_num], cmap="gray")
    ax[1].set_title("T1ce")

    ax[2].imshow(t2[:, :, slice_num], cmap="gray")
    ax[2].set_title("T2")

    ax[3].imshow(flair[:, :, slice_num], cmap="gray")
    ax[3].set_title("FLAIR")

    masked_seg = np.ma.masked_where(seg[:, :, slice_num] == 0, seg[:, :, slice_num])
    ax[4].imshow(t1ce[:, :, slice_num], cmap="gray", alpha=0.6)  # Background image
    ax[4].imshow(masked_seg, cmap="prism", alpha=0.5)  # Overlay
    ax[4].set_title("Segmentation Overlay")

    for a in ax:
        a.axis("off")

    plt.show()
