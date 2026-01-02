import glob
import cv2
import numpy as np
import os
from skimage.util import view_as_windows

# Assuming get_ref_df is defined elsewhere as in your snippet
from ..patch_extraction.extraction_utils import get_ref_df


def get_patches(image_mat, mask_mat, stride):
    """
    Extract patches from an image and assign labels based on the mask.
    :param image_mat: The image as a matrix (H, W, 3)
    :param mask_mat: The ground truth mask (H, W) where 255 = Forged, 0 = Authentic
    :param stride: The stride of the patch extraction process
    :returns: A tuple (patches, labels)
    """
    window_shape = (128, 128, 3)
    mask_window_shape = (128, 128)

    # Extract sliding windows for both the image and the mask
    img_windows = view_as_windows(image_mat, window_shape, step=stride)
    mask_windows = view_as_windows(mask_mat, mask_window_shape, step=stride)

    patches = []
    labels = []

    for m in range(img_windows.shape[0]):
        for n in range(img_windows.shape[1]):
            # Get the current image patch
            # windows shape is (rows, cols, 1, 128, 128, 3), so we take [m][n][0]
            patch = img_windows[m][n][0]

            # Get the corresponding mask patch
            mask_patch = mask_windows[m][n]

            # --- PATCH-LEVEL LABELING LOGIC ---
            # If the mask patch contains ANY forged pixels (255), label as Forged (1).
            # Otherwise, label as Authentic (0).
            if np.any(mask_patch == 255):
                labels.append(1)
            else:
                labels.append(0)

            patches.append(patch)

    return patches, labels


def get_images_and_labels(tampered_path, authentic_path, mask_path=None):
    """
    Get the images, their corresponding labels, and optional masks.
    :param tampered_path: Glob path for tampered images (e.g., 'path/to/tamp/*.jpg')
    :param authentic_path: Glob path for authentic images
    :param mask_path: Folder path containing the Ground Truth masks
    :returns: Dictionary with images, labels, and mask matrices
    """
    images = {}

    # 1. Process Authentic Images (Always Label 0)
    for im in glob.glob(authentic_path):
        images[im] = {
            'mat': cv2.imread(im),
            'label': 0,
            'mask': None  # No mask needed for fully authentic images
        }

    # 2. Process Tampered Images (Require masks for patch extraction)
    for im in glob.glob(tampered_path):
        img_mat = cv2.imread(im)
        h, w, _ = img_mat.shape

        # Load the mask if a path is provided
        mask_mat = None
        if mask_path:
            # Assuming mask filename matches image filename (common in MICC-F600)
            img_name = os.path.basename(im)
            # You may need to adjust this to match your mask filenames (e.g., .png instead of .jpg)
            potential_mask = os.path.join(mask_path, img_name.replace(".jpg", ".png"))

            if os.path.exists(potential_mask):
                mask_mat = cv2.imread(potential_mask, cv2.IMREAD_GRAYSCALE)
            else:
                # If no mask is found, default to a black mask (all authentic)
                mask_mat = np.zeros((h, w), dtype=np.uint8)

        images[im] = {
            'mat': img_mat,
            'label': 1,  # Image-level label
            'mask': mask_mat
        }

    return images