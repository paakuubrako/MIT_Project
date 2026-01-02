import cv2
import numpy as np
from pathlib import Path


def extract_patches_grid(img, patch_size, stride):
    """Generate grid coordinates for patch extraction."""
    h, w = img.shape[:2]
    coords = []

    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            coords.append((x, y))

    if len(coords) == 0:
        # Image smaller than patch → center patch
        x = max(0, (w - patch_size) // 2)
        y = max(0, (h - patch_size) // 2)
        coords = [(x, y)]

    return coords


def extract_and_pad(img, x, y, patch_size):
    """Extract a patch and pad if needed."""
    patch = img[y:y + patch_size, x:x + patch_size]

    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.copyMakeBorder(
            patch,
            0, patch_size - patch.shape[0],
            0, patch_size - patch.shape[1],
            cv2.BORDER_REFLECT
        )

    return patch


def rotated_versions(patch, num_rotations):
    """Return rotated copies of patch."""
    if num_rotations == 1:
        return [patch]

    versions = []
    for r in range(num_rotations):
        versions.append(np.rot90(patch, k=r).copy())

    return versions
