import os
import cv2
import numpy as np
import random
from tqdm import tqdm

SOURCE_DIR = "data/MICC_F600"
TARGET_DIR = "data/patch_dataset"
PATCH_SIZE = 128
PATCHES_PER_IMAGE = 50


def extract_patch_final():
    for label in ['0', '1']:
        os.makedirs(os.path.join(TARGET_DIR, label), exist_ok=True)

    all_files = os.listdir(SOURCE_DIR)
    # 1. Separate files based on your naming convention
    auth_only = [f for f in all_files if '_scale' in f.lower() or f.lower().startswith('erlangen')]
    masks = [f for f in all_files if '_gt' in f.lower()]
    # Tampered images are the ones left that aren't masks or the originals
    tampered_candidates = [f for f in all_files if
                           f not in auth_only and f not in masks and f.lower().endswith(('.png', '.jpg'))]

    print(f"Detected: {len(auth_only)} Authentic, {len(tampered_candidates)} Tampered images.")

    patch_counts = {0: 0, 1: 0}

    # Process all images
    for f_name in tqdm(auth_only + tampered_candidates):
        img = cv2.imread(os.path.join(SOURCE_DIR, f_name))
        if img is None: continue
        h, w = img.shape[:2]

        # Determine if we need to check a mask
        mask = None
        if f_name in tampered_candidates:
            base = os.path.splitext(f_name)[0]
            # Match the mask (e.g., 'beach' -> 'beach_gt.png')
            for m in masks:
                if m.lower().startswith(base.lower() + "_gt"):
                    mask = cv2.imread(os.path.join(SOURCE_DIR, m), cv2.IMREAD_GRAYSCALE)
                    break

        count = 0
        attempts = 0

        # If it's a tampered image, we force it to find Forged Patches
        while count < PATCHES_PER_IMAGE and attempts < 1000:
            attempts += 1
            y, x = random.randint(0, h - PATCH_SIZE), random.randint(0, w - PATCH_SIZE)

            label = 0
            if mask is not None:
                mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                # If there are white pixels, it's forged
                if np.max(mask_patch) > 10:
                    # Only take Label 1 if we still need them for this image
                    if count < (PATCHES_PER_IMAGE // 2):
                        label = 1
                    else:
                        continue  # We already have enough Label 1s, look for Label 0
                else:
                    # It's background
                    if count < (PATCHES_PER_IMAGE // 2):
                        continue  # We are still looking for Label 1s first

            # Save the patch
            p_name = f"{os.path.splitext(f_name)[0]}_p{count}.png"
            cv2.imwrite(os.path.join(TARGET_DIR, str(label), p_name), img[y:y + PATCH_SIZE, x:x + PATCH_SIZE])
            patch_counts[label] += 1
            count += 1

    print(f"\nSUCCESS! Found {patch_counts[0]} Authentic (0) and {patch_counts[1]} Forged (1) patches.")


if __name__ == "__main__":
    extract_patch_final()