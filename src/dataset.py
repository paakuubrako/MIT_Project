import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

# Suppress the libpng warnings in the console
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'


class MICCF600PatchDataset(Dataset):
    def __init__(self, data_dir, patch_size=128, stride=64, transform=None):
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = []
        self.targets = []

        valid_exts = ('.jpg', '.jpeg', '.png')
        all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)]

        # 1. Identify Masks and Originals
        mask_files = [f for f in all_files if "_gt" in f.lower()]
        # Everything else is a potential image
        image_files = [f for f in all_files if f not in mask_files]

        print(f"--- Dataset Scan ---")
        print(f"Total Images found: {len(image_files)}")
        print(f"Total Masks found: {len(mask_files)}")

        for f_name in image_files:
            img_path = os.path.join(data_dir, f_name)
            img_mat = cv2.imread(img_path)
            if img_mat is None: continue

            # 2. Try to find a matching mask
            base_name = os.path.splitext(f_name)[0]
            # Handle cases where image is 'erlangen1' and mask is 'erlangen1_gt'
            mask_found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = os.path.join(data_dir, base_name + "_gt" + ext)
                if os.path.exists(potential_mask):
                    mask_mat = cv2.imread(potential_mask, cv2.IMREAD_GRAYSCALE)
                    self._extract(img_mat, mask_mat, is_tampered=True)
                    mask_found = True
                    break

            # 3. IF NO MASK FOUND: Treat the whole image as Authentic (Label 0)
            if not mask_found:
                # Create a blank mask (all zeros)
                blank_mask = np.zeros((img_mat.shape[0], img_mat.shape[1]), dtype=np.uint8)
                self._extract(img_mat, blank_mask, is_tampered=False)

        print(f"Extraction complete: {len(self.patches)} patches total.")

    def _extract(self, img, mask, is_tampered):
        h, w, _ = img.shape
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                mask_patch = mask[y:y + self.patch_size, x:x + self.patch_size]

                # If we know it's tampered, check the mask.
                # If it's authentic, it's always label 0.
                if is_tampered:
                    label = 1 if np.any(mask_patch == 255) else 0
                else:
                    label = 0

                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                self.patches.append(Image.fromarray(patch_rgb))
                self.targets.append(label)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        target = self.targets[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch, target