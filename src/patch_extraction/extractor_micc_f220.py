import os
import cv2
from torch.utils.data import Dataset


def extract_micc_f220(input_dir, output_dir, patch_size=128, stride=128):
    """
    Extracts patches from MICC-F220 dataset (AU → 0, TU → 1).
    """

    os.makedirs(os.path.join(output_dir, "0"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "1"), exist_ok=True)

    count_auth = 0
    count_forg = 0

    for cls in ["AU", "TU"]:
        cls_dir = os.path.join(input_dir, cls)
        label = 0 if cls == "AU" else 1

        for filename in os.listdir(cls_dir):
            path = os.path.join(cls_dir, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            h, w, _ = img.shape

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = img[y:y+patch_size, x:x+patch_size]

                    out_path = os.path.join(output_dir, str(label), f"{cls}_{filename}_{x}_{y}.png")
                    cv2.imwrite(out_path, patch)

                    if label == 0:
                        count_auth += 1
                    else:
                        count_forg += 1

    print("MICC-F220 extraction complete")
    print("Authentic patches:", count_auth)
    print("Forged patches:", count_forg)
    print("Saved to:", output_dir)
