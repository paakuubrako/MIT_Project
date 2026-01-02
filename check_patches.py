import os
import random
import matplotlib.pyplot as plt
import cv2

PATCH_DIR = "data/patch_dataset/1"  # Looking at the forged folder


def visualize_forgeries():
    # Get list of all forged patches
    patch_files = [f for f in os.listdir(PATCH_DIR) if f.endswith('.png')]

    if not patch_files:
        print("No forged patches found! Check your extraction logic.")
        return

    # Select 16 random patches
    sample = random.sample(patch_files, min(len(patch_files), 16))

    plt.figure(figsize=(12, 12))
    for i, file_name in enumerate(sample):
        img_path = os.path.join(PATCH_DIR, file_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Patch {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_forgeries()