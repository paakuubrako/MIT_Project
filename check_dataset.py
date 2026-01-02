import os
import random
import matplotlib

# Use a non-interactive backend to avoid the TclError
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

FORGED_DIR = "data/patch_dataset/1"


def final_visual_check():
    files = [f for f in os.listdir(FORGED_DIR) if f.endswith('.png')]
    if not files:
        print("Error: No patches found in folder 1!")
        return

    sample = random.sample(files, min(len(files), 16))
    plt.figure(figsize=(10, 10))

    for i, f_name in enumerate(sample):
        img = cv2.imread(os.path.join(FORGED_DIR, f_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Patch {i}", fontsize=8)

    plt.tight_layout()
    # Save the result instead of showing it
    output_path = "verification_grid.png"
    plt.savefig(output_path)
    print(f"Success! Verification image saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    final_visual_check()