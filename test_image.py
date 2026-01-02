import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "forgery_detector_v1.pth"
INPUT_DIR = "data/Test"  # Your source folder
OUTPUT_DIR = "test_results"
PATCH_SIZE = 128
STRIDE = 64  # Increased stride for faster batch processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_process():
    # 1. Setup folders
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Load the Model
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Get list of images (excluding masks)
    all_files = os.listdir(INPUT_DIR)
    images = [f for f in all_files if f.lower().endswith(('.png', '.jpg')) and '_gt' not in f.lower()]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Starting batch process for {len(images)} images...")

    for f_name in tqdm(images):
        img_path = os.path.join(INPUT_DIR, f_name)
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            for y in range(0, h - PATCH_SIZE, STRIDE):
                for x in range(0, w - PATCH_SIZE, STRIDE):
                    patch = img_rgb[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                    input_tensor = transform(patch).unsqueeze(0).to(DEVICE)

                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    heatmap[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += probs[0][1].item()

        # Save visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Original: {f_name}")

        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.axis('off')
        plt.title("Detection Heatmap")

        plt.savefig(os.path.join(OUTPUT_DIR, f"result_{f_name}"))
        plt.close()  # Important to clear memory between images

    print(f"\nAll results saved to the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    batch_process()