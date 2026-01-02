import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms
import torch.nn as nn

# --- CONFIGURATION ---
MODEL_PATH = "forgery_detector_v1.pth"
SOURCE_DIR = "data/MICC_F600"
PATCH_SIZE = 128
STRIDE = 64
THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_AREA = 1000  # Minimum pixel cluster size to be considered a forgery


def multi_threshold_eval():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    all_files = os.listdir(SOURCE_DIR)
    images = [f for f in all_files if f.lower().endswith(('.png', '.jpg')) and '_gt' not in f.lower()]

    # Pixel stats (tp, fp, tn, fn) and Image-level stats (correct_images)
    stats = {t: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'img_correct': 0} for t in THRESHOLDS}

    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for f_name in tqdm(images):
        base = os.path.splitext(f_name)[0]
        mask_file = next((m for m in all_files if m.lower().startswith(base.lower() + "_gt")), None)

        img = cv2.imread(os.path.join(SOURCE_DIR, f_name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Ground Truth
        is_actually_tampered = False
        if mask_file:
            gt = (cv2.imread(os.path.join(SOURCE_DIR, mask_file), 0) > 128).astype(np.uint8)
            if np.any(gt): is_actually_tampered = True
        else:
            gt = np.zeros((h, w), dtype=np.uint8)

        # Generate Probabilities
        probs = np.zeros((h, w), dtype=np.float32)
        with torch.no_grad():
            for y in range(0, h - PATCH_SIZE, STRIDE):
                for x in range(0, w - PATCH_SIZE, STRIDE):
                    patch = img_rgb[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                    tensor = transform(patch).unsqueeze(0).to(DEVICE)
                    p = torch.softmax(model(tensor), dim=1)[0][1].item()
                    # Max pooling probabilities for smoother heatmap
                    probs[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = np.maximum(probs[y:y + PATCH_SIZE, x:x + PATCH_SIZE], p)

        for t in THRESHOLDS:
            # Create Binary Prediction
            pred = (probs > t).astype(np.uint8)

            # --- MODIFICATION: AREA FILTERING ---
            # Remove small noise clusters smaller than MIN_AREA
            num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
            for i in range(1, num_labels):
                if cc_stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                    pred[labels == i] = 0

            # --- PIXEL-LEVEL METRICS ---
            stats[t]['tp'] += np.logical_and(pred == 1, gt == 1).sum()
            stats[t]['fp'] += np.logical_and(pred == 1, gt == 0).sum()
            stats[t]['tn'] += np.logical_and(pred == 0, gt == 0).sum()
            stats[t]['fn'] += np.logical_and(pred == 0, gt == 1).sum()

            # --- MODIFICATION: IMAGE-LEVEL METRICS ---
            model_flagged_tampered = np.any(pred == 1)
            if model_flagged_tampered == is_actually_tampered:
                stats[t]['img_correct'] += 1

    print("\n| Threshold | Pixel Acc | Precision | Recall | F1-Score | Image-Level Acc |")
    print("|-----------|-----------|-----------|--------|----------|-----------------|")
    for t in THRESHOLDS:
        s = stats[t]
        prec = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
        rec = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        acc = (s['tp'] + s['tn']) / (s['tp'] + s['tn'] + s['fp'] + s['fn'])
        img_acc = (s['img_correct'] / len(images)) * 100
        print(f"| {t:<9} | {acc:.4f}    | {prec:.4f}    | {rec:.4f} | {f1:.4f}   | {img_acc:.2f}%          |")


if __name__ == "__main__":
    multi_threshold_eval()