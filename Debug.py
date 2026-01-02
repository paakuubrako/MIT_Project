import os

SOURCE_DIR = "data/MICC_F600"
all_files = os.listdir(SOURCE_DIR)

# Separate them
images = [f for f in all_files if '_gt' not in f and f.lower().endswith(('.png', '.jpg'))]
masks = [f for f in all_files if '_gt' in f]

print(f"Total Images: {len(images)}")
print(f"Total Masks: {len(masks)}")

# Test the first few
print("\n--- Pairing Test ---")
for i in range(min(5, len(images))):
    img_name = images[i]
    base = os.path.splitext(img_name)[0]

    # What the code is looking for:
    expected_mask = base + "_gt"

    # Check if any mask starts with that
    found = [m for m in masks if m.startswith(expected_mask)]

    if found:
        print(f"✅ MATCH: {img_name} <---> {found[0]}")
    else:
        print(f"❌ FAIL:  {img_name} (Looking for something starting with '{expected_mask}')")