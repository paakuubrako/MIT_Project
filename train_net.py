import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data/patch_dataset"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    # 1. Data Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset from Folders
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

    # 3. Handle Class Imbalance (Weighting)
    # We want to show the model 'Forged' patches more often because there are fewer of them
    targets = full_dataset.targets
    class_count = [len(os.listdir(os.path.join(DATA_DIR, '0'))),
                   len(os.listdir(os.path.join(DATA_DIR, '1')))]

    weights = 1. / torch.tensor(class_count, dtype=torch.float)
    samples_weights = weights[targets]

    sampler = WeightedRandomSampler(weights=samples_weights,
                                    num_samples=len(samples_weights),
                                    replacement=True)

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # 4. Define Model (Using a simple ResNet18 as a base)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary output: 0 or 1
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print(f"Starting training on {DEVICE}...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
            loop.set_postfix(loss=running_loss / len(train_loader), acc=100. * correct / total)

    # 6. Save the Model
    torch.save(model.state_dict(), "forgery_detector_v1.pth")
    print("Training complete! Model saved as forgery_detector_v1.pth")


if __name__ == "__main__":
    train_model()