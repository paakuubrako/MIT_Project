import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler  # <-- ADDED IMPORT
import time
import numpy as np  # <-- ADDED IMPORT


def create_loss_and_optimizer(net, learning_rate=0.001):
    """Creates loss function and optimizer."""
    loss = nn.CrossEntropyLoss()
    # Using Adam as discussed
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
    return loss, optimizer


def train_net(net, dataset, n_epochs=30, learning_rate=0.001,
              batch_size=32, val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net.to(device)

    # ----- Dataset Split -----
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    g = torch.Generator().manual_seed(0)  # reproducible split
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=g)

    # =========================================================================
    # ----- CRITICAL FIX: WEIGHTED RANDOM SAMPLER FOR IMBALANCE -----
    # =========================================================================

    print("Calculating weights for WeightedRandomSampler...")

    # 1. Extract labels for the training set only
    try:
        train_targets = np.array(dataset.targets)[train_set.indices]
    except AttributeError:
        print("Warning: Dataset.targets not found. Manually extracting training labels.")
        train_targets = [dataset.targets[i] for i in train_set.indices]
        train_targets = np.array(train_targets)

    # 2. Calculate weights: inverse of class frequency
    class_count = np.array([len(np.where(train_targets == t)[0]) for t in np.unique(train_targets)])
    weight = 1. / class_count

    # 3. Create a weight for every sample
    samples_weight = torch.from_numpy(weight[train_targets]).double()

    # 4. Create the Sampler (forces balanced batches)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # =========================================================================
    # ----- Update DataLoaders -----
    # =========================================================================

    # Use the sampler and set shuffle=False
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)

    # Validation loader remains the same
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ----- Training Loop Restoration (Fixes the NoneType Error) -----
    criterion, optimizer = create_loss_and_optimizer(net, learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    epoch_loss = []

    # Initialize variables to hold final metrics
    train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0

    for epoch in range(n_epochs):
        start_time = time.time()
        net.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # ----- Validation -----
        net.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                v_loss = criterion(outputs, labels)

                val_running_loss += v_loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_running_loss / len(val_loader)

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {time.time() - start_time:.1f}s"
        )

        epoch_loss.append(train_loss)

    # CRITICAL FIX: Return the metrics to solve the TypeError
    return train_loss, train_acc, val_loss, val_acc