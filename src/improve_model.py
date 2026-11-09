
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.architectures import DynamicCNN


# -------- SETTINGS --------
EPOCHS = 10
BATCH_SIZE = 8
TRAIN_SAMPLES = 12000
VAL_SAMPLES = 2000
LEARNING_RATE = 0.001

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(2)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}", flush=True)


# -------- IMAGE TRANSFORMS --------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


def prepare_data():
    train_full = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    val_full = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=val_transform
    )

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(train_full), generator=generator
    ).tolist()

    train_indices = indices[:TRAIN_SAMPLES]

    val_indices = indices[
        TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES
    ]

    train_data = Subset(train_full, train_indices)
    val_data = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")

    return train_loader, val_loader


def train_improved_model():
    checkpoint_path = (
        MODEL_DIR / "best_nas_model.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Run the NAS search before improved training."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    config = checkpoint["config"]

    print("\nSelected NAS architecture:")
    print(json.dumps(config, indent=2))

    train_loader, val_loader = prepare_data()

    # Fresh model using selected architecture.
    model = DynamicCNN(**config).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_accuracy = -1.0
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):

        # TRAINING
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item() * labels.size(0)
            )

            train_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = running_loss / train_total
        train_acc = 100 * train_correct / train_total

        # VALIDATION
        model.eval()

        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += (
                    loss.item() * labels.size(0)
                )

                val_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()

                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_total / val_total

        scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 2)
        }

        history.append(record)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.2f}% | "
            f"Validation: {val_acc:.2f}%",
            flush=True
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "history": history.copy(),
                    "validation_accuracy": best_accuracy
                },
                MODEL_DIR / "improved_nas_model.pth"
            )

            print(
                f"New best checkpoint: {val_acc:.2f}%",
                flush=True
            )

        # Keep progress after each epoch.
        with open(
            RESULTS_DIR / "improved_training_history.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2)

    elapsed = time.time() - start_time

    print("\nIMPROVED TRAINING COMPLETED")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Training time: {elapsed:.1f} seconds")
    print("Model: models/improved_nas_model.pth")


if __name__ == "__main__":
    train_improved_model()
