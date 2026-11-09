
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.advanced_architectures import AdvancedCNN


# ---------------- CONFIGURATION ----------------

SEED = 42
EPOCHS = 15
BATCH_SIZE = 8
TRAIN_SAMPLES = 16000
VAL_SAMPLES = 2000
LEARNING_RATE = 0.001

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_PATH = MODEL_DIR / "advanced_cnn.pth"
HISTORY_PATH = RESULTS_DIR / "advanced_history.json"
METRICS_PATH = RESULTS_DIR / "advanced_evaluation.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type == "cpu":
    torch.set_num_threads(2)


# ---------------- DATASET ----------------

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


def prepare_datasets():
    print("Loading CIFAR-10...", flush=True)

    training_source = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform
    )

    validation_source = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=eval_transform
    )

    test_source = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=eval_transform
    )

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(training_source),
        generator=generator
    ).tolist()

    train_indices = indices[:TRAIN_SAMPLES]

    val_indices = indices[
        TRAIN_SAMPLES:TRAIN_SAMPLES + VAL_SAMPLES
    ]

    train_loader = DataLoader(
        Subset(training_source, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        Subset(validation_source, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_source,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Training images: {TRAIN_SAMPLES}")
    print(f"Validation images: {VAL_SAMPLES}")
    print(f"Test images: {len(test_source)}")

    return train_loader, val_loader, test_loader


# ---------------- EVALUATION HELPER ----------------

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total
    )


# ---------------- TRAINING ----------------

def train_advanced():
    print("\nAUTONET - ADVANCED CNN TRAINING")
    print("=" * 50)
    print("Device:", DEVICE)

    train_loader, val_loader, test_loader = (
        prepare_datasets()
    )

    model = AdvancedCNN().to(DEVICE)

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {parameters:,}")

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

    best_val_accuracy = -1.0
    best_epoch = 0
    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item() * labels.size(0)
            )

            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_accuracy = (
            100.0 * train_correct / train_total
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 2),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 2)
        }

        history.append(record)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {val_accuracy:.2f}%",
            flush=True
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "architecture": "AdvancedCNN",
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "validation_accuracy": round(
                        best_val_accuracy, 2
                    ),
                    "num_classes": 10,
                    "train_samples": TRAIN_SAMPLES,
                    "val_samples": VAL_SAMPLES
                },
                MODEL_PATH
            )

            print(
                f"Saved best model: "
                f"{best_val_accuracy:.2f}%",
                flush=True
            )

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2)

    elapsed = time.time() - start_time

    # Load the best validation checkpoint.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Evaluate only after training is finished.
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    metrics = {
        "architecture": "AdvancedCNN",
        "training_images": TRAIN_SAMPLES,
        "validation_images": VAL_SAMPLES,
        "test_images": 10000,
        "epochs_completed": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_accuracy": round(
            best_val_accuracy, 2
        ),
        "test_accuracy": round(test_accuracy, 2),
        "test_loss": round(test_loss, 4),
        "parameters": parameters,
        "training_time_seconds": round(elapsed, 2)
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n" + "=" * 50)
    print("ADVANCED CNN TRAINING COMPLETED")
    print("=" * 50)
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )
    print(
        f"Test accuracy: {test_accuracy:.2f}%"
    )
    print(
        f"Training time: {elapsed:.1f} seconds"
    )
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    train_advanced()
