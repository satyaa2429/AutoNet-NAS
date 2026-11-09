
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.architectures import DynamicCNN
from src.dataset import load_data


def train_model(
    config,
    epochs=2,
    learning_rate=0.001,
    save_path=None
):
    """
    Train a CNN architecture on CIFAR-10 with
    reduced memory usage for CPU-based systems.
    """

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cpu":
        torch.set_num_threads(2)

    print(f"\nUsing device: {device}", flush=True)

    # Reduced batch size to prevent memory allocation errors
    train_loader, val_loader, _ = load_data(batch_size=8)

    model = DynamicCNN(**config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # TRAINING
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            train_total += batch_size

        train_loss = running_loss / train_total
        train_accuracy = 100 * train_correct / train_total

        # VALIDATION
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)

                val_loss_sum += loss.item() * batch_size
                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                val_total += batch_size

        val_loss = val_loss_sum / val_total
        val_accuracy = 100 * val_correct / val_total

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 2),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 2)
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%",
            flush=True
        )

    elapsed = time.time() - start_time

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "history": history,
                "training_time_seconds": round(
                    elapsed, 2
                )
            },
            save_path
        )

        print(f"Model saved to: {save_path}")

    print(
        f"Total training time: {elapsed:.2f} seconds"
    )

    return model, history


if __name__ == "__main__":
    configuration = {
        "num_layers": 2,
        "filters": 16,
        "kernel_size": 3,
        "dropout": 0.2
    }

    train_model(
        config=configuration,
        epochs=2,
        learning_rate=0.001,
        save_path="models/first_cnn.pth"
    )
