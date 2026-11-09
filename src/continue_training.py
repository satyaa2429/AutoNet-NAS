"""Fine-tune the existing AdvancedCNN using more CIFAR-10 training images.

The original validation indices stay excluded from training. Validation, not
the test set, decides which checkpoint is kept. Original models stay intact.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.advanced_architectures import AdvancedCNN
from src.inference import CLASSES, ROOT, cifar_transform, file_sha256


def fixed_split(checkpoint, size=50000):
    """Reconstruct the split used by the user's original advanced trainer."""
    permutation = torch.randperm(size, generator=torch.Generator().manual_seed(42)).tolist()
    if "validation_indices" in checkpoint:
        validation = list(checkpoint["validation_indices"])
    else:
        start = int(checkpoint.get("train_samples", 16000))
        count = int(checkpoint.get("val_samples", 2000))
        validation = permutation[start:start + count]
    if not validation or len(set(validation)) != len(validation):
        raise ValueError("Validation indices are empty or duplicated.")
    if min(validation) < 0 or max(validation) >= size:
        raise ValueError("Validation indices are outside this dataset.")
    held_out = set(validation)
    training = [index for index in permutation if index not in held_out]
    if not training:
        raise ValueError("No training samples remain.")
    return training, validation


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    if not total:
        raise ValueError("Cannot evaluate an empty dataset.")
    return loss_sum / total, 100 * correct / total


def run(args):
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError("Use positive epochs/rate and batch size of at least two.")
    if args.patience < 1:
        raise ValueError("Patience must be at least one.")
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = args.checkpoint or (
        args.output if args.output.is_file() else ROOT / "models" / "advanced_cnn.pth"
    )
    if args.output.is_file() and source.resolve() != args.output.resolve():
        raise ValueError("The output already exists. Continue from it or choose a new --output path.")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "AdvancedCNN":
        raise ValueError("This trainer continues AdvancedCNN checkpoints only.")
    if tuple(checkpoint.get("classes", CLASSES)) != CLASSES:
        raise ValueError("Checkpoint class order is not CIFAR-10.")
    source_hash = file_sha256(source)
    augmentation = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    training_source = datasets.CIFAR10(str(args.data_dir), train=True, download=args.download, transform=augmentation)
    validation_source = datasets.CIFAR10(str(args.data_dir), train=True, transform=cifar_transform())
    if tuple(training_source.classes) != CLASSES:
        raise ValueError("Dataset classes do not match the checkpoint.")
    train_indices, val_indices = fixed_split(checkpoint, len(training_source))
    if args.train_samples:
        if not 0 < args.train_samples <= len(train_indices):
            raise ValueError(f"Train samples must be between 1 and {len(train_indices)}.")
        train_indices = train_indices[:args.train_samples]
    train_loader = DataLoader(
        Subset(training_source, train_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(Subset(validation_source, val_indices), batch_size=128, num_workers=0)
    model = AdvancedCNN(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    baseline_loss, baseline_accuracy = evaluate(model, val_loader, device)
    print(f"Device: {device} | CPU threads: {torch.get_num_threads()}", flush=True)
    print(f"Starting checkpoint: {source.name}", flush=True)
    print(f"Training: {len(train_indices):,} | Fixed validation: {len(val_indices):,}", flush=True)
    print(f"Starting validation accuracy: {baseline_accuracy:.2f}%", flush=True)
    best_accuracy = baseline_accuracy
    best_epoch = 0
    history = []
    output_metadata = {
        "architecture": "AdvancedCNN", "num_classes": 10,
        "classes": list(CLASSES), "image_size": 32,
        "mean": [0.5] * 3, "std": [0.5] * 3,
        "validation_indices": val_indices, "split_seed": 42,
        "train_samples": len(train_indices), "val_samples": len(val_indices),
        "source_checkpoint_sha256": source_hash,
    }

    def save_best(epoch, accuracy):
        atomic_checkpoint({
            **output_metadata,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "validation_accuracy": accuracy, "best_epoch": epoch,
        }, args.output)

    # Save the baseline too: an unsuccessful run cannot replace it with worse weights.
    save_best(0, baseline_accuracy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stale = 0
    start = time.monotonic()
    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            loss_sum = correct = total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; the saved best model is retained.")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                correct += int((logits.argmax(1) == labels).sum())
                total += len(labels)
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            scheduler.step()
            history.append({
                "epoch": epoch, "train_loss": round(loss_sum / total, 5),
                "train_accuracy": round(100 * correct / total, 4),
                "val_loss": round(val_loss, 5), "val_accuracy": round(val_accuracy, 4),
                "seconds": round(time.monotonic() - epoch_start, 2),
            })
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch, stale = val_accuracy, epoch, 0
                save_best(epoch, best_accuracy)
            else:
                stale += 1
            atomic_json(history, args.results_dir / "continued_history.json")
            print(
                f"Epoch {epoch}/{args.epochs} | train {100 * correct / total:.2f}% | "
                f"validation {val_accuracy:.2f}% | best {best_accuracy:.2f}% | "
                f"{time.monotonic() - epoch_start:.0f}s", flush=True,
            )
            if stale >= args.patience:
                print("Early stopping; keeping the best validation checkpoint.", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped training. Evaluating the saved best checkpoint.", flush=True)
    best = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    # The test set is first consulted after checkpoint selection has finished.
    test_source = datasets.CIFAR10(str(args.data_dir), train=False, download=args.download, transform=cifar_transform())
    test_loss, test_accuracy = evaluate(model, DataLoader(test_source, batch_size=128, num_workers=0), device)
    metrics = {
        "architecture": "AdvancedCNN", "dataset": "CIFAR-10",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_images": len(train_indices), "validation_images": len(val_indices),
        "test_images": len(test_source), "epochs_completed": len(history),
        "best_epoch": best_epoch, "interrupted": interrupted,
        "baseline_validation_accuracy": round(baseline_accuracy, 4),
        "best_validation_accuracy": round(best_accuracy, 4),
        "validation_improved": best_accuracy > baseline_accuracy,
        "test_accuracy": round(test_accuracy, 4), "test_loss": round(test_loss, 5),
        "checkpoint_sha256": file_sha256(args.output),
        "source_checkpoint_sha256": source_hash,
        "validation_split_sha256": sha256(json.dumps(val_indices).encode()).hexdigest(),
        "training_time_seconds": round(time.monotonic() - start, 2),
    }
    atomic_json(metrics, args.results_dir / "continued_evaluation.json")
    print(f"Best validation: {best_accuracy:.2f}% (started at {baseline_accuracy:.2f}%)", flush=True)
    print(f"Held-out test accuracy: {test_accuracy:.2f}%", flush=True)
    print(f"Saved: {args.output}", flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "continued_cnn.pth")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--train-samples", type=int, default=0, help="0 uses every non-validation training image")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--download", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
