"""Re-evaluate actual saved checkpoints on labelled CIFAR-10 test images."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from src.inference import CLASSES, ROOT, file_sha256, load_cifar_classifier


def evaluate_classifier(classifier, loader):
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    classifier.model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predicted = classifier.model(images).argmax(1).cpu()
            matrix += torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
    counts = matrix.sum(1)
    total = int(counts.sum())
    if not total:
        raise ValueError("The evaluation dataset is empty.")
    return {
        "correct": int(matrix.diag().sum()),
        "total": total,
        "test_accuracy": round(100 * int(matrix.diag().sum()) / total, 4),
        "per_class_accuracy": {
            label: round(100 * int(matrix[i, i]) / int(counts[i]), 4)
            if int(counts[i]) else None
            for i, label in enumerate(CLASSES)
        },
        "confusion_matrix": matrix.tolist(),
    }


def save_examples(classifier, raw_dataset, destination, checkpoint_path):
    # A fixed random sample, including mistakes; never select only successes.
    indices = torch.randperm(
        len(raw_dataset), generator=torch.Generator().manual_seed(42)
    )[:24].tolist()
    predictions = []
    canvas = Image.new("RGB", (6 * 170, 4 * 177), "#f4f6fa")
    draw = ImageDraw.Draw(canvas)
    example_dir = destination / "prediction_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    for position, index in enumerate(indices):
        image, target = raw_dataset[index]
        tensor = classifier.transform(image).unsqueeze(0)
        with torch.inference_mode():
            scores = torch.softmax(classifier.model(tensor), 1)[0]
        predicted = int(scores.argmax())
        item = {
            "dataset_index": index, "actual": CLASSES[target],
            "predicted": CLASSES[predicted], "correct": predicted == target,
        }
        predictions.append(item)
        image.save(example_dir / f"test_{index:05d}_{CLASSES[target]}.png")
        x, y = (position % 6) * 170, (position // 6) * 177
        canvas.paste(image.resize((128, 128), Image.Resampling.NEAREST), (x + 18, y + 4))
        draw.text((x + 6, y + 136), f"Actual: {CLASSES[target]}", fill="black")
        draw.text(
            (x + 6, y + 153), f"Model: {CLASSES[predicted]}",
            fill="#167542" if predicted == target else "#b52424",
        )
    canvas.save(destination / "prediction_examples.png")
    (destination / "prediction_examples.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    (destination / "prediction_examples_model.json").write_text(json.dumps({
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "sampling": "24 random test images, seed 42; no filtering by correctness",
    }, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--all", action="store_true", help="Verify all NAS/CNN checkpoints")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    if args.checkpoint and args.all:
        parser.error("Use either --checkpoint or --all.")
    paths = [ROOT / "models" / name for name in (
        "best_nas_model.pth", "improved_nas_model.pth", "advanced_cnn.pth", "continued_cnn.pth"
    )]
    if args.checkpoint:
        paths = [args.checkpoint]
    elif not args.all:
        paths = [next((p for p in reversed(paths) if p.is_file()), paths[-2])]
    else:
        paths = [p for p in paths if p.is_file()]
    if not paths:
        parser.error("No model checkpoints were found.")
    raw = CIFAR10(str(args.data_dir), train=False, download=args.download)
    if tuple(raw.classes) != CLASSES:
        raise ValueError("Dataset label order differs from the trained models.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "CIFAR-10 test", "created_at": datetime.now(timezone.utc).isoformat(), "models": {}}
    final_classifier = None
    for path in paths:
        print(f"Verifying {path.name} on {len(raw):,} labelled test images...", flush=True)
        classifier = load_cifar_classifier(path)
        dataset = CIFAR10(str(args.data_dir), train=False, transform=classifier.transform)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        result = evaluate_classifier(classifier, loader)
        result["checkpoint_sha256"] = file_sha256(path)
        result["classes"] = list(CLASSES)
        report["models"][path.name] = result
        print(f"  Correct: {result['correct']}/{result['total']} ({result['test_accuracy']:.2f}%)", flush=True)
        with (args.output_dir / f"{path.stem}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual / predicted", *CLASSES])
            writer.writerows([label, *row] for label, row in zip(CLASSES, result["confusion_matrix"]))
        final_classifier = classifier
    # This is an evidence report, not a criterion for model selection.
    save_examples(final_classifier, raw, args.output_dir, paths[-1])
    path = args.output_dir / "verified_evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", path, flush=True)
    print("Sample grid:", args.output_dir / "prediction_examples.png", flush=True)


if __name__ == "__main__":
    main()
