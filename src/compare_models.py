
import json
import gc
from pathlib import Path

import torch
from src.architectures import DynamicCNN
from src.dataset import load_data

torch.set_num_threads(2)

MODELS = {
    "Original NAS": Path("models/best_nas_model.pth"),
    "Improved NAS": Path("models/improved_nas_model.pth")
}


def evaluate_checkpoint(path, test_loader):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = DynamicCNN(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    del model, checkpoint
    gc.collect()

    return round(100 * correct / total, 2)


def main():
    _, _, test_loader = load_data(batch_size=8)

    results = {}

    for name, path in MODELS.items():
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            continue

        print(f"\nEvaluating {name}...", flush=True)
        accuracy = evaluate_checkpoint(path, test_loader)

        results[name] = accuracy
        print(f"{name} Test Accuracy: {accuracy:.2f}%")

    if len(results) == 2:
        difference = (
            results["Improved NAS"] - results["Original NAS"]
        )

        print("\nMODEL COMPARISON")
        print(f"Original: {results['Original NAS']:.2f}%")
        print(f"Improved: {results['Improved NAS']:.2f}%")
        print(f"Difference: {difference:+.2f} percentage points")

    Path("results").mkdir(exist_ok=True)

    with open(
        "results/model_comparison.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
