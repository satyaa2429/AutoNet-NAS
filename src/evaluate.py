
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.architectures import DynamicCNN
from src.dataset import load_data


MODEL_PATH = Path("models/best_nas_model.pth")
RESULTS_DIR = Path("results")
CHARTS_DIR = RESULTS_DIR / "charts"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def evaluate_best_model():

    if not MODEL_PATH.exists():
        print("Best NAS model not found.")
        print("First run: python -u -m src.search")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.set_num_threads(2)

    print("\nLoading best NAS model...")

    # Load only our locally generated trusted checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    config = checkpoint["config"]

    model = DynamicCNN(**config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Selected configuration:")
    print(json.dumps(config, indent=4))

    print("\nLoading CIFAR-10 test dataset...")

    _, _, test_loader = load_data(batch_size=8)

    all_predictions = []
    all_labels = []

    print("Evaluating on 10,000 test images...")

    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    # Calculate overall test accuracy
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    # Calculate per-class precision, recall and F1
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # Calculate confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(10))
    )

    # Save evaluation metrics
    evaluation = {
        "test_accuracy_percent": round(
            accuracy * 100, 2
        ),
        "total_test_images": len(all_labels),
        "best_configuration": config,
        "classification_report": report
    }

    with open(
        RESULTS_DIR / "test_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(evaluation, file, indent=4)

    # Generate confusion matrix visualization
    plt.figure(figsize=(11, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title("AutoNet-NAS: CIFAR-10 Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR / "confusion_matrix.png",
        dpi=200
    )

    plt.close()

    print("\n" + "=" * 55)
    print("FINAL MODEL EVALUATION")
    print("=" * 55)

    print(f"Test Images: {len(all_labels)}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(10)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nResults saved successfully!")
    print("results/test_evaluation.json")
    print("results/charts/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_best_model()
