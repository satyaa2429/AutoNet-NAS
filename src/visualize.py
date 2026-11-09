
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
CHARTS_DIR = RESULTS_DIR / "charts"


def generate_visualizations():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "nas_search_results.csv"

    if not csv_path.exists():
        print("NAS search results not found.")
        print("Run: python -u -m src.search")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("The results file is empty.")
        return

    # Use consistent architecture labels
    df["model_name"] = (
        "CNN " + df["architecture_id"].astype(str)
    )

    # 1. Validation accuracy comparison
    plt.figure(figsize=(9, 5))
    plt.bar(
        df["model_name"],
        df["validation_accuracy"]
    )
    plt.xlabel("CNN Architecture")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("NAS - CNN Validation Accuracy Comparison")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "accuracy_comparison.png",
        dpi=200
    )
    plt.close()

    # 2. Parameter count comparison
    plt.figure(figsize=(9, 5))
    plt.bar(
        df["model_name"],
        df["parameters"]
    )
    plt.xlabel("CNN Architecture")
    plt.ylabel("Trainable Parameters")
    plt.title("NAS - Model Complexity Comparison")
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "parameter_comparison.png",
        dpi=200
    )
    plt.close()

    # 3. Training time comparison
    plt.figure(figsize=(9, 5))
    plt.bar(
        df["model_name"],
        df["training_time_seconds"]
    )
    plt.xlabel("CNN Architecture")
    plt.ylabel("Training Time (seconds)")
    plt.title("NAS - Training Time Comparison")
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "training_time_comparison.png",
        dpi=200
    )
    plt.close()

    # 4. Accuracy vs model complexity
    plt.figure(figsize=(9, 5))

    for _, row in df.iterrows():
        plt.scatter(
            row["parameters"],
            row["validation_accuracy"],
            s=100
        )

        plt.annotate(
            row["model_name"],
            (
                row["parameters"],
                row["validation_accuracy"]
            ),
            xytext=(6, 6),
            textcoords="offset points"
        )

    plt.xlabel("Trainable Parameters")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("NAS - Accuracy vs Model Complexity")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "accuracy_vs_parameters.png",
        dpi=200
    )
    plt.close()

    # 5. Training and validation accuracy
    plt.figure(figsize=(9, 5))
    found_history = False

    for architecture_id in df["architecture_id"]:
        history_path = RESULTS_DIR / (
            f"cnn_{architecture_id}_history.json"
        )

        if not history_path.exists():
            continue

        with open(
            history_path,
            "r",
            encoding="utf-8"
        ) as file:
            history = json.load(file)

        epochs = [
            item["epoch"] for item in history
        ]
        val_accuracy = [
            item["val_accuracy"] for item in history
        ]

        plt.plot(
            epochs,
            val_accuracy,
            marker="o",
            label=f"CNN {architecture_id}"
        )
        found_history = True

    if found_history:
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("NAS - Validation Accuracy by Epoch")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            CHARTS_DIR / "validation_accuracy_curves.png",
            dpi=200
        )

    plt.close()

    print("\nNAS visualizations generated successfully!")
    print(f"Charts saved in: {CHARTS_DIR.resolve()}")


if __name__ == "__main__":
    generate_visualizations()
