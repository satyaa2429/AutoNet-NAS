
import csv
import gc
import json
import time
from pathlib import Path

import torch

from src.train import train_model


SEARCH_SPACE = [
    {
        "num_layers": 2,
        "filters": 16,
        "kernel_size": 3,
        "dropout": 0.2
    },
    {
        "num_layers": 3,
        "filters": 16,
        "kernel_size": 3,
        "dropout": 0.3
    },
    {
        "num_layers": 3,
        "filters": 32,
        "kernel_size": 5,
        "dropout": 0.3
    }
]


RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


def save_results(results, epochs):
    """Save all completed architecture results."""

    if not results:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ranked_results = sorted(
        results,
        key=lambda row: row["validation_accuracy"],
        reverse=True
    )

    csv_path = RESULTS_DIR / "nas_search_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(ranked_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(ranked_results)

    best = ranked_results[0]

    summary = {
        "best_configuration": {
            key: best[key]
            for key in (
                "num_layers",
                "filters",
                "kernel_size",
                "dropout"
            )
        },
        "best_validation_accuracy": best[
            "validation_accuracy"
        ],
        "candidate_count_completed": len(results),
        "epochs_per_candidate": epochs
    }

    with open(
        RESULTS_DIR / "best_architecture.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)


def run_search(epochs=2):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    best_accuracy = -1.0

    print("\n" + "=" * 60)
    print("AUTONET - NEURAL ARCHITECTURE SEARCH")
    print("=" * 60)
    print(f"Architectures: {len(SEARCH_SPACE)}")
    print(f"Epochs per architecture: {epochs}")
    print("Memory-efficient training enabled")

    for index, config in enumerate(
        SEARCH_SPACE,
        start=1
    ):
        print("\n" + "-" * 60)
        print(
            f"Training Architecture "
            f"{index}/{len(SEARCH_SPACE)}"
        )
        print(f"Configuration: {config}", flush=True)

        model = None
        start_time = time.time()

        try:
            model, history = train_model(
                config=config,
                epochs=epochs,
                learning_rate=0.001
            )

            validation_accuracy = history[-1][
                "val_accuracy"
            ]

            duration = time.time() - start_time

            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )

            result = {
                "architecture_id": index,
                "num_layers": config["num_layers"],
                "filters": config["filters"],
                "kernel_size": config["kernel_size"],
                "dropout": config["dropout"],
                "parameters": parameter_count,
                "validation_accuracy": validation_accuracy,
                "training_time_seconds": round(
                    duration, 2
                )
            }

            results.append(result)

            # Save each architecture's training history
            with open(
                RESULTS_DIR / f"cnn_{index}_history.json",
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(history, file, indent=4)

            # Save the best checkpoint
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.copy(),
                        "history": history,
                        "validation_accuracy": best_accuracy
                    },
                    MODELS_DIR / "best_nas_model.pth"
                )

                print("New best architecture found!")

            # Save completed results immediately
            save_results(results, epochs)

            print(
                f"CNN {index} completed | "
                f"Validation Accuracy: "
                f"{validation_accuracy:.2f}% | "
                f"Parameters: {parameter_count:,}"
            )

        except (RuntimeError, MemoryError) as error:
            print(
                f"\nCNN {index} failed: {error}",
                flush=True
            )
            print(
                "Skipping this architecture and "
                "continuing the search."
            )

            failures.append({
                "architecture_id": index,
                "configuration": config,
                "error": str(error)
            })

        finally:
            if model is not None:
                del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        with open(
            RESULTS_DIR / "failed_architectures.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(failures, file, indent=4)

    if not results:
        print(
            "\nNo architectures completed successfully."
        )
        return []

    results.sort(
        key=lambda row: row["validation_accuracy"],
        reverse=True
    )

    print("\n" + "=" * 60)
    print("NEURAL ARCHITECTURE SEARCH FINISHED")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank {rank} | "
            f"CNN {result['architecture_id']} | "
            f"Val Accuracy: "
            f"{result['validation_accuracy']:.2f}% | "
            f"Parameters: {result['parameters']:,}"
        )

    best = results[0]

    print("\nBest architecture configuration:")
    print(
        json.dumps(
            {
                "num_layers": best["num_layers"],
                "filters": best["filters"],
                "kernel_size": best["kernel_size"],
                "dropout": best["dropout"]
            },
            indent=4
        )
    )

    print(
        f"\nBest Validation Accuracy: "
        f"{best['validation_accuracy']:.2f}%"
    )

    print(
        f"Completed: {len(results)}/"
        f"{len(SEARCH_SPACE)} architectures"
    )

    print("Results saved in results/")
    print("Best checkpoint saved in models/")

    return results


if __name__ == "__main__":
    run_search(epochs=2)
