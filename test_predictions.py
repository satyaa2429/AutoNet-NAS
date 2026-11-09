"""Evaluate the current saved CNN on the full CIFAR-10 test set.

For a normal photo: python -m src.predict_file path/to/photo.jpg
"""

from src.verify_predictions import main


if __name__ == "__main__":
    main()
