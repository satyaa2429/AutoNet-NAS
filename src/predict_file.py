"""Scan a local image with the same pipeline used in Vision Studio."""

import argparse
import json
from pathlib import Path

from PIL import Image

from src.inference import ROOT, load_cifar_classifier
from src.people import annotate_people
from src.vision import scan_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--mode", choices=("smart", "photo", "people", "cifar"), default="photo")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", type=Path, help="Save a copy with person boxes when detected")
    args = parser.parse_args(argv)
    if args.checkpoint and args.mode != "cifar":
        parser.error("--checkpoint is for CIFAR-10 models; use --mode cifar.")
    if args.output and args.output.resolve() == args.image.resolve():
        parser.error("Choose a different path for the marked image.")
    classifier = None
    if args.mode == "cifar":
        path = args.checkpoint or ROOT / "models" / "continued_cnn.pth"
        if args.checkpoint is None and not path.is_file():
            path = ROOT / "models" / "advanced_cnn.pth"
        classifier = load_cifar_classifier(path)
    with Image.open(args.image) as image:
        result = scan_image(image, args.mode, args.threshold, classifier=classifier)
        if args.output and result["kind"] == "people":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            marked = annotate_people(image, result["people"])
            marked.save(args.output)
            result["output_image"] = str(args.output)
    if args.mode in {"photo", "cifar"}:
        result = {"model": result["model"], "labels": result["dataset"], "predictions": result["ranked"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
