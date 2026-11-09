"""Shared prediction code for the app and command-line checks.

The CIFAR-10 experiment and the pretrained ImageNet classifier have different
labels and preprocessing. They deliberately use separate model loaders.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.advanced_architectures import AdvancedCNN
from src.architectures import DynamicCNN

ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
PHOTO_WEIGHTS = ROOT / "models" / "photo_efficientnet_b0.pth"


@dataclass
class Classifier:
    model: torch.nn.Module
    transform: Callable
    classes: tuple[str, ...]
    name: str
    dataset: str


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(image):
    """Respect phone orientation before converting to the model's RGB input."""
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA" or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def cifar_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def load_cifar_classifier(path):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path.name} is not an AutoNet model checkpoint.")
    classes = tuple(checkpoint.get("classes", CLASSES))
    if classes != CLASSES or int(checkpoint.get("num_classes", 10)) != 10:
        raise ValueError("This checkpoint does not use the CIFAR-10 label order.")
    if (tuple(checkpoint.get("mean", (0.5,) * 3)) != (0.5,) * 3
            or tuple(checkpoint.get("std", (0.5,) * 3)) != (0.5,) * 3
            or checkpoint.get("image_size", 32) != 32):
        raise ValueError("Checkpoint preprocessing differs from this CIFAR-10 pipeline.")
    if "config" in checkpoint:
        model = DynamicCNN(**checkpoint["config"])
    elif checkpoint.get("architecture") == "AdvancedCNN":
        model = AdvancedCNN(num_classes=10)
    else:
        raise ValueError("Unknown checkpoint architecture; no model was guessed.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return Classifier(model, cifar_transform(), classes, path.stem, "CIFAR-10")


def load_photo_classifier(path=PHOTO_WEIGHTS):
    """Use the published ImageNet labels AND transforms for these weights."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    path = Path(path)
    if path.is_file():
        model = efficientnet_b0(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        # Torchvision verifies the published hash when downloading the weights.
        model = efficientnet_b0(weights=weights)
    model.eval()
    return Classifier(
        model, weights.transforms(), tuple(weights.meta["categories"]),
        "EfficientNet-B0 (pretrained)", "ImageNet-1K",
    )


def predict_scores(image, classifier):
    tensor = classifier.transform(prepare_image(image)).unsqueeze(0)
    with torch.inference_mode():
        logits = classifier.model(tensor)
        if logits.shape != (1, len(classifier.classes)):
            raise ValueError("Model output and class labels have different sizes.")
        if not torch.isfinite(logits).all():
            raise ValueError("The model produced non-finite values.")
        return torch.softmax(logits, dim=1)[0].cpu()


def rank_predictions(scores, classes, top_k=5):
    values, indices = torch.topk(scores, k=min(top_k, len(classes)))
    return [
        {"class": classes[index], "score_percent": round(value * 100, 4)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]
