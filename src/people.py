"""Local person detection using Torchvision's published COCO weights."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import ImageDraw, ImageFont
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from src.inference import ROOT, prepare_image

PEOPLE_WEIGHTS = ROOT / "models" / "people_ssdlite320.pth"


@dataclass
class PersonDetector:
    model: torch.nn.Module
    transform: Callable
    person_label: int
    name: str = "SSDLite MobileNetV3"


def load_person_detector(path=PEOPLE_WEIGHTS):
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    path = Path(path)
    categories = tuple(weights.meta["categories"])
    if path.is_file():
        # Disable the backbone download: the complete detector is already local.
        model = ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=len(categories),
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        model = ssdlite320_mobilenet_v3_large(weights=weights, weights_backbone=None)
    model.eval()
    return PersonDetector(model, weights.transforms(), categories.index("person"))


def detect_people(image, detector, threshold=0.55):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The detection threshold must be between zero and one.")
    prepared = prepare_image(image)
    width, height = prepared.size
    tensor = detector.transform(prepared)
    with torch.inference_mode():
        prediction = detector.model([tensor])[0]
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    people = []
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != detector.person_label or not torch.isfinite(score):
            continue
        if float(score) < threshold or not torch.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        people.append({
            "label": "Person", "score_percent": round(float(score) * 100, 4),
            "box": [round(v, 2) for v in (x1, y1, x2, y2)],
        })
    return sorted(people, key=lambda item: item["score_percent"], reverse=True)


def annotate_people(image, people):
    annotated = prepare_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    font_size = max(14, min(32, round(min(width, height) * 0.035)))
    font = None
    for candidate in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            pass
    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
    line_width = max(2, round(min(width, height) * 0.006))
    for index, person in enumerate(people, 1):
        x1, y1, x2, y2 = person["box"]
        box = (x1, y1, x2, y2)
        draw.rectangle(box, outline="#FFFFFF", width=2 * line_width)
        draw.rectangle(box, outline="#171717", width=line_width)
        label = f"Person {index}  {person['score_percent']:.0f}%"
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width = bounds[2] - bounds[0] + 14
        label_height = bounds[3] - bounds[1] + 12
        left = max(0, min(x1, width - label_width))
        top = max(0, y1 - label_height)
        draw.rounded_rectangle(
            (left, top, left + label_width, top + label_height),
            radius=4, fill="#171717",
        )
        draw.text((left + 7, top + 6 - bounds[1]), label, font=font, fill="white")
    return annotated
