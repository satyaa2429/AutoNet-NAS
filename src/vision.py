"""Select person detection or image classification for the app and CLI."""

from src.inference import (
    load_photo_classifier, predict_scores, rank_predictions,
)
from src.people import detect_people, load_person_detector


def scan_image(
    image, mode="smart", threshold=0.55, classifier=None, detector=None,
    *, photo_classifier_loader=None,
):
    if mode not in {"smart", "photo", "people", "cifar"}:
        raise ValueError("Unknown scan mode.")
    if mode in {"smart", "people"}:
        detector = detector or load_person_detector()
        people = detect_people(image, detector, threshold)
        if people or mode == "people":
            return {
                "kind": "people" if people else "no_people", "mode": mode,
                "people": people, "model": detector.name,
                "threshold": threshold,
            }
    if classifier is None:
        if mode == "cifar":
            raise ValueError("Choose a trained CIFAR-10 checkpoint.")
        classifier = (photo_classifier_loader or load_photo_classifier)()
    scores = predict_scores(image, classifier)
    return {
        "kind": "classification", "mode": mode,
        "ranked": rank_predictions(scores, classifier.classes),
        "model": classifier.name, "dataset": classifier.dataset,
        "people_checked": mode == "smart",
    }
