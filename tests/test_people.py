"""Person detection regressions without downloading data or models."""

import pytest
import torch
from PIL import Image

from src.people import annotate_people, detect_people, load_person_detector


@pytest.fixture(scope="module")
def detector():
    torch.set_num_threads(2)
    return load_person_detector()


def test_blank_image_does_not_create_a_person(detector):
    assert detect_people(Image.new("RGB", (400, 400), "white"), detector) == []


def test_person_annotations_preserve_source_photo():
    image = Image.new("RGB", (240, 300), "white")
    before = image.tobytes()
    output = annotate_people(image, [{"box": [40, 50, 200, 270], "score_percent": 95.0}])
    assert image.tobytes() == before
    assert output.size == image.size
    assert output.getpixel((40, 150)) != image.getpixel((40, 150))


def test_smart_scan_avoids_an_object_guess_when_a_person_is_found(monkeypatch):
    from src import vision
    from types import SimpleNamespace
    expected = [{"label": "Person", "score_percent": 98.0, "box": [1, 2, 80, 100]}]
    monkeypatch.setattr(vision, "detect_people", lambda *args: expected)
    def fail_classification(*args):
        pytest.fail("A person was detected; object classification should not run.")
    monkeypatch.setattr(vision, "predict_scores", fail_classification)
    result = vision.scan_image(
        Image.new("RGB", (100, 120)), detector=SimpleNamespace(name="person detector"),
        photo_classifier_loader=fail_classification,
    )
    assert result["kind"] == "people"
    assert result["people"] == expected
