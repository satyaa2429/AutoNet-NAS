"""Regressions for label alignment, phone photos, and held-out splits."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.continue_training import fixed_split
from src.inference import CLASSES, ROOT, load_cifar_classifier, predict_scores, prepare_image


def test_phone_orientation_is_applied():
    image = Image.new("RGB", (4, 2), "black")
    image.putpixel((0, 0), (255, 0, 0))
    image.getexif()[274] = 6  # EXIF: display with a 90-degree clockwise rotation.
    prepared = prepare_image(image)
    assert prepared.size == (2, 4)
    assert prepared.getpixel((1, 0)) == (255, 0, 0)


def test_transparent_background_is_composited():
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 0))
    assert prepare_image(image).getpixel((0, 0)) == (255, 255, 255)


def test_wrong_class_order_is_rejected(tmp_path):
    path = tmp_path / "wrong_labels.pth"
    torch.save({"classes": list(reversed(CLASSES)), "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="label order"):
        load_cifar_classifier(path)


def test_mismatched_normalization_is_rejected(tmp_path):
    path = tmp_path / "wrong_normalization.pth"
    torch.save({"mean": [0.0] * 3, "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="preprocessing"):
        load_cifar_classifier(path)


def test_continuing_training_preserves_validation_membership():
    train, validation = fixed_split({"validation_indices": [1, 4, 8]}, size=12)
    assert validation == [1, 4, 8]
    assert set(train) == {0, 2, 3, 5, 6, 7, 9, 10, 11}
    assert set(train).isdisjoint(validation)


@pytest.mark.parametrize("name", ["best_nas_model.pth", "improved_nas_model.pth", "advanced_cnn.pth"])
def test_real_saved_models_predict_without_training_side_effects(name):
    torch.set_num_threads(2)
    classifier = load_cifar_classifier(ROOT / "models" / name)
    before = {key: value.clone() for key, value in classifier.model.state_dict().items()}
    scores = predict_scores(Image.new("RGB", (64, 48), "white"), classifier)
    assert scores.shape == (10,)
    assert torch.isfinite(scores).all()
    assert float(scores.sum()) == pytest.approx(1.0, abs=1e-6)
    for key, value in classifier.model.state_dict().items():
        assert torch.equal(value, before[key]), key
