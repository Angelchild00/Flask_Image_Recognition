"""Tests for the preprocessing and prediction functions in model.py."""
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import UnidentifiedImageError

from conftest import SAMPLE_IMAGE
from model import preprocess_img, predict_result


# preprocess_img

def test_preprocess_output_shape_and_range():
    """A real JPEG becomes a (1, 224, 224, 3) array scaled to 0-1."""
    result = preprocess_img(SAMPLE_IMAGE)
    assert result.shape == (1, 224, 224, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_accepts_file_stream(make_image):
    """preprocess_img works with a file-like object, as Flask passes."""
    result = preprocess_img(make_image(size=(500, 300)))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image():
    """Bytes that are not an image raise UnidentifiedImageError."""
    with pytest.raises(UnidentifiedImageError):
        preprocess_img(BytesIO(b"not an image"))


def test_preprocess_missing_file_raises():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("test_images/does_not_exist.jpeg")


# predict_result

def test_predict_result_returns_argmax():
    """predict_result returns the index of the highest probability.

    The Keras model is replaced with a mock so the test is fast and
    the expected answer is known.
    """
    fake_probs = np.zeros((1, 10))
    fake_probs[0, 4] = 0.9
    fake_model = MagicMock()
    fake_model.predict.return_value = fake_probs

    with patch("model.model", fake_model):
        assert predict_result(np.zeros((1, 224, 224, 3))) == 4
    fake_model.predict.assert_called_once()


def test_predict_result_real_model_gives_valid_digit():
    """The real model returns an integer class between 0 and 9."""
    prediction = predict_result(preprocess_img(SAMPLE_IMAGE))
    assert isinstance(prediction, (int, np.integer))
    assert 0 <= prediction <= 9

@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_preprocess_handles_non_rgb_images(make_image, mode):
    """Transparent (RGBA) and grayscale (L) images are converted to RGB.

    The model expects exactly 3 colour channels, so other modes must be
    converted before reshaping.
    """
    result = preprocess_img(make_image(mode=mode))
    assert result.shape == (1, 224, 224, 3)
