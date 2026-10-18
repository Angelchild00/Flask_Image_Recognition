"""Shared pytest fixtures for the Flask image recognition tests."""
from io import BytesIO

import pytest
from PIL import Image

from app import app as flask_app

SAMPLE_IMAGE = "test_images/3/Sign 3 (30).jpeg"


@pytest.fixture(name="client")
def fixture_client():
    """Provide a Flask test client with testing mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture(name="make_image")
def fixture_make_image():
    """Return a helper that builds an in-memory image file.

    The helper takes a Pillow colour mode (e.g. "RGB", "RGBA", "L"),
    an image format and a size, and returns a BytesIO positioned at 0.
    """
    def _make_image(mode="RGB", fmt="PNG", size=(64, 64)):
        buffer = BytesIO()
        Image.new(mode, size).save(buffer, format=fmt)
        buffer.seek(0)
        return buffer

    return _make_image
