"""Tests for the Flask routes in app.py."""
from io import BytesIO
from unittest.mock import patch

from conftest import SAMPLE_IMAGE

ERROR_MESSAGE = b"File cannot be processed."


def upload(client, file_obj, filename):
    """POST a file to the /prediction route and return the response."""
    return client.post(
        "/prediction",
        data={"file": (file_obj, filename)},
        content_type="multipart/form-data",
    )


# Normal cases

def test_home_page_loads(client):
    """GET / returns 200 and renders the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b'name="file"' in response.data
    assert b"Hand Sign Digit Language Detection" in response.data


@patch("app.predict_result", return_value=7)
def test_valid_image_shows_prediction(mock_predict, client, make_image):
    """A valid image is preprocessed and the predicted digit is shown.

    The model is mocked so this test checks the route logic only.
    """
    response = upload(client, make_image(), "digit.png")
    assert response.status_code == 200
    assert b">7</h2>" in response.data
    assert ERROR_MESSAGE not in response.data
    mock_predict.assert_called_once()


def test_real_image_end_to_end(client):
    """A real hand-sign image of a 3 is predicted as 3 by the real model."""
    with open(SAMPLE_IMAGE, "rb") as image_file:
        response = upload(client, image_file, "sign3.jpeg")
    assert response.status_code == 200
    assert b">3</h2>" in response.data


#  Error and edge cases

def test_missing_file_shows_error(client):
    """A POST with no file field shows the error message, not a crash."""
    response = client.post(
        "/prediction", data={}, content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


def test_non_image_file_shows_error(client):
    """A text file renamed to .jpg is rejected with the error message."""
    response = upload(client, BytesIO(b"this is not an image"), "fake.jpg")
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


def test_empty_file_shows_error(client):
    """An empty (0-byte) upload is rejected with the error message."""
    response = upload(client, BytesIO(b""), "empty.png")
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


def test_get_on_prediction_not_allowed(client):
    """The prediction route only accepts POST; GET returns 405."""
    response = client.get("/prediction")
    assert response.status_code == 405
