"""Model loading, image preprocessing and prediction for digit recognition."""
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Load the trained model once when module is imported
model = load_model("digit_model.h5")


# Preparing and pre-processing the image
def preprocess_img(img_path):
    """Open an image and convert it into the model's input format.
    Args:
        img_path: a file path or file like object containing the image to be processed.
    Returns:
        A numpy array of shape (1, 224, 224, 3) representing the preprocessed image.
    """
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    """Return the predicted digit (0-9) for the given preprocessed image array."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
