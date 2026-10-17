"""Flask web application for hand-sign digit recognition.

Serves an upload form and returns the digit predicted by the trained
Keras model for an uploaded hand-sign image.
"""
from flask import Flask, render_template, request
from PIL import UnidentifiedImageError
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Process the uploaded image and display the prediction.
    
    Returns:
        Rendered HTML page with the predicted digit or an error message.
    """
    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))

    except(KeyError, UnidentifiedImageError, ValueError, OSError):
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
