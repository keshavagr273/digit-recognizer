from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import re
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model("best_mnist_model.h5")

def preprocess_image(img_data):
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    decoded = base64.b64decode(img_str)
    img = Image.open(BytesIO(decoded)).convert('L')  # convert to grayscale

    # DO NOT invert the image anymore

    # Resize to 28x28 directly
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Normalize
    img_array = np.array(img) / 255.0

    # Optional: Threshold (binarize)
    img_array = (img_array > 0.5).astype(np.float32)

    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)

    print("Pixel values range:", img_array.min(), "to", img_array.max())
    return img_array






@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img_data = request.get_json()['image']
    img = preprocess_image(img_data)

    # Debugging prints
    print("Pixel values range:", img.min(), "to", img.max())

    prediction = model.predict(img)
    print("Prediction probabilities:", prediction)

    return jsonify({'prediction': int(np.argmax(prediction))})


if __name__ == '__main__':
    app.run(debug=True)
