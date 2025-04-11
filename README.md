# Handwritten Digit Recognition Web App

A simple yet powerful web application for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

This project includes:

- A trained CNN model
- Flask backend
- HTML5 Canvas for digit input
- Frontend rendering using JavaScript

## Features

- Draw digits using mouse or touch on a canvas
- Real-time prediction using a trained CNN model
- Preprocessing pipeline to convert canvas drawings into model-compatible inputs
- Responsive and interactive UI

## Tech Stack

### 🧠 Machine Learning

- TensorFlow (CNN Model trained on MNIST)

### 🌐 Web Framework

- Flask (Python)

### 🎨 Frontend

- HTML/CSS/JavaScript
- Canvas for drawing digits

## Dataset

- [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/)

## Model Architecture (CNN)

```
Conv2D(32) -> MaxPool -> Dropout
Conv2D(64) -> MaxPool -> Dropout
Flatten -> Dense(64) -> Dropout -> Dense(10 - softmax)
```

## Setup Instructions

### 🔧 Local Setup

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/digit-recognizer.git
cd digit-recognizer
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
python app.py
```

4. Open `http://127.0.0.1:5000` in your browser.

### 🛠 Requirements (Python Packages)

- Flask
- numpy
- Pillow
- TensorFlow
- h5py

### 🧪 Model Training

To retrain the model:

```python
python train_model.py
```

Or use the provided `best_mnist_model.h5`.

## Project Structure

```
.
├── app.py                 # Flask app
├── templates/
│   └── index.html         # Frontend with drawing canvas
├── static/
│   └── style.css          # Styling (optional)
├── best_mnist_model.h5    # Trained TensorFlow model
├── train_model.py         # Training script for CNN
├── requirements.txt
└── README.md
```

## Deployment

You can deploy this on Render, Heroku, or any cloud provider.

- Make sure `requirements.txt` and `app.py` are present in the root directory.
- Place your `.h5` model file in the same directory as `app.py`.

## Future Improvements

- Convert model to TFLite for lightweight deployment
- Add dark mode and drawing enhancements
- Add ability to clear canvas
- Improve prediction confidence display

## License

This project is open-source and free to use under the MIT License.

---

Happy Coding! ✨

> For suggestions, contributions, or feedback, feel free to raise an issue or pull request on GitHub.

