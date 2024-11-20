# Character-Recognition-using-CNN-and-N-Grams
This project is a web application that performs digit and character recognition using a deep learning model trained on the MNIST and EMNIST datasets. It uses Flask as the backend framework and PyTorch for model inference. The frontend is built with HTML and CSS for a simple user interface that allows users to upload images for recognition.

## Features

- **Digit Recognition**: Recognize digits (0-9) from the MNIST dataset.
- **Character Recognition**: Recognize uppercase letters (A-Z) from the EMNIST dataset.
- **Image Upload**: Upload an image for prediction.
- **Real-time Prediction**: The model predicts the class and provides probabilities for each class.

## Tech Stack

- **Backend**: Flask, PyTorch
- **Frontend**: HTML, CSS (TailwindCSS)
- **Model**: Convolutional Neural Network (CNN) for digit and letter recognition.
- **Libraries**:
  - `Flask` for creating the web server and handling routes.
  - `Pillow` for image processing.
  - `Torch` and `Torchvision` for model inference and preprocessing.
  - `NumPy` for numerical computations.
