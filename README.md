Handwritten Digit Recognition System

This project implements a Handwritten Digit Recognition System using TensorFlow and Keras. The system trains a neural network model on the MNIST dataset, a widely used benchmark dataset of handwritten digits, to classify digits from 0 to 9. Additionally, it supports digit prediction from external images.

Table of Contents
Introduction
Features
Technologies Used
Dataset
Project Workflow
Installation
Usage
Results
Future Enhancements
Contributing
License

Introduction
The primary goal of this project is to demonstrate the power of neural networks for image classification tasks. By training a model on the MNIST dataset, this system learns to identify digits in images and predicts the correct class with high accuracy.

Features
Data Preprocessing: Normalization of MNIST data for efficient training.
Neural Network: A Sequential model with dense layers for digit classification.
Prediction from Images: Ability to predict digits from external images using OpenCV.
Visualization: Displays predictions with matplotlib for better understanding.

Technologies Used
Python
TensorFlow / Keras
NumPy
OpenCV
Matplotlib

Dataset
The project uses the MNIST dataset, a collection of 60,000 training and 10,000 testing grayscale images of handwritten digits (28x28 pixels). The dataset is available directly in TensorFlow.


Project Workflow
Data Loading: Load and split the MNIST dataset into training and testing sets.
Data Preprocessing: Normalize the pixel values to a range of 0 to 1.
Model Creation:
Flatten layer to convert 2D images into 1D arrays.
Two Dense layers with ReLU activation.
Final Dense layer with softmax activation for classification.
Model Training: Train the model for 3 epochs using sparse categorical cross-entropy loss and Adam optimizer.
Evaluation: Evaluate the model on the test dataset to compute accuracy and loss.
Prediction: Predict handwritten digits from external images using OpenCV for preprocessing.

Installation
Follow these steps to set up the project locally:

Clone the repository:
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition

Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python main.py

Usage
Train the Model: Run the script to train the model on the MNIST dataset.
python main.py
The model will output accuracy and loss after training.

Predict from External Images: Place your images (1.png, 2.png, etc.) in the root directory. The script will preprocess these images and display the predictions.

Results
Model Accuracy on Test Data: High accuracy (~97-98%).
Example Predictions:
Input Image:
Predicted Digit:
The predicted value is: 7

Future Enhancements
Implement a Convolutional Neural Network (CNN) for improved accuracy.
Add support for batch image predictions.
Develop a GUI or web interface for easier usability.
Extend the model to support other image classification datasets.


License
This project is licensed under the MIT License. See the LICENSE file for details.

