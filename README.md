# Handwritten Digit Recognition System

This project implements a Handwritten Digit Recognition System using TensorFlow and Keras. <br> The system trains a neural network model on the MNIST dataset, a widely used benchmark dataset of handwritten digits, to classify digits from 0 to 9. Additionally, it supports digit prediction from external images.

# Table of Contents
## Introduction <br>
## Features <br>
## Technologies Used <br>
## Dataset <br>
## Project Workflow <br>
## Installation <br>
## Usage <br>
## Results <br>
## Future Enhancements <br>
## Contributing <br>
## License <br>

# Introduction
The primary goal of this project is to demonstrate the power of neural networks for image classification tasks. <br>
By training a model on the MNIST dataset, this system learns to identify digits in images and predicts the correct class with high accuracy.

# Features
Data Preprocessing: Normalization of MNIST data for efficient training. <br>
Neural Network: A Sequential model with dense layers for digit classification. <br>
Prediction from Images: Ability to predict digits from external images using OpenCV. <br>
Visualization: Displays predictions with matplotlib for better understanding. <br>

# Technologies Used
### Python <br>
### TensorFlow / Keras <br>
### NumPy <br>
### OpenCV <br>
### Matplotlib <br>

# Dataset
The project uses the MNIST dataset, a collection of 60,000 training and 10,000 testing grayscale images of handwritten digits (28x28 pixels).<br>
The dataset is available directly in TensorFlow.

# Project Workflow
Data Loading: Load and split the MNIST dataset into training and testing sets. <br>
Data Preprocessing: Normalize the pixel values to a range of 0 to 1. <br>

# Model Creation:
Flatten layer to convert 2D images into 1D arrays. <br>
Two Dense layers with ReLU activation. <br>
Final Dense layer with softmax activation for classification. <br>
Model Training: Train the model for 3 epochs using sparse categorical cross-entropy loss and Adam optimizer. <br>
Evaluation: Evaluate the model on the test dataset to compute accuracy and loss. <br>
Prediction: Predict handwritten digits from external images using OpenCV for preprocessing. <br>

# Installation
Follow these steps to set up the project locally:

# Clone the repository:
git clone https://github.com/your-username/handwritten-digit-recognition.git <br>
cd handwritten-digit-recognition <br>

### Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate <br>
# On Windows: <br>
venv\Scripts\activate <br>

### pip install -r requirements.txt <br>

### python main.py

# Usage
Train the Model: Run the script to train the model on the MNIST dataset.<br>
## python main.py
The model will output accuracy and loss after training. <br>

Predict from External Images: Place your images (1.png, 2.png, etc.) in the root directory. The script will preprocess these images and display the predictions.

# Results
Model Accuracy on Test Data: High accuracy (~97-98%). <br>
Example Predictions: <br>
Input Image: <br>
Predicted Digit: <br>
The predicted value is: 7 <br>

# Future Enhancements
Implement a Convolutional Neural Network (CNN) for improved accuracy. <br>
Add support for batch image predictions. <br>
Develop a GUI or web interface for easier usability. <br>
Extend the model to support other image classification datasets. <br>


# License
## This project is licensed under the MIT License. See the LICENSE file for details.

