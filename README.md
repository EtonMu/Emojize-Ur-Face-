# Emojize Ur Face!

## Overview
"Emojize Ur Face!" is an innovative application that uses machine learning to analyze facial expressions in real-time and overlay corresponding emojis on the user's face. This project aims to enrich digital interactions by adding an expressive, fun twist.

## How to Load and Run

### Prerequisites
Ensure you have the following prerequisites installed:
- Python
- numpy
- opencv-python (cv2)
- joblib
- sklearn
- mediapipe

### Steps
1. **Download the Project**: Clone or download the project repository to your local machine.
2. **Obtain the Dataset**: Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and put them into the program directory.
3. **Train the Model**: Run `train_model.py` to train the emotion detection model.
4. **Run the Application**: Execute `emojize.py` to start the application. This script captures video from a webcam, analyzes facial expressions, and overlays emojis on the face in real-time.

## Machine Learning Model and Strategy
The application employs a machine learning pipeline that includes Principal Component Analysis (PCA) for dimensionality reduction and a RandomForestClassifier for emotion detection. The workflow is as follows:

- **Data Preparation**: Images are converted to grayscale, resized, and flattened into vectors.
- **PCA**: Reduces the dimensionality of the image data, improving efficiency.
- **Random Forest Classifier**: An ensemble method used for classifying the emotion based on the processed image data.
- **Model Evaluation**: The model's performance is evaluated using accuracy as the metric.

