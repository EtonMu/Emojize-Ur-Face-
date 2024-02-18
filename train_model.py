#train_model.py
import os
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

# Define paths and labels
train_path = "train"
test_path = "test"
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Initialize data and labels lists
X = []
y = []


# Function to process and resize images
def process_image(image_path, size=(48, 48)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, size)  # Resize for uniformity
    return resized_img.flatten()


# Load and process images
def load_images(folder_path, label):
    images = os.listdir(folder_path)
    for img_filename in tqdm(images, desc=f"Processing {label}"):
        img_path = os.path.join(folder_path, img_filename)
        X.append(process_image(img_path))
        y.append(labels.index(label))


# Process both training and testing images
for label in labels:
    load_images(os.path.join(train_path, label), label)
    load_images(os.path.join(test_path, label), label)

# Convert lists to NumPy arrays for machine learning
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine Learning Pipeline: PCA for dimensionality reduction + RandomForest Classifier
pipeline = make_pipeline(
    PCA(n_components=150, whiten=True, random_state=42),
    RandomForestClassifier(max_depth=30, min_samples_leaf=2, n_estimators=200, random_state=42)
)

print("Training the classifier with PCA and RandomForest...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict and evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")

# Save the trained model
joblib.dump(pipeline, 'emotion_detection_model.pkl')
print("Model saved as 'emotion_detection_model.pkl'")
