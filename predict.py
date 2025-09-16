import tensorflow as tf
import numpy as np
import cv2
import os

# Load the saved model
model = tf.keras.models.load_model("models/saved_models/basic_cnn_20250916_212443.h5")

# Folder containing test images
test_folder = "tests"


# Preprocess function (RGB, 64x64)
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # load in color (BGR)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None

    img = cv2.resize(img, (64, 64))  # use 64x64 (matches your model)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 64, 64, 3)
    return img


# Loop over digits 0–9
for i in range(10):
    image_path = os.path.join(test_folder, f"test{i}.png")
    img = preprocess_image(image_path)
    if img is None:
        continue

    # Predict
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    print(f"File: test{i}.png | Predicted Digit: {predicted_class} | Confidence: {confidence:.2f}%")

# good models
# basic_cnn_20250916_155311.h5
# basic_cnn_20250916_144311.h5
# basic_cnn_20250916_212443
# avg models
# models/saved_models/resnet_cnn_20250916_213750.h5
