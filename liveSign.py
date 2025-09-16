import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/saved_models/basic_cnn_20250916_212443.h5")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame (mirror effect for natural interaction)
    frame = cv2.flip(frame, 1)

    # Define Region of Interest (ROI) box
    x, y, w, h = 100, 100, 300, 300
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Extract ROI in color (RGB)
    roi = frame[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (64, 64))   # match IMAGE_SIZE in config
    roi_resized = roi_resized.astype("float32") / 255.0
    roi_resized = np.expand_dims(roi_resized, axis=0)  # shape -> (1,64,64,3)

    # Prediction
    pred = model.predict(roi_resized)
    digit = np.argmax(pred)

    # Show prediction on frame
    cv2.putText(frame, f"Prediction: {digit}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display window
    cv2.imshow("Hand Sign Recognition", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
