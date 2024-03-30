import sys

import cv2
import numpy as np
import tensorflow as tf

# PARAMS
emotion_labels = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sad', 6: 'surprise'}
if len(sys.argv) == 2:
    MODEL_PATH = sys.argv[1]
else:
    MODEL_PATH = "../trained-models/fer_best.keras"

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# initialize webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the detected face
        face_roi = frame_gray[y:y + h, x:x + w]

        # Resize the face ROI to match input size of the model (e.g., 48x48 for grayscale)
        face_resized = cv2.resize(face_roi, (48, 48))
        face_resized = np.expand_dims(face_resized, axis=-1)  # Add batch dimension
        face_resized = np.expand_dims(face_resized, axis=0)  # Add channel dimension

        # Run inference on the face ROI
        predictions = model.predict(face_resized)
        predicted_class = np.argmax(predictions)

        # Display the emotion prediction on the frame
        emotion_label = emotion_labels[predicted_class]
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the detected face

    # Display the frame with face detection and emotion prediction
    cv2.imshow('Live Emotion Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
