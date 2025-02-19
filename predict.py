# predict.py

import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# Set parameters and paths
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

# Update MODEL_PATH to point to an existing model file.
MODEL_PATH = "LRCN_model___Date_Time_2025_02_17__13_43_01___Loss_0.4177633225917816___Accuracy_0.8606557250022888.h5"
LRCN_model = load_model(MODEL_PATH)

def predict_on_video_live(video_file_path, sequence_length):
    """
    Opens the given video file, processes it frame by frame, and overlays the predicted action.
    The prediction is updated when a sliding window of frames (of length sequence_length) is available.
    The video is displayed in a window and will close if you press 'q'.
    """
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Error opening video file:", video_file_path)
        return

    frames_queue = deque(maxlen=sequence_length)
    predicted_class_name = "Waiting..."
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the current frame
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        # When we have enough frames, run prediction
        if len(frames_queue) == sequence_length:
            predicted_probs = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_probs)
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Overlay the prediction on the original frame (for better display, use the original size)
        cv2.putText(frame, predicted_class_name, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Prediction", frame)

        # Wait briefly; exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Optional: for testing the function directly.
if __name__ == "__main__":
    test_video = os.path.join("test_videos", "video1.mp4")  # Update if needed
    predict_on_video_live(test_video, SEQUENCE_LENGTH)
