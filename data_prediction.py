# data_prediction.py

import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

class Prediction:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
        self.sequence_length = 20

    def predict_on_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_queue = deque(maxlen=self.sequence_length)
        predicted_class_name = "Waiting..."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (64, 64)) / 255.0
            frames_queue.append(resized_frame)

            if len(frames_queue) == self.sequence_length:
                predicted_probs = self.model.predict(np.expand_dims(frames_queue, axis=0))[0]
                predicted_class_name = self.classes_list[np.argmax(predicted_probs)]

            cv2.putText(frame, predicted_class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Prediction", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
