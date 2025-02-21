# ui.py

import cv2
import numpy as np
from collections import deque
from data_prediction import Prediction

class ActionRecognitionUI:
    def __init__(self):
        self.model_path = "saved_models/LRCN_best.keras"
        self.predictor = Prediction(self.model_path)
        self.sequence_length = self.predictor.sequence_length
        self.frames_queue = deque(maxlen=self.sequence_length)
        self.classes_list = self.predictor.classes_list

    def start_real_time_prediction(self):
        cap = cv2.VideoCapture(0)  # Use webcam

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (64, 64)) / 255.0
            self.frames_queue.append(resized_frame)

            predicted_class_name = "Waiting..."
            if len(self.frames_queue) == self.sequence_length:
                predicted_probs = self.predictor.model.predict(np.expand_dims(self.frames_queue, axis=0))[0]
                predicted_class_name = self.classes_list[np.argmax(predicted_probs)]

            cv2.putText(frame, f"Action: {predicted_class_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Real-Time Action Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ui = ActionRecognitionUI()
    ui.start_real_time_prediction()
