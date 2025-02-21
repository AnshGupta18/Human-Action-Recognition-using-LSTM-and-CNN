# main.py

from data_preparation import DataPreparation
from data_training import Training
from data_prediction import Prediction

class ActionRecognition:
    def __init__(self):
        self.data_prep = DataPreparation()
        self.model_path = "saved_models/LRCN_best.keras"
        self.predictor = Prediction(self.model_path)

    def run(self):
        print("Visualizing Dataset...")
        self.data_prep.visualize_dataset()

        print("Creating Dataset...")
        features_train, features_test, labels_train, labels_test = self.data_prep.create_dataset()

        print("Training Model...")
        training = Training()
        training.run(features_train, labels_train, features_test, labels_test)

        print("Running Predictions...")
        test_video = "test_videos/sample_video.mp4"
        self.predictor.predict_on_video(test_video)

if __name__ == "__main__":
    main = ActionRecognition()
    main.run()
