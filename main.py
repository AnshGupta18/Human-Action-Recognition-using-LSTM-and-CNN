import data_preparation
from data_training import Training
import data_prediction

obj=Training
def main():
    # Visualize dataset
    data_preparation.visualize_dataset()

    # Create dataset
    features_train, features_test, labels_train, labels_test, video_files_paths = data_preparation.create_dataset()
    
    # Run training (assuming you add a function in train.py to run training)
    # Create an instance of Training class
    obj = Training()  # Instantiate the class
    # For example, if you define a function called run_training in train.py:
    obj.run()  # <-- You need to define this function in train.py

    print("Starting prediction...")
    # Run predictions (assuming you add a function in predict.py to run predictions)
    data_prediction.run_predictions()  # <-- You need to define this function in predict.py

if __name__ == "__main__":
    main()
