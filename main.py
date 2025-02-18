import data_preparation
import train
import predict

def main():
    # Visualize dataset
    data_preparation.visualize_dataset()

    # Create dataset
    features_train, features_test, labels_train, labels_test, video_files_paths = data_preparation.create_dataset()
    
    # Run training (assuming you add a function in train.py to run training)
    # For example, if you define a function called run_training in train.py:
    train.run_training()  # <-- You need to define this function in train.py

    print("Starting prediction...")
    # Run predictions (assuming you add a function in predict.py to run predictions)
    predict.run_predictions()  # <-- You need to define this function in predict.py

if __name__ == "__main__":
    main()
