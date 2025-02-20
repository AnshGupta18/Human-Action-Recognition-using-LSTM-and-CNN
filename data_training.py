# data_training.py

import os
import tensorflow as tf
import datetime as dt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from data_preparation import create_dataset
from data_modelling import create_LRCN_model

class Training:
    def run(self):
        # Define model parameters
        SEQUENCE_LENGTH = 20
        IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
        NUM_CHANNELS = 3
        CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
        NUM_CLASSES = len(CLASSES_LIST)

        # Define directory to save models
        MODEL_DIR = "saved_models"
        os.makedirs(MODEL_DIR, exist_ok=True)  # Create directory if it doesn't exist

        # Define checkpoint filenames
        LRCN_model_path = os.path.join(MODEL_DIR, "LRCN_best.keras")

        # Load dataset
        features_train, features_test, labels_train, labels_test, _ = create_dataset()

        # Function to create checkpoint callback that deletes previous checkpoints
        def get_checkpoint_callback(model_name):
            checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.keras")
            # Delete previous checkpoint if it exists
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            return ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,  # Save only the best model
                mode="min",
                verbose=1
            )

        # Train or load LRCN Model
        if os.path.exists(LRCN_model_path):
            print("LRCN model already exists. Loading the model...")
            LRCN_model = tf.keras.models.load_model(LRCN_model_path)
        else:
            print("No saved LRCN model found. Training the model...")
            if features_train is not None and features_train.size > 0:
                LRCN_model = create_LRCN_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES)
                plot_model(LRCN_model, to_file=os.path.join(MODEL_DIR, 'LRCN_model_structure.png'), show_shapes=True, show_layer_names=True)
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
                LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
                
                LRCN_model.fit(
                    features_train, labels_train,
                    epochs=70, batch_size=4, shuffle=True, validation_split=0.2,
                    callbacks=[early_stopping, get_checkpoint_callback("LRCN")]
                )
                
                # Load best model after training
                LRCN_model.load_weights(LRCN_model_path)
            else:
                print("No training data available for LRCN model.")
                return

        # Evaluate LRCN Model
        loss, acc = LRCN_model.evaluate(features_test, labels_test)
        print(f"LRCN Test Accuracy: {acc:.4f}")
        