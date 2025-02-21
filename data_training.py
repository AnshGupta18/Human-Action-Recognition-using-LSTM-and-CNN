# data_training.py

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from data_modelling import ModelBuilder
from data_preparation import DataPreparation
from data_modelling import ModelBuilder

class Training:
    def run(self, features_train, labels_train, features_test, labels_test):
         # Create an instance of DataPreparation
        data_prep = DataPreparation()
        
        # Call prepare_data() method
        features_train, features_test, labels_train, labels_test = data_prep.create_dataset()
        SEQUENCE_LENGTH = 20
        IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
        NUM_CHANNELS = 3
        CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
        NUM_CLASSES = len(CLASSES_LIST)
        
        # Define directory to save models
        MODEL_DIR = "saved_models"
        os.makedirs(MODEL_DIR, exist_ok=True)  # Create directory if it doesn't exist

        # Define checkpoint filename
        LRCN_model_path = os.path.join(MODEL_DIR, "LRCN_best.keras")

        # Always delete previous checkpoint if it exists
        if os.path.exists(LRCN_model_path):
            print("Deleting previous model checkpoint...")
            os.remove(LRCN_model_path)

        # Load dataset
        features_train, features_test, labels_train, labels_test = data_prep.create_dataset()

        # Function to create checkpoint callback that also deletes any previous checkpoint file
        def get_checkpoint_callback(model_name):
            checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.keras")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            return ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,  # Save only the best model
            mode="min",
            verbose=1
        )

        print("Training a new LRCN model...")
        if features_train is not None and features_train.size > 0:
        
            # Create a new model
            LRCN_model = ModelBuilder.create_LRCN_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES)
            plot_model(LRCN_model, to_file=os.path.join(MODEL_DIR, 'LRCN_model_structure.png'),show_shapes=True, show_layer_names=True)

        
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
            LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
        
            # Train the model and use the checkpoint callback
            LRCN_model.fit(
            features_train, labels_train,
            epochs=70, batch_size=4, shuffle=True, validation_split=0.2,
            callbacks=[early_stopping, get_checkpoint_callback("LRCN")]
        )
        
        # Load best model after training
        LRCN_model.load_weights(LRCN_model_path)
