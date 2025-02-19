# train.py

import datetime as dt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from data_preparation import create_dataset
from models import create_convlstm_model, create_LRCN_model

# Load dataset
features_train, features_test, labels_train, labels_test, _ = create_dataset()

# Define model parameters
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
NUM_CHANNELS = 3
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
NUM_CLASSES = len(CLASSES_LIST)
date_time_format = '%Y_%m_%d__%H_%M_%S'

# Train ConvLSTM Model
if features_train is not None and features_train.size > 0:
    convlstm_model = create_convlstm_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES)
    plot_model(convlstm_model, to_file='convlstm_model_structure_plot.png', show_shapes=True, show_layer_names=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    convlstm_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    history_convlstm = convlstm_model.fit(features_train, labels_train, epochs=50, batch_size=4, shuffle=True, validation_split=0.2, callbacks=[early_stopping])
    loss, acc = convlstm_model.evaluate(features_test, labels_test)
    current_time = dt.datetime.strftime(dt.datetime.now(), date_time_format)
    model_filename = f'convlstm_model_{current_time}_loss_{loss}_acc_{acc}.keras'
    convlstm_model.save(model_filename)
    # Optionally, plot training metrics here.
else:
    print("No training data available for ConvLSTM model.")

# Train LRCN Model
if features_train is not None and features_train.size > 0:
    LRCN_model = create_LRCN_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES)
    plot_model(LRCN_model, to_file='LRCN_model_structure_plot.png', show_shapes=True, show_layer_names=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    history_LRCN = LRCN_model.fit(features_train, labels_train, epochs=70, batch_size=4, shuffle=True, validation_split=0.2, callbacks=[early_stopping])
    loss, acc = LRCN_model.evaluate(features_test, labels_test)
    current_time = dt.datetime.strftime(dt.datetime.now(), date_time_format)
    model_filename = 'LRCN_model_latest.keras'
    LRCN_model.save(model_filename)

else:
    print("No training data available for LRCN model.")
