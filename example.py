#!/usr/bin/env python3
"""
Complete Human Action Recognition Code using ConvLSTM and LRCN models.
This version assumes that two videos (video1.mp4 and video2.mp4) are already downloaded
and located in the test_videos directory.
"""

import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Import MoviePy's editor
from moviepy import VideoFileClip

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

#####################################################################
# Visualize sample frames from the UCF50 dataset (if available)
#####################################################################

if not os.path.exists('UCF50'):
    print("UCF50 dataset directory not found. Please download and extract UCF50 into the working directory.")
else:
    plt.figure(figsize=(20, 20))
    all_classes_names = os.listdir('UCF50')
    num_samples = min(20, len(all_classes_names))
    random_range = random.sample(range(len(all_classes_names)), num_samples)
    for counter, random_index in enumerate(random_range, 1):
        selected_class = all_classes_names[random_index]
        class_path = os.path.join('UCF50', selected_class)
        if not os.path.isdir(class_path):
            continue
        video_files = os.listdir(class_path)
        if not video_files:
            continue
        selected_video = random.choice(video_files)
        video_path = os.path.join(class_path, selected_video)
        video_reader = cv2.VideoCapture(video_path)
        success, bgr_frame = video_reader.read()
        video_reader.release()
        if not success:
            continue
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        cv2.putText(rgb_frame, selected_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        plt.subplot(5, 4, counter)
        plt.imshow(rgb_frame)
        plt.axis('off')
    plt.show()

#####################################################################
# Dataset and Frame Extraction Settings
#####################################################################

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
DATASET_DIR = "UCF50"
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

def frames_extraction(video_path):
    """Extract SEQUENCE_LENGTH frames from a video file, resizing and normalizing each frame."""
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list

def create_dataset():
    """Create dataset by extracting frames from videos in the UCF50 directory for specified classes."""
    features = []
    labels = []
    video_files_paths = []
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data for Class: {class_name}')
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} not found. Skipping.")
            continue
        files_list = os.listdir(class_dir)
        for file_name in files_list:
            video_path = os.path.join(class_dir, file_name)
            frames = frames_extraction(video_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_path)
    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels, video_files_paths

if os.path.exists(DATASET_DIR):
    features, labels, video_files_paths = create_dataset()
else:
    print("UCF50 dataset not found; skipping dataset creation.")
    features, labels, video_files_paths = np.array([]), np.array([]), []

if features.size > 0:
    one_hot_encoded_labels = to_categorical(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state=seed_constant)
else:
    features_train = features_test = labels_train = labels_test = None

#####################################################################
# Model Building: ConvLSTM Model
#####################################################################

def create_convlstm_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True,
                         input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(len(CLASSES_LIST), activation="softmax"))
    model.summary()
    return model

if features_train is not None:
    convlstm_model = create_convlstm_model()
    print("ConvLSTM Model Created Successfully!")
    plot_model(convlstm_model, to_file='convlstm_model_structure_plot.png', show_shapes=True, show_layer_names=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    convlstm_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    convlstm_model_training_history = convlstm_model.fit(
        x=features_train, y=labels_train, epochs=50, batch_size=4, shuffle=True, validation_split=0.2,
        callbacks=[early_stopping_callback])
    model_evaluation_history = convlstm_model.evaluate(features_test, labels_test)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_string = dt.datetime.strftime(dt.datetime.now(), date_time_format)
    # Save using the new native Keras format (.keras)
    model_file_name = f'convlstm_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.keras'
    convlstm_model.save(model_file_name)
else:
    print("No training data available for ConvLSTM model.")

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_title):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    epochs = range(len(metric_value_1))
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)
    plt.title(plot_title)
    plt.legend()
    plt.show()

if features_train is not None:
    plot_metric(convlstm_model_training_history, 'loss', 'val_loss', 'Total Loss vs Validation Loss (ConvLSTM)')
    plot_metric(convlstm_model_training_history, 'accuracy', 'val_accuracy', 'Accuracy vs Validation Accuracy (ConvLSTM)')

#####################################################################
# Model Building: LRCN Model
#####################################################################

def create_LRCN_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'),
                              input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))
    model.summary()
    return model

if features_train is not None:
    LRCN_model = create_LRCN_model()
    print("LRCN Model Created Successfully!")
    plot_model(LRCN_model, to_file='LRCN_model_structure_plot.png', show_shapes=True, show_layer_names=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    LRCN_model_training_history = LRCN_model.fit(
        x=features_train, y=labels_train, epochs=70, batch_size=4, shuffle=True, validation_split=0.2,
        callbacks=[early_stopping_callback])
    model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    current_date_time_string = dt.datetime.strftime(dt.datetime.now(), date_time_format)
    model_file_name = f'LRCN_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.keras'
    LRCN_model.save(model_file_name)
    plot_metric(LRCN_model_training_history, 'loss', 'val_loss', 'Total Loss vs Validation Loss (LRCN)')
    plot_metric(LRCN_model_training_history, 'accuracy', 'val_accuracy', 'Accuracy vs Validation Accuracy (LRCN)')
else:
    print("No training data available for LRCN model.")

#####################################################################
# Video Prediction Functions (Using Pre-Downloaded Videos)
#####################################################################

# Ensure test_videos directory exists
test_videos_directory = 'test_videos'
os.makedirs(test_videos_directory, exist_ok=True)

# List of pre-downloaded video filenames (video1.mp4 and video2.mp4)
video_filenames = ["video1.mp4", "video2.mp4"]

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    """
    Perform action recognition on a video using the LRCN model and write the output video.
    """
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_probs = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_probs)
            predicted_class_name = CLASSES_LIST[predicted_label]
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(frame)
    video_reader.release()
    video_writer.release()

# Loop over the two video files for prediction
for video_file in video_filenames:
    input_video_file_path = os.path.join(test_videos_directory, video_file)
    output_video_file_path = os.path.join(test_videos_directory, f'{os.path.splitext(video_file)[0]}-Output-SeqLen{SEQUENCE_LENGTH}.mp4')
    if os.path.exists(input_video_file_path) and os.path.getsize(input_video_file_path) > 0:
        print(f"Processing {video_file} ...")
        predict_on_video(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)
        try:
            clip = VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None))
            clip.ipython_display()
        except Exception as e:
            print(f"Error displaying output video for {video_file}:", e)
    else:
        print(f"Skipping {video_file} because the file is missing or empty.")

def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    """
    Predict a single action from a video using the LRCN model.
    """
    video_reader = cv2.VideoCapture(video_file_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    frames_list = []
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    predicted_probs = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_probs)
    predicted_class_name = CLASSES_LIST[predicted_label]
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_probs[predicted_label]}')
    video_reader.release()

# Optionally, run single-action prediction on each video
for video_file in video_filenames:
    input_video_file_path = os.path.join(test_videos_directory, video_file)
    if os.path.exists(input_video_file_path) and os.path.getsize(input_video_file_path) > 0:
        print(f"Single-action prediction for {video_file}:")
        predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
    else:
        print(f"Skipping single-action prediction for {video_file} because the file is missing or empty.")
