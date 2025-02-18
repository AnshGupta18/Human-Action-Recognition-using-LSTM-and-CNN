# data_preparation.py

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
DATASET_DIR = "UCF50"
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)

def visualize_dataset():
    if not os.path.exists(DATASET_DIR):
        print("UCF50 dataset directory not found. Please download and extract UCF50 into the working directory.")
        return
    plt.figure(figsize=(20, 20))
    all_classes_names = os.listdir(DATASET_DIR)
    num_samples = min(20, len(all_classes_names))
    random_range = random.sample(range(len(all_classes_names)), num_samples)
    for counter, random_index in enumerate(random_range, 1):
        selected_class = all_classes_names[random_index]
        class_path = os.path.join(DATASET_DIR, selected_class)
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
    one_hot_encoded_labels = to_categorical(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state=seed_constant)
    return features_train, features_test, labels_train, labels_test, video_files_paths

# Optionally, you can add a main section to run these functions directly.
if __name__ == "__main__":
    visualize_dataset()
