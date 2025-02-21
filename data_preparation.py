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
SEED_CONSTANT = 27

np.random.seed(SEED_CONSTANT)
random.seed(SEED_CONSTANT)

class DataPreparation:
    def __init__(self):
        self.features = []
        self.labels = []
        self.video_files_paths = []

    def visualize_dataset(self):
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

    def frames_extraction(self, video_path):
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

        # ✅ Ensure frames_list is always a NumPy array before returning
        return np.array(frames_list) if len(frames_list) == SEQUENCE_LENGTH else None

    def create_dataset(self):
        self.features = []  # ✅ Ensure it's a list before appending
        self.labels = []
        self.video_files_paths = []

        for class_index, class_name in enumerate(CLASSES_LIST):
            print(f'Extracting Data for Class: {class_name}')
            class_dir = os.path.join(DATASET_DIR, class_name)

            if not os.path.exists(class_dir):
                print(f"Directory {class_dir} not found. Skipping.")
                continue

            for file_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, file_name)
                frames = self.frames_extraction(video_path)

                if frames is not None:  # ✅ Ensure frames are valid
                    self.features.append(frames)  
                    self.labels.append(class_index)
                    self.video_files_paths.append(video_path)

        self.features = np.array(self.features)  # ✅ Convert only after all appends
        self.labels = np.array(self.labels)
        one_hot_encoded_labels = to_categorical(self.labels)

        return train_test_split(
            self.features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state=SEED_CONSTANT
        )
