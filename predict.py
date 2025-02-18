# predict.py

import os
import cv2
import numpy as np
from collections import deque
from moviepy import VideoFileClip
from tensorflow.keras.models import load_model

# Set parameters and paths
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
test_videos_directory = 'test_videos'
os.makedirs(test_videos_directory, exist_ok=True)
video_filenames = ["video1.mp4", "video2.mp4"]

# Load your saved model (update the file name as needed)
LRCN_model = load_model("LRCN_model_2025_02_18__12_00_00_loss_0.5_acc_0.85.keras")  # change this filename

def predict_on_video(video_file_path, output_file_path, sequence_length):
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (original_video_width, original_video_height))
    frames_queue = deque(maxlen=sequence_length)
    predicted_class_name = ''
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)
        if len(frames_queue) == sequence_length:
            predicted_probs = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_probs)
            predicted_class_name = CLASSES_LIST[predicted_label]
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(frame)
    video_reader.release()
    video_writer.release()

def predict_single_action(video_file_path, sequence_length):
    video_reader = cv2.VideoCapture(video_file_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)
    frames_list = []
    for frame_counter in range(sequence_length):
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

# Process videos for prediction
for video_file in video_filenames:
    input_video_file_path = os.path.join(test_videos_directory, video_file)
    output_video_file_path = os.path.join(test_videos_directory, f'{os.path.splitext(video_file)[0]}-Output-SeqLen{SEQUENCE_LENGTH}.mp4')
    if os.path.exists(input_video_file_path) and os.path.getsize(input_video_file_path) > 0:
        print(f"Processing {video_file} ...")
        predict_on_video(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)
        try:
            clip = VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None))
            clip.ipython_display()  # This may not work in all environments
        except Exception as e:
            print(f"Error displaying output video for {video_file}:", e)
    else:
        print(f"Skipping {video_file} because the file is missing or empty.")

# Single-action prediction for each video
for video_file in video_filenames:
    input_video_file_path = os.path.join(test_videos_directory, video_file)
    if os.path.exists(input_video_file_path) and os.path.getsize(input_video_file_path) > 0:
        print(f"Single-action prediction for {video_file}:")
        predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
    else:
        print(f"Skipping single-action prediction for {video_file} because the file is missing or empty.")
