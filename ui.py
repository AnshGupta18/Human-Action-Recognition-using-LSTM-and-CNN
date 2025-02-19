# ui.py

import tkinter as tk
from tkinter import messagebox
import os
from predict import predict_on_video_live  # Ensure this function is defined in predict.py

# Constants
SEQUENCE_LENGTH = 20
# Define paths to your video files. Adjust these if your videos are stored elsewhere.
video1_path = os.path.join("test_videos", "video1.mp4")
video2_path = os.path.join("test_videos", "video2.mp4")

def run_video_prediction(video_path):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        messagebox.showerror("Error", f"Video file not found or empty:\n{video_path}")
        return
    # You can print to the console or update the UI if needed.
    print(f"Starting prediction on: {video_path}")
    predict_on_video_live(video_path, SEQUENCE_LENGTH)

# Create the main GUI window
root = tk.Tk()
root.title("Video Action Recognition")

# Create buttons for each video
btn_video1 = tk.Button(root, text="Run Prediction on Video 1", 
                       command=lambda: run_video_prediction(video1_path))
btn_video1.pack(pady=10)

btn_video2 = tk.Button(root, text="Run Prediction on Video 2", 
                       command=lambda: run_video_prediction(video2_path))
btn_video2.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
