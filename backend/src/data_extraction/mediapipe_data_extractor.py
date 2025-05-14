import os
import argparse
import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

body_parts = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


def extract_landmarks_from_video(
    video_path,
    output_csv_path,
    model_path=os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "pose_landmarker_heavy.task"
    ),
):
    """
    Process a video file to extract pose landmarks and save them in a CSV file.

    Parameters:
        video_path (str): Path to the input video file.
        output_csv_path (str): Path to the output CSV file.
        model_path (str): Path to the MediaPipe Pose Landmarker model file.
    """
    # Prepare video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    # List to store landmarks
    landmarks_list = []

    # Initialize PoseLandmarker with video mode
    BaseOptions = mp_python.BaseOptions
    PoseLandmarker = mp_vision.PoseLandmarker
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
        running_mode=VisionRunningMode.VIDEO,
    )

    selected_landmarks = [16, 14, 12, 24, 23, 11, 13, 15]  # Key landmarks

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert frame to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Calculate timestamp in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)

            # Perform pose landmarking
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Extract selected landmarks
            if pose_landmarker_result.pose_landmarks:
                frame_landmarks = []
                for i in selected_landmarks:
                    landmark = pose_landmarker_result.pose_landmarks[0][i]
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks_list.append([frame_idx] + frame_landmarks)
            else:
                # If no landmarks detected, append NaNs
                num_landmarks = len(selected_landmarks)
                frame_landmarks = [float("nan")] * (num_landmarks * 3)
                landmarks_list.append([frame_idx] + frame_landmarks)

            frame_idx += 1

        cap.release()

    # Create DataFrame from landmarks
    columns = ["frame"]
    for i in selected_landmarks:
        columns.extend(
            [
                f"landmark_{body_parts[i]}_x",
                f"landmark_{body_parts[i]}_y",
                f"landmark_{body_parts[i]}_z",
            ]
        )

    df = pd.DataFrame(landmarks_list, columns=columns)
    df.to_csv(output_csv_path, index=False)
