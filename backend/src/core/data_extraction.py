import os
import cv2
import sys
import mediapipe as mp
import pandas as pd
import logging
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import load_exercise_config, get_landmark_mapping

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_landmarks_from_video(
    video_path,
    output_csv_path,
    exercise,
    config_path="config_expanded.json",
    smoothing_window=5,
    min_brightness=5,
    model_path=None,
):
    """
    Process a video file to extract pose landmarks for a specific exercise
    and save them to a CSV file.

    Parameters:
        video_path (str): Path to the input video file.
        output_csv_path (str): Path to the output CSV file.
        exercise (str): Name of the exercise to extract.
        config_path (str): Path to the expanded configuration file.
        smoothing_window (int): Size of the window for smoothing the data.
        min_brightness (int): Minimum brightness threshold to consider a frame valid.
        model_path (str): Path to the pose landmarker model.

    Returns:
        DataFrame: The processed landmarks data.
    """
    logger.info(
        f"Extracting landmarks for exercise '{exercise}' from video: {video_path}"
    )

    # Configuration and initialization
    selected_landmarks = get_exercise_landmarks(exercise, config_path)
    landmark_mapping = get_landmark_mapping()

    # Setup video capture and pose detector
    cap = open_video(video_path)
    landmarker = initialize_pose_landmarker(model_path)

    landmarks_list = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Statistical counters
    processed_frames = 0
    skipped_frames = 0
    empty_detection_frames = 0

    # Main frame processing loop
    logger.info(f"Processing {total_frames} frames from the video...")

    try:
        # Use tqdm if available to show progress
        from tqdm import tqdm

        frame_iterator = tqdm(range(total_frames), desc="Processing frames")
    except ImportError:
        frame_iterator = range(total_frames)
        logger.info("Processing frames...")

    for _ in frame_iterator:
        if not cap.isOpened():
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Skip low-quality frames
        if is_low_quality_frame(frame, min_brightness):
            frame_idx += 1
            skipped_frames += 1
            continue

        processed_frames += 1

        # Process frame with MediaPipe
        timestamp_ms = convert_frame_to_timestamp(frame_idx, fps)
        try:
            detection_result = detect_pose(landmarker, frame, timestamp_ms)

            # Extract landmarks if detected
            if has_valid_landmarks(detection_result):
                frame_data = extract_landmark_coordinates_for_frame(
                    detection_result, frame_idx, selected_landmarks
                )
                landmarks_list.append(frame_data)
            else:
                empty_detection_frames += 1
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")

        frame_idx += 1

    cap.release()

    # Processing summary
    logger.info(
        f"Processing completed: {processed_frames} frames processed, "
        f"{skipped_frames} frames skipped, "
        f"{empty_detection_frames} frames without detections"
    )

    # Create and process dataframe
    df = create_landmarks_dataframe(
        landmarks_list, selected_landmarks, landmark_mapping
    )
    df = apply_smoothing(df, smoothing_window)
    save_to_csv(df, output_csv_path)

    return df


# Helper functions


def get_exercise_landmarks(exercise, config_path):
    """
    Load the landmark configuration for the exercise.

    Parameters:
        exercise (str): Name of the exercise.
        config_path (str): Path to the configuration file.

    Returns:
        list: Selected landmarks for the exercise.

    Raises:
        ValueError: If no landmarks are configured for the exercise.
    """
    exercise_config = load_exercise_config(exercise, config_path)
    selected_landmarks = exercise_config.get("landmarks", [])
    if not selected_landmarks:
        raise ValueError(f"No landmarks configured for exercise '{exercise}'")
    logger.info(f"Selected landmarks: {selected_landmarks}")
    return selected_landmarks


def open_video(video_path):
    """
    Open the video file and return the capture object.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: The video capture object.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    return cap


def initialize_pose_landmarker(model_path):
    """
    Initialize and configure the MediaPipe pose landmarker.

    Parameters:
        model_path (str): Path to the model file.

    Returns:
        PoseLandmarker: The configured pose landmarker.

    Raises:
        FileNotFoundError: If the model file is not found.
        Exception: If there is an error loading the model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    BaseOptions = mp_python.BaseOptions
    PoseLandmarker = mp_vision.PoseLandmarker
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
            running_mode=VisionRunningMode.VIDEO,
        )
        return PoseLandmarker.create_from_options(options)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def is_low_quality_frame(frame, min_brightness):
    """
    Check if the frame should be skipped due to low quality.

    Parameters:
        frame (numpy.ndarray): The video frame.
        min_brightness (int): Minimum brightness threshold.

    Returns:
        bool: True if the frame should be skipped, False otherwise.
    """
    return frame.mean() < min_brightness


def convert_frame_to_timestamp(frame_idx, fps):
    """
    Convert frame index to timestamp in milliseconds.

    Parameters:
        frame_idx (int): The frame index.
        fps (float): Frames per second of the video.

    Returns:
        int: Timestamp in milliseconds.
    """
    return int(frame_idx * 1000 / fps)


def detect_pose(landmarker, frame, timestamp_ms):
    """
    Process the frame with MediaPipe to detect pose landmarks.

    Parameters:
        landmarker (PoseLandmarker): The pose landmarker object.
        frame (numpy.ndarray): The video frame.
        timestamp_ms (int): Timestamp in milliseconds.

    Returns:
        PoseLandmarkerResult: The detection result.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def has_valid_landmarks(detection_result):
    """
    Check if valid pose landmarks were detected.

    Parameters:
        detection_result (PoseLandmarkerResult): The detection result.

    Returns:
        bool: True if valid landmarks were detected, False otherwise.
    """
    return detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0


def extract_landmark_coordinates_for_frame(
    detection_result, frame_idx, selected_landmarks
):
    """
    Extract coordinates for selected landmarks from the detection result.

    Parameters:
        detection_result (PoseLandmarkerResult): The detection result.
        frame_idx (int): The frame index.
        selected_landmarks (list): The selected landmarks to extract.

    Returns:
        list: Frame data with landmark coordinates [frame_idx, x1, y1, z1, x2, y2, z2, ...].
    """
    frame_data = [frame_idx]

    for i in selected_landmarks:
        try:
            landmark = detection_result.pose_landmarks[0][i]
            frame_data.extend([landmark.x, landmark.y, landmark.z])
        except (IndexError, AttributeError):
            frame_data.extend([float("nan"), float("nan"), float("nan")])

    return frame_data


def create_landmarks_dataframe(landmarks_list, selected_landmarks, landmark_mapping):
    """
    Create a DataFrame from the landmarks list with appropriate column names.

    Parameters:
        landmarks_list (list): List of frame data with landmark coordinates.
        selected_landmarks (list): The selected landmarks.
        landmark_mapping (dict): Mapping from landmark indices to names.

    Returns:
        DataFrame: Structured landmarks data.

    Raises:
        ValueError: If no landmarks were detected in any frame.
    """
    if not landmarks_list:
        raise ValueError("No landmarks detected in any frame of the video")

    columns = ["frame"]
    for i in selected_landmarks:
        name = landmark_mapping.get(i, f"unknown_{i}")
        columns.extend(
            [f"landmark_{name}_x", f"landmark_{name}_y", f"landmark_{name}_z"]
        )

    return pd.DataFrame(landmarks_list, columns=columns)


def apply_smoothing(df, window_size):
    """
    Apply rolling window smoothing to the landmark coordinates.

    Parameters:
        df (DataFrame): The landmarks DataFrame.
        window_size (int): Size of the smoothing window.

    Returns:
        DataFrame: Smoothed landmarks data.
    """
    logger.info(f"Smoothing data with window size {window_size}")
    for col in df.columns[1:]:  # Skip frame column
        df[col] = df[col].rolling(window=window_size, min_periods=1).mean()
    return df


def save_to_csv(df, output_path):
    """
    Save the DataFrame to a CSV file.

    Parameters:
        df (DataFrame): The landmarks DataFrame.
        output_path (str): Path to the output CSV file.
    """
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
