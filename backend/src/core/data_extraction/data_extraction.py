# backend/src/core/data_extraction/data_extraction.py
import os
import cv2
import sys
import mediapipe as mp
import pandas as pd
import logging
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Importar la nueva clase PoseLandmarker desde el mismo directorio
from src.core.data_extraction.pose_landmarker import PoseLandmarker

# Asegúrate de que el path a las configuraciones es correcto
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from src.config.config_manager import load_exercise_config, get_landmark_mapping

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Singleton global para el modelo
_pose_landmarker = None


def get_pose_landmarker(model_path=None):
    """Obtiene o crea la instancia singleton del PoseLandmarker."""
    global _pose_landmarker

    if _pose_landmarker is None and model_path is not None:
        _pose_landmarker = PoseLandmarker(model_path)
    elif (
        _pose_landmarker is not None
        and model_path is not None
        and _pose_landmarker.model_path != model_path
    ):
        # Si se solicita un modelo diferente al actual, recargar
        _pose_landmarker.load_model(model_path)

    return _pose_landmarker


def extract_landmarks_from_video(
    video_path,
    output_csv_path,
    exercise,
    config_path="config.json",
    smoothing_window=5,
    min_brightness=5,
    model_path=None,
    force_model_reload=True,  # Añadido para forzar recarga
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
        force_model_reload (bool): Si True, fuerza la recarga del modelo.

    Returns:
        DataFrame: The processed landmarks data.
    """
    global _pose_landmarker

    logger.info(
        f"Extracting landmarks for exercise '{exercise}' from video: {video_path}"
    )

    # Forzar recarga del modelo si se solicita
    if force_model_reload and _pose_landmarker:
        logger.info("Forzando recarga del modelo según lo solicitado")
        _pose_landmarker.release_resources()
        _pose_landmarker = None

    # Configuration and initialization
    selected_landmarks = get_exercise_landmarks(exercise, config_path)
    landmark_mapping = get_landmark_mapping()

    # Obtener o inicializar el landmarker
    landmarker = get_pose_landmarker(model_path)
    if landmarker is None:
        raise ValueError(
            "No se pudo inicializar el modelo de pose. Verifique la ruta del modelo."
        )

    # Setup video capture
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    landmarks_list = []
    frame_idx = 0

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
            # Usar la instancia de PoseLandmarker para detectar
            detection_result = landmarker.detect_pose(frame, timestamp_ms)

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


# Helper functions - estas funciones permanecen sin cambios


def get_exercise_landmarks(exercise, config_path):
    """
    Load the landmark configuration for the exercise.
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
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    return cap


def initialize_pose_landmarker(model_path):
    """
    Initialize and configure the MediaPipe pose landmarker.
    COMPATIBILIDAD: Esta función se mantiene por compatibilidad con código existente.
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
    """
    return frame.mean() < min_brightness


def convert_frame_to_timestamp(frame_idx, fps):
    """
    Convert frame index to timestamp in milliseconds.
    """
    return int(frame_idx * 1000 / fps)


def detect_pose(landmarker, frame, timestamp_ms):
    """
    Process the frame with MediaPipe to detect pose landmarks.
    COMPATIBILIDAD: Esta función se mantiene por compatibilidad.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def has_valid_landmarks(detection_result):
    """
    Check if valid pose landmarks were detected.
    """
    return detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0


def extract_landmark_coordinates_for_frame(
    detection_result, frame_idx, selected_landmarks
):
    """
    Extract coordinates for selected landmarks from the detection result.
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
    """
    logger.info(f"Smoothing data with window size {window_size}")
    for col in df.columns[1:]:  # Skip frame column
        df[col] = df[col].rolling(window=window_size, min_periods=1).mean()
    return df


def save_to_csv(df, output_path):
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
