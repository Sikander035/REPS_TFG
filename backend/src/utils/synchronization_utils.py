"""
Utilities for synchronizing data between user and expert exercise recordings.

This module provides utility functions for temporal alignment and synchronization
of exercise data captured from users and experts, allowing for accurate
comparison regardless of execution speed.

These utilities are used by the main synchronize_data function but can also
be used independently for more granular control over the synchronization process.

Main utility functions:
- validate_input_data: Validates input DataFrames
- preprocess_dataframe: Applies smoothing and fills NaN values
- get_matched_repetitions: Detects and matches repetitions between user and expert
- interpolate_segment: Resamples a data segment to a target length
- divide_segment_by_height: Divides segments based on landmark height
- process_repetition_pair: Synchronizes a pair of exercise repetitions
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging
import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add paths for imports from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import detect_repetitions - this is the module we should use
try:
    from src.core.data_segmentation.detect_repetitions import detect_repetitions
except ImportError:
    logger.warning("Could not import detect_repetitions, some functions may be limited")


#############################################################################
# VALIDATION AND PREPROCESSING FUNCTIONS
#############################################################################


def validate_input_data(
    user_data: pd.DataFrame, expert_data: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """
    Validates that input DataFrames have the correct structure and required landmarks.

    Args:
        user_data: DataFrame with user data
        expert_data: DataFrame with expert data
        config: Configuration dictionary

    Raises:
        ValueError: If there are issues with the input data
    """
    # Check DataFrames are not empty
    if user_data.empty or expert_data.empty:
        raise ValueError("User or expert DataFrames are empty")

    # Check required landmarks are present
    required_landmarks = config.get(
        "landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
    )

    for landmark in required_landmarks:
        for suffix in ["_x", "_y", "_z"]:
            column = f"{landmark}{suffix}"
            if column not in user_data.columns:
                raise ValueError(f"Column {column} not found in user data")
            if column not in expert_data.columns:
                raise ValueError(f"Column {column} not found in expert data")

    # Check for too many NaN values in critical columns
    for landmark in required_landmarks:
        for suffix in ["_x", "_y", "_z"]:
            column = f"{landmark}{suffix}"
            user_nans = user_data[column].isna().mean()
            expert_nans = expert_data[column].isna().mean()

            if user_nans > 0.3:  # More than 30% NaN values
                logger.warning(
                    f"Column {column} in user data has {user_nans:.1%} NaN values"
                )
            if expert_nans > 0.3:
                logger.warning(
                    f"Column {column} in expert data has {expert_nans:.1%} NaN values"
                )


def preprocess_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesses a DataFrame by applying smoothing filters and filling NaN values.

    Args:
        df: DataFrame to preprocess
        config: Configuration with preprocessing parameters

    Returns:
        Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    result = df.copy()

    # Get preprocessing parameters
    window_size = config.get("smoothing_window", 7)
    poly_order = config.get("poly_order", 2)

    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Apply Savitzky-Golay smoothing to specific columns
    landmark_columns = [
        col
        for col in df.columns
        if col.startswith("landmark_") and col.endswith(("_x", "_y", "_z"))
    ]

    for col in landmark_columns:
        # Fill NaN values with interpolation before smoothing
        result[col] = result[col].interpolate(method="linear", limit_direction="both")

        # Apply smoothing only if there are enough points
        if len(result) > window_size:
            result[col] = savgol_filter(result[col], window_size, poly_order)

    return result


#############################################################################
# REPETITION DETECTION AND MATCHING FUNCTIONS
#############################################################################


def get_repetitions(
    data: pd.DataFrame, config: Dict[str, Any], is_user: bool = True
) -> List[Dict[str, int]]:
    """
    Detects repetitions in a single dataset.

    Args:
        data: DataFrame with movement data
        config: Configuration for repetition detection
        is_user: Flag to indicate if this is user data (for logging)

    Returns:
        List of dictionaries with repetition info (start_frame, mid_frame, end_frame)

    Raises:
        ValueError: If no repetitions are detected
    """
    # Extract parameters for repetition detection
    prominence = config.get("rep_prominence", 0.2)
    smoothing_window = config.get("rep_smoothing_window", 11)
    polyorder = config.get("rep_polyorder", 2)
    positive_distance = config.get("rep_positive_distance", 20)
    negative_distance = config.get("rep_negative_distance", 50)
    peak_height_threshold = config.get("rep_peak_height_threshold", -0.8)

    # Set the entity name for logging
    entity = "user" if is_user else "expert"

    # Detect repetitions using detect_repetitions module
    logger.info(f"Detecting repetitions in {entity} data...")
    try:
        repetitions = detect_repetitions(
            data,
            prominence=prominence,
            smoothing_window=smoothing_window,
            polyorder=polyorder,
            positive_distance=positive_distance,
            negative_distance=negative_distance,
            peak_height_threshold=peak_height_threshold,
            plot_graph=False,
            config=config,
        )
    except Exception as e:
        logger.error(f"Error in repetition detection for {entity}: {e}")
        raise

    if not repetitions:
        raise ValueError(f"No repetitions detected in {entity} data")

    logger.info(f"Detected {len(repetitions)} repetitions in {entity} data")

    return repetitions


def match_repetitions(
    user_repetitions: List[Dict[str, int]],
    expert_repetitions: List[Dict[str, int]],
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    config: Dict[str, Any],
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Matches repetitions between user and expert based on the specified strategy.

    Args:
        user_repetitions: List of user repetitions
        expert_repetitions: List of expert repetitions
        user_data: DataFrame with user data (needed for similarity matching)
        expert_data: DataFrame with expert data (needed for similarity matching)
        config: Configuration with matching strategy

    Returns:
        List of tuples (user_repetition, expert_repetition) that are matched
    """
    # Apply matching strategy based on configuration
    matching_strategy = config.get("matching_strategy", "first_only")

    logger.info(f"Matching repetitions using strategy: {matching_strategy}")

    if matching_strategy == "best_example":
        # Use best expert repetition for all user repetitions
        best_expert_rep = find_best_expert_repetition(expert_repetitions, expert_data)
        return [(user_rep, best_expert_rep) for user_rep in user_repetitions]

    elif matching_strategy == "one_to_one":
        # Match repetitions 1:1 until we run out
        max_pairs = min(len(user_repetitions), len(expert_repetitions))
        return [(user_repetitions[i], expert_repetitions[i]) for i in range(max_pairs)]

    elif matching_strategy == "similarity":
        # Match based on similarity in duration/shape
        return match_repetitions_by_similarity(
            user_repetitions, expert_repetitions, user_data, expert_data
        )

    else:  # 'first_only' (default behavior)
        # Use first expert repetition for all user repetitions
        default_expert_rep = expert_repetitions[0]
        logger.info(
            f"Using first expert repetition for all ({len(user_repetitions)}) user repetitions"
        )
        return [(user_rep, default_expert_rep) for user_rep in user_repetitions]


def find_best_expert_repetition(
    expert_repetitions: List[Dict[str, int]], expert_data: pd.DataFrame
) -> Dict[str, int]:
    """
    Finds the "best" expert repetition based on criteria like
    duration, movement smoothness, etc.

    Args:
        expert_repetitions: List of detected expert repetitions
        expert_data: DataFrame with expert data

    Returns:
        The repetition considered as the best example
    """
    # If only one repetition, return it
    if len(expert_repetitions) == 1:
        return expert_repetitions[0]

    # Criteria to evaluate repetition quality
    rep_scores = []

    for rep in expert_repetitions:
        start_frame = rep["start_frame"]
        mid_frame = rep.get("mid_frame")
        end_frame = rep["end_frame"]

        # If no mid_frame, assign low score
        if mid_frame is None or np.isnan(mid_frame):
            rep_scores.append(-1)
            continue

        # Calculate duration
        duration = end_frame - start_frame

        # Check movement smoothness using variance of derivatives
        rep_data = expert_data.iloc[int(start_frame) : int(end_frame)]
        wrist_cols = [
            col for col in rep_data.columns if "wrist" in col and col.endswith("_y")
        ]

        if not wrist_cols:
            rep_scores.append(0)
            continue

        smoothness = 0
        for col in wrist_cols:
            # Calculate first derivative
            deriv = np.diff(rep_data[col].values)
            # Calculate variance (less variance = smoother)
            smoothness -= np.var(deriv)

        # Combine criteria: prefer repetitions of medium duration and smooth movement
        # Ideal duration is around 60-100 frames
        duration_score = (
            -abs(duration - 80) / 20
        )  # Penalize distance from ideal duration

        # Combine criteria
        total_score = duration_score + smoothness
        rep_scores.append(total_score)

    # Return the repetition with highest score
    best_rep_idx = np.argmax(rep_scores)
    return expert_repetitions[best_rep_idx]


def match_repetitions_by_similarity(
    user_repetitions: List[Dict[str, int]],
    expert_repetitions: List[Dict[str, int]],
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Matches repetitions based on similarity in duration and trajectory shape.

    Args:
        user_repetitions: List of user repetitions
        expert_repetitions: List of expert repetitions
        user_data: DataFrame with user data
        expert_data: DataFrame with expert data

    Returns:
        List of matched (user_repetition, expert_repetition) pairs
    """
    # Similarity matrix between each repetition pair
    similarity_matrix = np.zeros((len(user_repetitions), len(expert_repetitions)))

    for i, user_rep in enumerate(user_repetitions):
        u_start = int(user_rep["start_frame"])
        u_end = int(user_rep["end_frame"])
        u_duration = u_end - u_start

        for j, expert_rep in enumerate(expert_repetitions):
            e_start = int(expert_rep["start_frame"])
            e_end = int(expert_rep["end_frame"])
            e_duration = e_end - e_start

            # Relative duration difference
            duration_diff = abs(u_duration - e_duration) / max(u_duration, e_duration)

            # Extract wrist trajectory (time-normalized)
            u_wrist_y = user_data.iloc[u_start:u_end]["landmark_right_wrist_y"].values
            u_wrist_y = u_wrist_y - u_wrist_y.min()  # Normalize to 0
            u_wrist_y = (
                u_wrist_y / u_wrist_y.max() if u_wrist_y.max() > 0 else u_wrist_y
            )  # Normalize to 1

            e_wrist_y = expert_data.iloc[e_start:e_end]["landmark_right_wrist_y"].values
            e_wrist_y = e_wrist_y - e_wrist_y.min()  # Normalize to 0
            e_wrist_y = (
                e_wrist_y / e_wrist_y.max() if e_wrist_y.max() > 0 else e_wrist_y
            )  # Normalize to 1

            # Resample to compare directly
            u_time = np.linspace(0, 1, len(u_wrist_y))
            e_time = np.linspace(0, 1, len(e_wrist_y))

            common_length = 50  # Common length for comparison
            common_time = np.linspace(0, 1, common_length)

            u_interp = interp1d(
                u_time, u_wrist_y, bounds_error=False, fill_value="extrapolate"
            )(common_time)
            e_interp = interp1d(
                e_time, e_wrist_y, bounds_error=False, fill_value="extrapolate"
            )(common_time)

            # Calculate average distance between trajectories
            traj_distance = np.mean(np.abs(u_interp - e_interp))

            # Similarity score (lower is better)
            similarity_score = duration_diff + traj_distance
            similarity_matrix[i, j] = similarity_score

    # Greedily assign repetitions
    matched_pairs = []
    unmatched_user_reps = list(range(len(user_repetitions)))

    while unmatched_user_reps:
        best_score = float("inf")
        best_pair = None

        for i in unmatched_user_reps:
            best_exp_j = np.argmin(similarity_matrix[i, :])
            score = similarity_matrix[i, best_exp_j]

            if score < best_score:
                best_score = score
                best_pair = (i, best_exp_j)

        if best_pair:
            i, j = best_pair
            matched_pairs.append((user_repetitions[i], expert_repetitions[j]))
            unmatched_user_reps.remove(i)

            # Penalize this column to avoid reusing same expert
            similarity_matrix[:, j] += 100
        else:
            break

    return matched_pairs


#############################################################################
# PHASE IDENTIFICATION AND DIVISION FUNCTIONS
#############################################################################


def identify_phases(
    data: pd.DataFrame, repetition: Dict[str, int], config: Dict[str, Any]
) -> List[Tuple[int, int]]:
    """
    Identifies movement phases within a repetition.
    By default, divides into upward and downward phases.

    Args:
        data: DataFrame with movement data
        repetition: Dictionary with repetition info
        config: Configuration for phase identification

    Returns:
        List of tuples (start_frame, end_frame) for each phase
    """
    start_frame = repetition.get("start_frame", 0)
    mid_frame = repetition.get("mid_frame", None)
    end_frame = repetition.get("end_frame", 0)

    # If we have the mid frame, divide into up and down phases
    if mid_frame is not None and not np.isnan(mid_frame):
        # Upward phase: start → mid
        # Downward phase: mid → end
        return [(int(start_frame), int(mid_frame)), (int(mid_frame), int(end_frame))]
    else:
        # If no mid_frame, analyze curve to divide
        phase_strategy = config.get("phase_strategy", "auto")

        if phase_strategy == "auto":
            # Automatic phase detection through curve analysis
            segment = data.iloc[int(start_frame) : int(end_frame)].copy()

            # Use combination of both wrists for robustness
            wrist_y = (
                -segment[["landmark_right_wrist_y", "landmark_left_wrist_y"]]
                .min(axis=1)
                .values
            )

            # Smooth to find more robust peaks/valleys
            wrist_y_smooth = savgol_filter(
                wrist_y,
                min(11, len(wrist_y) - 1 if len(wrist_y) % 2 == 0 else len(wrist_y)),
                2,
            )

            # Find lowest/highest point to divide into phases
            if config.get("exercise_type", "press") == "press":  # Push exercise
                lowest_point_idx = np.argmin(wrist_y_smooth)
                mid_frame_estimated = start_frame + lowest_point_idx
            else:  # Pull exercise
                highest_point_idx = np.argmax(wrist_y_smooth)
                mid_frame_estimated = start_frame + highest_point_idx

            # Check if mid point is within window
            if mid_frame_estimated <= start_frame or mid_frame_estimated >= end_frame:
                # Fallback: divide in half
                mid_frame_estimated = start_frame + (end_frame - start_frame) // 2
                logger.warning(
                    "Could not automatically identify phase. Dividing in half."
                )

            return [
                (int(start_frame), int(mid_frame_estimated)),
                (int(mid_frame_estimated), int(end_frame)),
            ]

        elif phase_strategy == "single":
            # Treat entire repetition as one phase
            return [(int(start_frame), int(end_frame))]

        elif phase_strategy == "thirds":
            # Divide into three equal parts
            third_length = (end_frame - start_frame) / 3
            third1 = start_frame + third_length
            third2 = start_frame + 2 * third_length
            return [
                (int(start_frame), int(third1)),
                (int(third1), int(third2)),
                (int(third2), int(end_frame)),
            ]

        else:
            # Default: divide in two equal parts
            midpoint = start_frame + (end_frame - start_frame) / 2
            return [(int(start_frame), int(midpoint)), (int(midpoint), int(end_frame))]


def divide_segment_by_height(
    data: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    num_divisions: int,
    landmarks: List[str] = None,
    axis: str = "y",
) -> List[int]:
    """
    Divides a segment into subsegments based on the height of specific landmarks.

    Args:
        data: DataFrame with data
        start_frame: Starting frame of segment
        end_frame: Ending frame of segment
        num_divisions: Number of divisions to create
        landmarks: List of landmarks to use (default: wrists)
        axis: Axis to consider ('y' by default)

    Returns:
        List of frames that divide the segment
    """
    if start_frame >= end_frame or start_frame < 0 or end_frame > len(data):
        raise ValueError(
            f"Invalid range: start_frame={start_frame}, end_frame={end_frame}"
        )

    # Data segment between start_frame and end_frame
    segment = data.iloc[start_frame:end_frame].reset_index(drop=True)

    # Default to wrists
    if landmarks is None:
        landmarks = ["landmark_right_wrist", "landmark_left_wrist"]

    # Collect relevant columns
    height_columns = []
    for landmark in landmarks:
        col = f"{landmark}_{axis}"
        if col in segment.columns:
            height_columns.append(col)

    if not height_columns:
        raise ValueError("No height columns found for specified landmarks")

    # For press exercises, invert Y axis
    heights = -segment[height_columns].min(axis=1)

    if heights.empty:
        raise ValueError(f"Empty heights in segment from {start_frame} to {end_frame}.")

    # Calculate division heights
    division_heights = np.linspace(heights.iloc[0], heights.iloc[-1], num_divisions + 1)

    # Identify division frames
    division_frames_relative = [
        (heights - target_height).abs().idxmin()
        for target_height in division_heights[1:-1]
    ]

    division_frames = (
        [start_frame]
        + [start_frame + rel for rel in division_frames_relative]
        + [end_frame]
    )

    # Ensure frames are ordered and unique
    division_frames = sorted(list(set(division_frames)))

    return division_frames


def divide_segment_adaptative(
    data: pd.DataFrame, start_frame: int, end_frame: int, config: Dict[str, Any]
) -> List[int]:
    """
    Divides a segment into subsegments using the specified strategy.

    Args:
        data: DataFrame with data
        start_frame: Starting frame of segment
        end_frame: Ending frame of segment
        config: Configuration dictionary with division parameters

    Returns:
        List of frames that divide the segment
    """
    division_strategy = config.get("division_strategy", "height")
    num_divisions = config.get("num_divisions", 7)

    if division_strategy == "height":
        # Height-based division
        landmarks = config.get(
            "division_landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
        )
        axis = config.get("division_axis", "y")
        return divide_segment_by_height(
            data, start_frame, end_frame, num_divisions, landmarks, axis
        )

    elif division_strategy == "equal":
        # Equal parts division
        step = (end_frame - start_frame) / num_divisions
        return [int(start_frame + i * step) for i in range(num_divisions + 1)]

    elif division_strategy == "acceleration":
        # Division based on acceleration changes
        segment = data.iloc[start_frame:end_frame].copy()
        landmarks = config.get(
            "division_landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
        )
        axis = config.get("division_axis", "y")

        # Calculate second derivative (acceleration)
        accel_columns = []
        for landmark in landmarks:
            col = f"{landmark}_{axis}"
            if col in segment.columns:
                # Calculate acceleration
                vel = np.gradient(segment[col].values)
                accel = np.gradient(vel)
                segment[f"{col}_accel"] = accel
                accel_columns.append(f"{col}_accel")

        if not accel_columns:
            # Fallback to equal division
            return [
                int(start_frame + i * (end_frame - start_frame) / num_divisions)
                for i in range(num_divisions + 1)
            ]

        # Use sum of accelerations
        accel_sum = segment[accel_columns].abs().sum(axis=1)

        # Find points of greatest acceleration change
        sorted_indices = np.argsort(accel_sum.values)[
            ::-1
        ]  # Sort by magnitude of change

        # Take the N-1 most significant points
        significant_points = sorted_indices[: num_divisions - 1]
        significant_points = np.sort(significant_points)  # Reorder by time

        # Convert to absolute frames
        division_frames = (
            [start_frame]
            + [start_frame + idx for idx in significant_points]
            + [end_frame]
        )

        return division_frames

    else:
        # Default: equal division
        step = (end_frame - start_frame) / num_divisions
        return [int(start_frame + i * step) for i in range(num_divisions + 1)]


#############################################################################
# INTERPOLATION AND SYNCHRONIZATION FUNCTIONS
#############################################################################


def interpolate_segment(
    original_data: pd.DataFrame, target_length: int, method: str = "linear"
) -> pd.DataFrame:
    """
    Interpolates a data segment to fit a target length.

    Args:
        original_data: DataFrame with original data
        target_length: Target length for interpolation
        method: Interpolation method ('linear', 'cubic', etc.)

    Returns:
        DataFrame with interpolated data
    """
    if original_data.empty:
        raise ValueError("Empty segment received in interpolate_segment.")

    if target_length <= 1:
        raise ValueError(f"Invalid target length: {target_length}")

    # If already the correct length, return unchanged
    if len(original_data) == target_length:
        return original_data.copy()

    original_frames = np.linspace(0, 1, len(original_data))
    target_frames = np.linspace(0, 1, target_length)

    interpolated_data = pd.DataFrame()

    # Group similar columns for optimization (optional)
    for col in original_data.select_dtypes(include=[np.number]).columns:
        # Skip if all NaN
        if original_data[col].isna().all():
            interpolated_data[col] = [np.nan] * target_length
            continue

        # Non-NaN values for interpolation
        valid_idx = ~original_data[col].isna()
        valid_frames = original_frames[valid_idx]
        valid_values = original_data[col].dropna().values

        # If not enough values
        if len(valid_values) <= 1:
            interpolated_data[col] = [
                valid_values[0] if len(valid_values) > 0 else np.nan
            ] * target_length
            continue

        # Create interpolator with valid values
        try:
            interpolator = interp1d(
                valid_frames,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated_data[col] = interpolator(target_frames)
        except Exception as e:
            logger.warning(f"Error interpolating column {col}: {e}")
            # Fallback to simpler method
            interpolated_data[col] = np.interp(
                target_frames,
                valid_frames,
                valid_values,
                left=valid_values[0],
                right=valid_values[-1],
            )

    return interpolated_data


def synchronize_subsegments(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    user_frames: List[int],
    expert_frames: List[int],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronizes subsegments between user and expert through interpolation.

    Args:
        user_data: DataFrame with user data
        expert_data: DataFrame with expert data
        user_frames: List of user frames delimiting subsegments
        expert_frames: List of expert frames delimiting subsegments
        config: Configuration for synchronization

    Returns:
        Tuple (user_processed, expert_processed) with synchronized segments
    """
    user_segments = []
    expert_segments = []

    # Interpolation method
    interp_method = config.get("interp_method", "linear")

    # Process each pair of subsegments
    for i in range(len(user_frames) - 1):
        user_sub_segment = user_data.iloc[user_frames[i] : user_frames[i + 1]]
        expert_sub_segment = expert_data.iloc[expert_frames[i] : expert_frames[i + 1]]

        if user_sub_segment.empty or expert_sub_segment.empty:
            logger.warning(
                f"Empty segment found: user={user_sub_segment.empty}, "
                f"expert={expert_sub_segment.empty}"
            )
            continue

        # Decide whether to interpolate expert to user or vice versa
        # Default adapts expert to user length
        adapt_direction = config.get("adapt_direction", "expert_to_user")

        if adapt_direction == "expert_to_user":
            # Interpolate expert_sub_segment to match user_sub_segment length
            interpolated_sub_segment = interpolate_segment(
                expert_sub_segment, len(user_sub_segment), interp_method
            )
            user_segments.append(user_sub_segment)
            expert_segments.append(interpolated_sub_segment)

        elif adapt_direction == "user_to_expert":
            # Interpolate user_sub_segment to match expert_sub_segment length
            interpolated_sub_segment = interpolate_segment(
                user_sub_segment, len(expert_sub_segment), interp_method
            )
            user_segments.append(interpolated_sub_segment)
            expert_segments.append(expert_sub_segment)

        elif adapt_direction == "both_to_average":
            # Interpolate both to average length
            avg_length = (len(user_sub_segment) + len(expert_sub_segment)) // 2
            interpolated_user = interpolate_segment(
                user_sub_segment, avg_length, interp_method
            )
            interpolated_expert = interpolate_segment(
                expert_sub_segment, avg_length, interp_method
            )
            user_segments.append(interpolated_user)
            expert_segments.append(interpolated_expert)

        else:
            # Default: adapt expert to user
            interpolated_sub_segment = interpolate_segment(
                expert_sub_segment, len(user_sub_segment), interp_method
            )
            user_segments.append(user_sub_segment)
            expert_segments.append(interpolated_sub_segment)

    # Concatenate segments
    if not user_segments or not expert_segments:
        raise ValueError("No valid segments generated during synchronization")

    user_processed = pd.concat(user_segments).reset_index(drop=True)
    expert_processed = pd.concat(expert_segments).reset_index(drop=True)

    return user_processed, expert_processed


def process_repetition_pair(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    user_rep: Dict[str, int],
    expert_rep: Dict[str, int],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes a pair of repetitions (user-expert) by dividing into phases
    and synchronizing the subsegments.

    Args:
        user_data: DataFrame with user data
        expert_data: DataFrame with expert data
        user_rep: Dictionary with user repetition info
        expert_rep: Dictionary with expert repetition info
        config: Configuration for processing

    Returns:
        Tuple (user_processed, expert_processed) with synchronized repetitions
    """
    # Identify phases for user and expert
    user_phases = identify_phases(user_data, user_rep, config)
    expert_phases = identify_phases(expert_data, expert_rep, config)

    # If different number of phases, use the minimum
    min_phases = min(len(user_phases), len(expert_phases))
    user_phases = user_phases[:min_phases]
    expert_phases = expert_phases[:min_phases]

    # Process each phase separately
    user_processed_segments = []
    expert_processed_segments = []

    for (user_phase_start, user_phase_end), (
        expert_phase_start,
        expert_phase_end,
    ) in zip(user_phases, expert_phases):
        # Divide each phase into subsegments
        user_frames = divide_segment_adaptative(
            user_data, user_phase_start, user_phase_end, config
        )

        expert_frames = divide_segment_adaptative(
            expert_data, expert_phase_start, expert_phase_end, config
        )

        # Ensure both have same number of divisions
        min_divisions = min(len(user_frames), len(expert_frames)) - 1
        user_frames = user_frames[: min_divisions + 1]
        expert_frames = expert_frames[: min_divisions + 1]

        # Synchronize the subsegments
        user_phase_proc, expert_phase_proc = synchronize_subsegments(
            user_data, expert_data, user_frames, expert_frames, config
        )

        user_processed_segments.append(user_phase_proc)
        expert_processed_segments.append(expert_phase_proc)

    # Concatenate all processed phases
    user_processed = pd.concat(user_processed_segments).reset_index(drop=True)
    expert_processed = pd.concat(expert_processed_segments).reset_index(drop=True)

    return user_processed, expert_processed


def process_all_repetitions(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    repetition_pairs: List[Tuple[Dict[str, int], Dict[str, int]]],
    config: Dict[str, Any],
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Processes all paired repetitions in parallel for better performance.

    Args:
        user_data: DataFrame with user data
        expert_data: DataFrame with expert data
        repetition_pairs: List of pairs (user_repetition, expert_repetition)
        config: Configuration for processing

    Returns:
        Tuple (user_processed_segments, expert_processed_segments) with lists of processed segments
    """
    user_processed_segments = []
    expert_processed_segments = []

    # Check if we should use parallelism
    use_parallel = config.get("use_parallel", False) and len(repetition_pairs) > 1

    if use_parallel:
        # Parallel processing for multiple repetitions
        with ProcessPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
            futures = []

            for user_rep, expert_rep in repetition_pairs:
                future = executor.submit(
                    process_repetition_pair,
                    user_data.copy(),  # Copy to avoid issues with parallelism
                    expert_data.copy(),
                    user_rep,
                    expert_rep,
                    config,
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    user_proc, expert_proc = future.result()
                    user_processed_segments.append(user_proc)
                    expert_processed_segments.append(expert_proc)
                except Exception as e:
                    logger.error(f"Error processing repetition in parallel: {e}")

    else:
        # Sequential processing
        for user_rep, expert_rep in repetition_pairs:
            try:
                user_proc, expert_proc = process_repetition_pair(
                    user_data, expert_data, user_rep, expert_rep, config
                )
                user_processed_segments.append(user_proc)
                expert_processed_segments.append(expert_proc)
            except Exception as e:
                logger.error(f"Error processing repetition: {e}")

    return user_processed_segments, expert_processed_segments


def combine_and_validate(
    user_segments: List[pd.DataFrame], expert_segments: List[pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines processed segments and verifies results are valid.

    Args:
        user_segments: List of DataFrames with processed user segments
        expert_segments: List of DataFrames with processed expert segments

    Returns:
        Tuple (final_user_data, final_expert_data) with complete synchronized data
    """
    if not user_segments or not expert_segments:
        raise ValueError("No segments to combine")

    # Check that for each user segment there's an expert segment
    if len(user_segments) != len(expert_segments):
        raise ValueError(
            f"Different number of segments: user={len(user_segments)}, "
            f"expert={len(expert_segments)}"
        )

    # Check that corresponding segments have same size
    for i, (user_seg, expert_seg) in enumerate(zip(user_segments, expert_segments)):
        if len(user_seg) != len(expert_seg):
            raise ValueError(
                f"Segment {i} has different lengths: "
                f"user={len(user_seg)}, expert={len(expert_seg)}"
            )

    # Combine all segments
    final_user_data = pd.concat(user_segments, axis=0).reset_index(drop=True)
    final_expert_data = pd.concat(expert_segments, axis=0).reset_index(drop=True)

    # Check they have same total number of frames
    if len(final_user_data) != len(final_expert_data):
        raise ValueError(
            f"Mismatch in number of frames: user={len(final_user_data)}, "
            f"expert={len(final_expert_data)}"
        )

    # Renumber frames if the column exists
    if "frame" in final_user_data.columns and "frame" in final_expert_data.columns:
        new_frames = np.arange(len(final_user_data))
        final_user_data["frame"] = new_frames
        final_expert_data["frame"] = new_frames

    logger.info(
        f"Synchronization completed: {len(final_user_data)} frames synchronized"
    )

    return final_user_data, final_expert_data
