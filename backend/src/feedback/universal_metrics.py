# backend/src/feedback/universal_metrics.py
import sys
import os
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.analysis_utils import (
    apply_unified_sensitivity,
    calculate_deviation_score,
    apply_sensitivity_to_threshold,
)
from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


def analyze_movement_amplitude_universal(
    user_data, expert_data, exercise_config, landmarks_config, config_path="config.json"
):
    """
    UNIVERSAL: Amplitude analysis extracted from current military press code.
    Only landmarks change, logic is EXACTLY the same.
    """
    # Get exercise_name from config_path (extract from exercise_config if possible)
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "amplitude", exercise_name, config_path
    )

    # Extract landmarks from configuration
    landmark_left = landmarks_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = landmarks_config.get("right_landmark", "landmark_right_elbow")
    axis = landmarks_config.get("axis", "y")
    movement_direction = landmarks_config.get("movement_direction", "up")
    feedback_context = landmarks_config.get("feedback_context", "elbows")

    # EXACT LOGIC FROM CURRENT CODE - calculate average of landmarks
    user_signal = (
        user_data[f"{landmark_left}_{axis}"].values
        + user_data[f"{landmark_right}_{axis}"].values
    ) / 2
    expert_signal = (
        expert_data[f"{landmark_left}_{axis}"].values
        + expert_data[f"{landmark_right}_{axis}"].values
    ) / 2

    # EXACT LOGIC FROM CURRENT CODE - In MediaPipe: Y=0 top, Y=1 bottom
    user_highest_point = np.min(user_signal)
    user_lowest_point = np.max(user_signal)
    expert_highest_point = np.min(expert_signal)
    expert_lowest_point = np.max(expert_signal)

    # EXACT LOGIC FROM CURRENT CODE - Calculate range of motion
    user_rom = user_lowest_point - user_highest_point
    expert_rom = expert_lowest_point - expert_highest_point
    rom_ratio = user_rom / expert_rom if expert_rom > 0 else 0

    # EXACT LOGIC FROM CURRENT CODE - Analyze difference at lowest point
    bottom_diff = (
        abs(user_lowest_point - expert_lowest_point) / expert_rom
        if expert_rom > 0
        else 0
    )

    # EXACT LOGIC FROM CURRENT CODE - Score and sensitivity
    # Get penalty from configuration
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="amplitude",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        rom_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(base_score, sensitivity_factor, "amplitude")

    # Get thresholds using config_manager
    rom_threshold = config_manager.get_analysis_threshold(
        "rom_threshold", exercise_name, config_path
    )
    bottom_diff_threshold = config_manager.get_analysis_threshold(
        "bottom_diff_threshold", exercise_name, config_path
    )

    # ADAPTED FEEDBACK - keep logic but change context
    feedback = {}
    if final_score >= 85:
        feedback["amplitude"] = f"Excellent range of motion in the {feedback_context}."
    elif final_score >= 70:
        if rom_ratio > 1.15:
            if movement_direction == "down":  # Squat
                feedback["amplitude"] = (
                    f"Your range of motion is excessive. Control the descent to avoid "
                    f"hyperflexion of the {feedback_context}."
                )
            else:  # Press, pull-up
                feedback["amplitude"] = (
                    f"Your range of motion is excessive. Control the descent to avoid "
                    f"hyperextension of the {feedback_context}."
                )
        else:
            if movement_direction == "down":
                feedback["amplitude"] = (
                    f"Your range of motion could be wider. Lower the {feedback_context} more."
                )
            else:
                feedback["amplitude"] = (
                    f"Your range of motion could be wider. Lower until the weights "
                    f"are approximately at shoulder height."
                )
    elif final_score >= 50:
        # EXACT LOGIC FROM CURRENT CODE
        rom_threshold_adj = apply_sensitivity_to_threshold(
            rom_threshold, sensitivity_factor
        )
        bottom_diff_threshold_adj = apply_sensitivity_to_threshold(
            bottom_diff_threshold, sensitivity_factor
        )

        if rom_ratio > 1.25:
            feedback["amplitude"] = (
                f"Your range of motion is excessively wide. It's critical "
                f"to control the descent to avoid hyperextension of the {feedback_context}."
            )
        elif bottom_diff > bottom_diff_threshold_adj * 1.5:
            feedback["amplitude"] = (
                f"Your range of motion is insufficient. It's important to lower until "
                f"the {feedback_context} reach the correct position for proper technique."
            )
        else:
            feedback["amplitude"] = (
                f"Your range of motion is limited. Lower the {feedback_context} more for complete flexion "
                f"and extend fully at the top."
            )
    else:
        # Critical cases
        if rom_ratio > 1.25:
            feedback["amplitude"] = (
                f"Your range of motion is excessively wide. It's critical "
                f"to control the descent to avoid hyperextension of the {feedback_context}."
            )
        else:
            feedback["amplitude"] = (
                f"Your range of motion is significantly limited. It's critical "
                f"to work on full amplitude: lower the {feedback_context} more and extend fully at the top."
            )

    # EXACT METRICS FROM CURRENT CODE
    metrics = {
        "user_rom": user_rom,
        "expert_rom": expert_rom,
        "rom_ratio": rom_ratio,
        "bottom_position_difference": bottom_diff,
        "user_highest_point": user_highest_point,
        "user_lowest_point": user_lowest_point,
        "expert_highest_point": expert_highest_point,
        "expert_lowest_point": expert_lowest_point,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_symmetry_universal(
    user_data, expert_data, exercise_config, landmarks_config, config_path="config.json"
):
    """
    UNIVERSAL: Symmetry analysis extracted from current military press code.
    Only landmarks change, logic is EXACTLY the same.
    """
    # Get exercise_name from config_path (extract from exercise_config if possible)
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "symmetry", exercise_name, config_path
    )

    # Extract landmarks from configuration
    landmark_left = landmarks_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = landmarks_config.get("right_landmark", "landmark_right_elbow")
    axis = landmarks_config.get("axis", "y")
    feedback_context = landmarks_config.get("feedback_context", "movement")

    # EXACT LOGIC FROM CURRENT CODE - Extract vertical positions
    user_r_signal = user_data[f"{landmark_right}_{axis}"].values
    user_l_signal = user_data[f"{landmark_left}_{axis}"].values

    # EXACT LOGIC FROM CURRENT CODE - Calculate average difference between sides
    height_diff = np.mean(np.abs(user_r_signal - user_l_signal))

    # EXACT LOGIC FROM CURRENT CODE - Normalize difference relative to average range of motion
    user_range = np.max(user_r_signal) - np.min(user_r_signal)
    normalized_diff = height_diff / user_range if user_range > 0 else 0

    # EXACT LOGIC FROM CURRENT CODE - Compare with expert symmetry
    expert_r_signal = expert_data[f"{landmark_right}_{axis}"].values
    expert_l_signal = expert_data[f"{landmark_left}_{axis}"].values
    expert_height_diff = np.mean(np.abs(expert_r_signal - expert_l_signal))
    expert_range = np.max(expert_r_signal) - np.min(expert_r_signal)
    expert_normalized_diff = (
        expert_height_diff / expert_range if expert_range > 0 else 0
    )

    # EXACT LOGIC FROM CURRENT CODE - User vs expert asymmetry ratio
    asymmetry_ratio = (
        normalized_diff / expert_normalized_diff if expert_normalized_diff > 0 else 1
    )

    # EXACT LOGIC FROM CURRENT CODE - Score and sensitivity
    # Get penalty from configuration
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="symmetry",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        asymmetry_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(base_score, sensitivity_factor, "symmetry")

    # EXACT LOGIC FROM CURRENT CODE - Feedback
    symmetry_threshold = config_manager.get_analysis_threshold(
        "symmetry_threshold", exercise_name, config_path
    )
    symmetry_threshold_adj = apply_sensitivity_to_threshold(
        symmetry_threshold, sensitivity_factor
    )

    feedback = {}
    if asymmetry_ratio > (1.8 / sensitivity_factor):
        if sensitivity_factor > 1.5:
            feedback["symmetry"] = (
                f"There is very notable asymmetry between your right and left side in the {feedback_context}. "
                f"It's a priority to work on balancing both arms."
            )
        else:
            feedback["symmetry"] = (
                f"There is notable asymmetry between your right and left side in the {feedback_context}. "
                f"Focus on lifting both arms equally."
            )
    elif normalized_diff > symmetry_threshold_adj:
        if sensitivity_factor > 1.5 and normalized_diff > symmetry_threshold_adj * 1.5:
            feedback["symmetry"] = (
                f"Significant asymmetry detected in the {feedback_context}. "
                f"It's important to work on keeping both sides at the same height."
            )
        else:
            feedback["symmetry"] = (
                f"Some asymmetry detected in the {feedback_context}. "
                f"Try to keep both sides at the same height."
            )
    else:
        feedback["symmetry"] = (
            f"Excellent bilateral symmetry in the {feedback_context}."
        )

    # EXACT METRICS FROM CURRENT CODE
    metrics = {
        "height_difference": height_diff,
        "normalized_difference": normalized_diff,
        "expert_normalized_difference": expert_normalized_diff,
        "asymmetry_ratio": asymmetry_ratio,
        "user_range_of_motion": user_range,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_movement_trajectory_3d_universal(
    user_data, expert_data, exercise_config, landmarks_config, config_path="config.json"
):
    """
    UNIVERSAL: Trajectory analysis extracted from current military press code.
    Only landmarks change, logic is EXACTLY the same.
    """
    # Get exercise_name from config_path (extract from exercise_config if possible)
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "trajectory", exercise_name, config_path
    )

    # Extract landmarks from configuration
    landmark_left = landmarks_config.get("left_landmark", "landmark_left_wrist")
    landmark_right = landmarks_config.get("right_landmark", "landmark_right_wrist")

    # EXACT LOGIC FROM CURRENT CODE - Use average of both wrists/landmarks for greater stability
    user_x = (
        user_data[f"{landmark_right}_x"].values + user_data[f"{landmark_left}_x"].values
    ) / 2
    user_y = (
        user_data[f"{landmark_right}_y"].values + user_data[f"{landmark_left}_y"].values
    ) / 2
    user_z = (
        user_data[f"{landmark_right}_z"].values + user_data[f"{landmark_left}_z"].values
    ) / 2

    expert_x = (
        expert_data[f"{landmark_right}_x"].values
        + expert_data[f"{landmark_left}_x"].values
    ) / 2
    expert_y = (
        expert_data[f"{landmark_right}_y"].values
        + expert_data[f"{landmark_left}_y"].values
    ) / 2
    expert_z = (
        expert_data[f"{landmark_right}_z"].values
        + expert_data[f"{landmark_left}_z"].values
    ) / 2

    # EXACT LOGIC FROM CURRENT CODE - Deviation analysis
    lateral_deviation_user = np.std(user_x)
    lateral_deviation_expert = np.std(expert_x)
    lateral_deviation_ratio = (
        lateral_deviation_user / lateral_deviation_expert
        if lateral_deviation_expert > 0
        else 1
    )

    frontal_deviation_user = np.std(user_z)
    frontal_deviation_expert = np.std(expert_z)
    frontal_deviation_ratio = (
        frontal_deviation_user / frontal_deviation_expert
        if frontal_deviation_expert > 0
        else 1
    )

    # EXACT LOGIC FROM CURRENT CODE - Score
    worst_deviation_ratio = max(lateral_deviation_ratio, frontal_deviation_ratio)
    # Get penalty from configuration
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="trajectory",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        worst_deviation_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "trajectory"
    )

    # EXACT LOGIC FROM CURRENT CODE - Feedback
    lateral_threshold = config_manager.get_analysis_threshold(
        "lateral_dev_threshold", exercise_name, config_path
    )
    frontal_threshold = config_manager.get_analysis_threshold(
        "frontal_dev_threshold", exercise_name, config_path
    )

    lateral_threshold_adj = apply_sensitivity_to_threshold(
        lateral_threshold, sensitivity_factor
    )
    frontal_threshold_adj = apply_sensitivity_to_threshold(
        frontal_threshold, sensitivity_factor
    )

    feedback = {}

    # EXACT LOGIC FROM CURRENT CODE - Direct differences with expert
    shoulder_width = np.mean(
        np.abs(
            user_data["landmark_right_shoulder_x"].values
            - user_data["landmark_left_shoulder_x"].values
        )
    )
    trajectory_diff_x = np.mean(np.abs(user_x - expert_x))
    trajectory_diff_z = np.mean(np.abs(user_z - expert_z))

    normalized_trajectory_diff_x = (
        trajectory_diff_x / shoulder_width if shoulder_width > 0 else 0
    )
    normalized_trajectory_diff_z = (
        trajectory_diff_z / shoulder_width if shoulder_width > 0 else 0
    )

    # EXACT LOGIC FROM CURRENT CODE - Evaluate lateral
    if lateral_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trajectory_lateral"] = (
            "Your movement deviates excessively in lateral direction. "
            "Focus urgently on keeping wrists in vertical line."
        )
    elif normalized_trajectory_diff_x > lateral_threshold_adj:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_x > lateral_threshold_adj * 1.5
        ):
            feedback["trajectory_lateral"] = (
                "Significant lateral deviation detected in your trajectory. "
                "It's important to correct to maintain a more vertical movement."
            )
        else:
            feedback["trajectory_lateral"] = (
                "Some lateral deviation detected in your trajectory. "
                "Try to maintain a more vertical movement."
            )
    else:
        feedback["trajectory_lateral"] = "Excellent lateral movement control."

    # EXACT LOGIC FROM CURRENT CODE - Evaluate frontal
    if frontal_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trajectory_frontal"] = (
            "Your movement deviates forward/backward significantly. "
            "Keep wrists in a consistent vertical plane."
        )
    elif normalized_trajectory_diff_z > frontal_threshold_adj:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_z > frontal_threshold_adj * 1.5
        ):
            feedback["trajectory_frontal"] = (
                "Significant frontal deviation detected in your movement. "
                "It's important to maintain a more consistent vertical plane."
            )
        else:
            feedback["trajectory_frontal"] = (
                "Some frontal deviation detected in your movement."
            )
    else:
        feedback["trajectory_frontal"] = "Good frontal movement control."

    # EXACT LOGIC FROM CURRENT CODE - General feedback
    if max(lateral_deviation_ratio, frontal_deviation_ratio) < (
        1.5 / sensitivity_factor
    ):
        feedback["trajectory"] = "Excellent 3D movement trajectory."
    else:
        feedback["trajectory"] = "Movement trajectory can be improved."

    # EXACT METRICS FROM CURRENT CODE
    metrics = {
        "user_lateral_deviation": lateral_deviation_user,
        "expert_lateral_deviation": lateral_deviation_expert,
        "lateral_deviation_ratio": lateral_deviation_ratio,
        "user_frontal_deviation": frontal_deviation_user,
        "expert_frontal_deviation": frontal_deviation_expert,
        "frontal_deviation_ratio": frontal_deviation_ratio,
        "trajectory_difference_x": normalized_trajectory_diff_x,
        "trajectory_difference_z": normalized_trajectory_diff_z,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_speed_universal(
    user_data,
    expert_data,
    exercise_config,
    landmarks_config,
    config_path="config.json",
    user_repetitions=None,
    expert_repetitions=None,
):
    """
    UNIVERSAL: Análisis de velocidad usando duración real de repeticiones.
    """
    # Verificar que tenemos repeticiones
    if not user_repetitions or not expert_repetitions:
        raise ValueError(
            "Se requieren user_repetitions y expert_repetitions para análisis de velocidad"
        )

    # Obtener exercise_name del config_path
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "speed", exercise_name, config_path
    )

    # Calcular duración real usando repeticiones
    from src.utils.repetition_utils import calculate_exercise_total_duration

    duration_metrics = calculate_exercise_total_duration(
        user_repetitions, expert_repetitions
    )
    speed_ratio = duration_metrics["speed_ratio"]

    # Obtener penalty de configuración y calcular score
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="speed",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        speed_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(base_score, sensitivity_factor, "velocidad")

    # Feedback basado en duración real
    feedback = {}
    if speed_ratio > 1.4:
        feedback["speed"] = (
            "You are doing the exercise too fast. Take more time to control the movement."
        )
    elif speed_ratio > 1.15:
        feedback["speed"] = (
            "The pace is a bit fast. Try to slow down slightly for better control."
        )
    elif speed_ratio < 0.6:
        feedback["speed"] = (
            "You are doing the exercise very slowly. You can increase the pace a bit."
        )
    elif speed_ratio < 0.85:
        feedback["speed"] = (
            "The pace is a bit slow. You can speed up slightly while maintaining control."
        )
    else:
        feedback["speed"] = "Excellent execution pace."

    metrics = {
        "analysis_method": "real_repetition_duration",
        **duration_metrics,
        "user_duration_seconds": duration_metrics["user_frames"] / 30,
        "expert_duration_seconds": duration_metrics["expert_frames_scaled"] / 30,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}
