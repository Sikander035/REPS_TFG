# backend/src/feedback/specific_metrics.py - UNIFIED VERSION
import sys
import numpy as np
import pandas as pd
import os
import logging
from scipy.signal import find_peaks, savgol_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.analysis_utils import (
    apply_unified_sensitivity,
    calculate_deviation_score,
    calculate_elbow_abduction_angle,  # Keep original function
    apply_sensitivity_to_threshold,
)
from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


# =============================================================================
# MILITARY PRESS - SPECIFIC METRICS (using config_manager)
# =============================================================================


def analyze_elbow_abduction_angle_press(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    MILITARY PRESS: Elbow abduction analysis using config_manager.
    """
    exercise_name = "military_press_dumbbell"

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "elbow_abduction", exercise_name, config_path
    )

    user_right_abduction = []
    user_left_abduction = []
    expert_right_abduction = []
    expert_left_abduction = []

    # Calculate abduction angle for ALL frames
    logger.info("Calculating lateral abduction angles (XZ projection with X axis)")

    for i in range(len(user_data)):
        try:
            # === USER - BOTH ELBOWS ===
            user_right_shoulder = [
                user_data.iloc[i]["landmark_right_shoulder_x"],
                user_data.iloc[i]["landmark_right_shoulder_y"],
                user_data.iloc[i]["landmark_right_shoulder_z"],
            ]
            user_right_elbow = [
                user_data.iloc[i]["landmark_right_elbow_x"],
                user_data.iloc[i]["landmark_right_elbow_y"],
                user_data.iloc[i]["landmark_right_elbow_z"],
            ]
            user_left_shoulder = [
                user_data.iloc[i]["landmark_left_shoulder_x"],
                user_data.iloc[i]["landmark_left_shoulder_y"],
                user_data.iloc[i]["landmark_left_shoulder_z"],
            ]
            user_left_elbow = [
                user_data.iloc[i]["landmark_left_elbow_x"],
                user_data.iloc[i]["landmark_left_elbow_y"],
                user_data.iloc[i]["landmark_left_elbow_z"],
            ]

            # Check for NaN in coordinates
            if (
                not np.isnan(user_right_shoulder).any()
                and not np.isnan(user_right_elbow).any()
                and not np.isnan(user_left_shoulder).any()
                and not np.isnan(user_left_elbow).any()
            ):
                # Calculate user abduction angles
                user_right_angle = calculate_elbow_abduction_angle(
                    user_right_shoulder, user_right_elbow
                )
                user_left_angle = calculate_elbow_abduction_angle(
                    user_left_shoulder, user_left_elbow
                )

                user_right_abduction.append(user_right_angle)
                user_left_abduction.append(user_left_angle)
            else:
                user_right_abduction.append(np.nan)
                user_left_abduction.append(np.nan)

        except Exception as e:
            logger.warning(f"Error calculating user abduction angle at frame {i}: {e}")
            user_right_abduction.append(np.nan)
            user_left_abduction.append(np.nan)

    # Calculate expert angles (similar to user)
    for i in range(len(expert_data)):
        try:
            expert_right_shoulder = [
                expert_data.iloc[i]["landmark_right_shoulder_x"],
                expert_data.iloc[i]["landmark_right_shoulder_y"],
                expert_data.iloc[i]["landmark_right_shoulder_z"],
            ]
            expert_right_elbow = [
                expert_data.iloc[i]["landmark_right_elbow_x"],
                expert_data.iloc[i]["landmark_right_elbow_y"],
                expert_data.iloc[i]["landmark_right_elbow_z"],
            ]
            expert_left_shoulder = [
                expert_data.iloc[i]["landmark_left_shoulder_x"],
                expert_data.iloc[i]["landmark_left_shoulder_y"],
                expert_data.iloc[i]["landmark_left_shoulder_z"],
            ]
            expert_left_elbow = [
                expert_data.iloc[i]["landmark_left_elbow_x"],
                expert_data.iloc[i]["landmark_left_elbow_y"],
                expert_data.iloc[i]["landmark_left_elbow_z"],
            ]

            if (
                not np.isnan(expert_right_shoulder).any()
                and not np.isnan(expert_right_elbow).any()
                and not np.isnan(expert_left_shoulder).any()
                and not np.isnan(expert_left_elbow).any()
            ):
                expert_right_angle = calculate_elbow_abduction_angle(
                    expert_right_shoulder, expert_right_elbow
                )
                expert_left_angle = calculate_elbow_abduction_angle(
                    expert_left_shoulder, expert_left_elbow
                )

                expert_right_abduction.append(expert_right_angle)
                expert_left_abduction.append(expert_left_angle)
            else:
                expert_right_abduction.append(np.nan)
                expert_left_abduction.append(np.nan)

        except Exception as e:
            logger.warning(
                f"Error calculating expert abduction angle at frame {i}: {e}"
            )
            expert_right_abduction.append(np.nan)
            expert_left_abduction.append(np.nan)

    if not user_right_abduction or not expert_right_abduction:
        return {
            "metrics": {},
            "feedback": {"elbow_abduction": "Could not analyze elbow abduction."},
            "score": 50,
        }

    user_right_abduction = np.array(user_right_abduction)
    user_left_abduction = np.array(user_left_abduction)
    expert_right_abduction = np.array(expert_right_abduction)
    expert_left_abduction = np.array(expert_left_abduction)

    user_avg_signal = (user_right_abduction + user_left_abduction) / 2
    expert_avg_signal = (expert_right_abduction + expert_left_abduction) / 2

    user_valid = ~np.isnan(user_avg_signal)
    expert_valid = ~np.isnan(expert_avg_signal)

    if np.sum(user_valid) < 10 or np.sum(expert_valid) < 10:
        return {
            "metrics": {},
            "feedback": {"elbow_abduction": "Insufficient data to analyze abduction."},
            "score": 50,
        }

    user_clean_signal = user_avg_signal[user_valid]
    expert_clean_signal = expert_avg_signal[expert_valid]

    try:
        window_length = min(9, len(user_clean_signal) // 4)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(3, window_length)

        user_smooth = savgol_filter(user_clean_signal, window_length, 2)
        expert_smooth = savgol_filter(expert_clean_signal, window_length, 2)

        prominence = max(2, np.std(user_smooth) * 0.3)
        distance = max(15, len(user_smooth) // 12)
        height = np.percentile(user_smooth, 30)

        user_valleys, _ = find_peaks(
            -user_smooth,
            prominence=prominence,
            distance=distance,
            height=-height,
        )

        if len(user_valleys) < 2:
            prominence = max(1, np.std(user_smooth) * 0.2)
            height = np.percentile(user_smooth, 40)
            user_valleys, _ = find_peaks(
                -user_smooth, prominence=prominence, distance=distance, height=-height
            )

        if len(user_valleys) < 2:
            user_threshold = np.percentile(user_smooth, 10)
            user_low_indices = np.where(user_smooth <= user_threshold)[0]
            if len(user_low_indices) < 3:
                user_threshold = np.percentile(user_smooth, 15)
                user_low_indices = np.where(user_smooth <= user_threshold)[0]
            user_valley_values = user_clean_signal[user_low_indices]
            expert_valley_values = expert_clean_signal[user_low_indices]
        else:
            user_valley_values = user_clean_signal[user_valleys]
            expert_valley_values = expert_clean_signal[user_valleys]

        user_min_abduction = np.mean(user_valley_values)
        expert_min_abduction = np.mean(expert_valley_values)
        user_absolute_min = np.min(user_valley_values)
        expert_absolute_min = np.min(expert_valley_values)
        abduction_diff = user_min_abduction - expert_min_abduction
        absolute_diff = user_absolute_min - expert_absolute_min

    except Exception as e:
        logger.error(f"Error in minimum detection: {e}")
        user_min_abduction = np.mean(user_clean_signal)
        expert_min_abduction = np.mean(expert_clean_signal)
        abduction_diff = user_min_abduction - expert_min_abduction
        absolute_diff = abduction_diff
        user_valley_values = user_clean_signal
        expert_valley_values = expert_clean_signal

    # Calculate base score using config_manager
    max_penalty = config_manager.get_penalty_config(
        exercise_name=exercise_name,
        metric_type="specific",
        metric_name="elbow_abduction",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        abs(abduction_diff), 0, max_penalty=max_penalty, metric_type="linear"
    )

    # Apply unified sensitivity
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "elbow_abduction"
    )

    # Feedback based on final score for consistency
    feedback = {}

    if final_score >= 80:
        feedback["elbow_abduction"] = "Excellent lateral elbow position."
    elif final_score >= 60:
        if abduction_diff > 0:  # User has higher angle = more closed
            feedback["elbow_abduction"] = (
                "Your elbows are slightly more closed than the expert. "
                "Try to separate them a bit more from your body."
            )
        else:  # User has lower angle = more open
            feedback["elbow_abduction"] = (
                "Your elbows are slightly more open than the expert. "
                "Bring them a bit closer to your body."
            )
    elif final_score >= 40:
        # Moderate cases
        if abduction_diff > 0:
            feedback["elbow_abduction"] = (
                "Your elbows are moderately more closed than the expert. "
                "Separate them more from your body for better mechanics."
            )
        else:
            feedback["elbow_abduction"] = (
                "Your elbows open moderately during the exercise. "
                "Bring them closer to your body for greater stability."
            )
    else:
        # Critical cases
        if abduction_diff > 0:
            feedback["elbow_abduction"] = (
                "Your elbows are significantly more closed than the expert. "
                "It's important to separate them more from your body for better mechanics."
            )
        else:
            feedback["elbow_abduction"] = (
                "Your elbows open excessively during the exercise. "
                "It's critical to bring them closer to your body for safety."
            )

    metrics = {
        "user_min_lateral_abduction": user_min_abduction,
        "expert_min_lateral_abduction": expert_min_abduction,
        "abduction_difference": abduction_diff,
        "user_absolute_min": user_absolute_min,
        "expert_absolute_min": expert_absolute_min,
        "absolute_difference": absolute_diff,
        "num_minimums_detected": len(user_valley_values),
        "total_user_frames": len(user_data),
        "total_expert_frames": len(expert_data),
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_scapular_stability_press(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    MILITARY PRESS: Scapular stability analysis using config_manager.
    """
    exercise_name = "military_press_dumbbell"

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "scapular_stability", exercise_name, config_path
    )

    try:
        # Calculate shoulder center
        user_r_shoulder_y = user_data["landmark_right_shoulder_y"].values
        user_l_shoulder_y = user_data["landmark_left_shoulder_y"].values
        expert_r_shoulder_y = expert_data["landmark_right_shoulder_y"].values
        expert_l_shoulder_y = expert_data["landmark_left_shoulder_y"].values

        user_shoulder_center_y = (user_r_shoulder_y + user_l_shoulder_y) / 2
        expert_shoulder_center_y = (expert_r_shoulder_y + expert_l_shoulder_y) / 2

        # Analyze stability as variability of shoulder center
        user_shoulder_movement = float(np.std(user_shoulder_center_y))
        expert_shoulder_movement = float(np.std(expert_shoulder_center_y))
        movement_ratio = (
            user_shoulder_movement / expert_shoulder_movement
            if expert_shoulder_movement > 0
            else 1.0
        )

        # Analyze shoulder symmetry
        user_shoulder_asymmetry = float(np.std(user_r_shoulder_y - user_l_shoulder_y))
        expert_shoulder_asymmetry = float(
            np.std(expert_r_shoulder_y - expert_l_shoulder_y)
        )
        asymmetry_ratio = (
            user_shoulder_asymmetry / expert_shoulder_asymmetry
            if expert_shoulder_asymmetry > 0
            else 1.0
        )

        # Calculate base score using worst ratio
        worst_stability_ratio = max(movement_ratio, asymmetry_ratio)
        # Get penalty from configuration
        max_penalty = config_manager.get_penalty_config(
            exercise_name=exercise_name,
            metric_type="specific",
            metric_name="scapular_stability",
            config_path=config_path,
        )
        base_score = calculate_deviation_score(
            worst_stability_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
        )

        # Apply unified sensitivity (SMOOTHED to avoid collapses)
        # Limit sensitivity impact for this specific metric
        capped_sensitivity = (
            min(sensitivity_factor, 2.0)
            if sensitivity_factor > 1.5
            else sensitivity_factor
        )
        final_score = apply_unified_sensitivity(
            base_score, capped_sensitivity, "scapular_stability"
        )

        # Get threshold using config_manager
        stability_threshold = config_manager.get_analysis_threshold(
            "scapular_stability_threshold", exercise_name, config_path
        )
        stability_threshold_adj = apply_sensitivity_to_threshold(
            stability_threshold, sensitivity_factor
        )

        feedback = {}
        if movement_ratio > (2.0 / sensitivity_factor):
            if sensitivity_factor > 1.5:
                feedback["scapular_stability"] = (
                    "Your shoulders move excessively during the press. "
                    "It's critical to maintain a much more stable scapular belt position."
                )
            else:
                feedback["scapular_stability"] = (
                    "Your shoulders move excessively during the press. "
                    "Maintain a more stable scapular belt position."
                )
        elif asymmetry_ratio > (2.0 / sensitivity_factor):
            feedback["scapular_stability"] = (
                "Asymmetry detected in your shoulder movement. "
                "Focus on keeping both shoulders balanced."
            )
        elif movement_ratio > stability_threshold_adj:
            if sensitivity_factor > 1.5:
                feedback["scapular_stability"] = (
                    "Notable instability detected in your scapular belt. "
                    "It's important to practice keeping your shoulders in a more fixed position."
                )
            else:
                feedback["scapular_stability"] = (
                    "Some instability detected in your scapular belt. "
                    "Practice keeping your shoulders in a more fixed position."
                )
        else:
            feedback["scapular_stability"] = "Good scapular belt stability."

        metrics = {
            "user_shoulder_movement": user_shoulder_movement,
            "expert_shoulder_movement": expert_shoulder_movement,
            "movement_ratio": float(movement_ratio),
            "user_shoulder_asymmetry": user_shoulder_asymmetry,
            "expert_shoulder_asymmetry": expert_shoulder_asymmetry,
            "asymmetry_ratio": float(asymmetry_ratio),
        }

        return {"metrics": metrics, "feedback": feedback, "score": final_score}

    except Exception as e:
        logger.error(f"Error in scapular stability analysis: {e}")
        return {
            "metrics": {
                "user_shoulder_movement": 0.0,
                "expert_shoulder_movement": 0.0,
                "movement_ratio": 1.0,
                "user_shoulder_asymmetry": 0.0,
                "expert_shoulder_asymmetry": 0.0,
                "asymmetry_ratio": 1.0,
            },
            "feedback": {
                "scapular_stability": "Scapular stability analysis not available."
            },
            "score": 50,
        }


# =============================================================================
# SQUAT - SPECIFIC METRICS (using config_manager)
# =============================================================================


def analyze_squat_depth(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    SQUAT: Specific depth analysis using knee angle.
    """
    exercise_name = "squat"

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "depth", exercise_name, config_path
    )

    user_knee_angles = []
    expert_knee_angles = []

    for i in range(len(user_data)):
        try:
            # USER - calculate knee angle (hip-knee-ankle)
            user_hip = [
                user_data.iloc[i]["landmark_left_hip_x"],
                user_data.iloc[i]["landmark_left_hip_y"],
                user_data.iloc[i]["landmark_left_hip_z"],
            ]
            user_knee = [
                user_data.iloc[i]["landmark_left_knee_x"],
                user_data.iloc[i]["landmark_left_knee_y"],
                user_data.iloc[i]["landmark_left_knee_z"],
            ]
            user_ankle = [
                user_data.iloc[i]["landmark_left_ankle_x"],
                user_data.iloc[i]["landmark_left_ankle_y"],
                user_data.iloc[i]["landmark_left_ankle_z"],
            ]

            if not any(np.isnan([*user_hip, *user_knee, *user_ankle])):
                angle = _calculate_angle_3points(user_hip, user_knee, user_ankle)
                user_knee_angles.append(angle)

            # EXPERT - similar
            expert_hip = [
                expert_data.iloc[i]["landmark_left_hip_x"],
                expert_data.iloc[i]["landmark_left_hip_y"],
                expert_data.iloc[i]["landmark_left_hip_z"],
            ]
            expert_knee = [
                expert_data.iloc[i]["landmark_left_knee_x"],
                expert_data.iloc[i]["landmark_left_knee_y"],
                expert_data.iloc[i]["landmark_left_knee_z"],
            ]
            expert_ankle = [
                expert_data.iloc[i]["landmark_left_ankle_x"],
                expert_data.iloc[i]["landmark_left_ankle_y"],
                expert_data.iloc[i]["landmark_left_ankle_z"],
            ]

            if not any(np.isnan([*expert_hip, *expert_knee, *expert_ankle])):
                angle = _calculate_angle_3points(expert_hip, expert_knee, expert_ankle)
                expert_knee_angles.append(angle)

        except Exception as e:
            logger.warning(f"Error calculating knee angle at frame {i}: {e}")
            continue

    if not user_knee_angles or not expert_knee_angles:
        return {
            "metrics": {},
            "feedback": {"squat_depth": "Could not analyze squat depth."},
            "score": 50,
        }

    # Calculate minimum angle (maximum flexion)
    user_min_angle = np.min(user_knee_angles)
    expert_min_angle = np.min(expert_knee_angles)
    angle_diff = abs(user_min_angle - expert_min_angle)

    # Calculate score using config_manager
    max_penalty = config_manager.get_penalty_config(
        exercise_name=exercise_name,
        metric_type="specific",
        metric_name="squat_depth",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        angle_diff, 0, max_penalty=max_penalty, metric_type="linear"
    )
    final_score = apply_unified_sensitivity(base_score, sensitivity_factor, "depth")

    # Generate feedback
    feedback = {}
    if final_score >= 85:
        feedback["squat_depth"] = "Excellent squat depth."
    elif final_score >= 70:
        if user_min_angle > expert_min_angle + 10:
            feedback["squat_depth"] = (
                "Your squat could be deeper. Lower your hips more."
            )
        else:
            feedback["squat_depth"] = "Good squat depth."
    else:
        if user_min_angle > expert_min_angle + 15:
            feedback["squat_depth"] = (
                "Your squat is very shallow. It's important to go lower for correct technique."
            )
        else:
            feedback["squat_depth"] = "Your squat depth needs work."

    metrics = {
        "user_min_angle": user_min_angle,
        "expert_min_angle": expert_min_angle,
        "angle_difference": angle_diff,
        "frames_analyzed": len(user_knee_angles),
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_knee_tracking_squat(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    SQUAT: Knee tracking analysis using config_manager.
    """
    exercise_name = "squat"

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "knee_tracking", exercise_name, config_path
    )

    # Calculate knee separation throughout movement
    user_knee_separation = []
    expert_knee_separation = []

    for i in range(len(user_data)):
        try:
            # Knee separation (distance in X)
            user_sep = abs(
                user_data.iloc[i]["landmark_left_knee_x"]
                - user_data.iloc[i]["landmark_right_knee_x"]
            )
            expert_sep = abs(
                expert_data.iloc[i]["landmark_left_knee_x"]
                - expert_data.iloc[i]["landmark_right_knee_x"]
            )

            if not np.isnan(user_sep) and not np.isnan(expert_sep):
                user_knee_separation.append(user_sep)
                expert_knee_separation.append(expert_sep)

        except Exception as e:
            logger.warning(f"Error calculating knee separation at frame {i}: {e}")
            continue

    if not user_knee_separation or not expert_knee_separation:
        return {
            "metrics": {},
            "feedback": {"knee_tracking": "Could not analyze knee tracking."},
            "score": 50,
        }

    # Analyze separation variability (less variability = better tracking)
    user_knee_stability = np.std(user_knee_separation)
    expert_knee_stability = np.std(expert_knee_separation)
    stability_ratio = (
        user_knee_stability / expert_knee_stability if expert_knee_stability > 0 else 1
    )

    # Calculate score using config_manager
    max_penalty = config_manager.get_penalty_config(
        exercise_name=exercise_name,
        metric_type="specific",
        metric_name="knee_tracking",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        stability_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "knee_tracking"
    )

    # Generate feedback
    feedback = {}
    if final_score >= 85:
        feedback["knee_tracking"] = "Excellent knee tracking control."
    elif final_score >= 70:
        feedback["knee_tracking"] = "Good overall knee control with slight variations."
    else:
        feedback["knee_tracking"] = (
            "Knees tend to move inward. Keep knees aligned with feet."
        )

    metrics = {
        "user_knee_stability": user_knee_stability,
        "expert_knee_stability": expert_knee_stability,
        "stability_ratio": stability_ratio,
        "user_avg_separation": np.mean(user_knee_separation),
        "expert_avg_separation": np.mean(expert_knee_separation),
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


# =============================================================================
# PULL-UP - SPECIFIC METRICS (using config_manager)
# =============================================================================


def analyze_body_swing_control_pullup(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    PULL-UP: Specific body swing control analysis.
    """
    exercise_name = "pull_up"

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "swing_control", exercise_name, config_path
    )

    # Analyze hip movement as swing indicator
    user_hip_center_x = (
        user_data["landmark_left_hip_x"].values
        + user_data["landmark_right_hip_x"].values
    ) / 2
    user_hip_center_z = (
        user_data["landmark_left_hip_z"].values
        + user_data["landmark_right_hip_z"].values
    ) / 2

    expert_hip_center_x = (
        expert_data["landmark_left_hip_x"].values
        + expert_data["landmark_right_hip_x"].values
    ) / 2
    expert_hip_center_z = (
        expert_data["landmark_left_hip_z"].values
        + expert_data["landmark_right_hip_z"].values
    ) / 2

    # Calculate hip movement variability
    user_swing_x = np.std(user_hip_center_x)
    user_swing_z = np.std(user_hip_center_z)
    expert_swing_x = np.std(expert_hip_center_x)
    expert_swing_z = np.std(expert_hip_center_z)

    # Calculate total swing ratio
    user_total_swing = np.sqrt(user_swing_x**2 + user_swing_z**2)
    expert_total_swing = np.sqrt(expert_swing_x**2 + expert_swing_z**2)
    swing_ratio = user_total_swing / expert_total_swing if expert_total_swing > 0 else 1

    # Calculate score using config_manager
    max_penalty = config_manager.get_penalty_config(
        exercise_name=exercise_name,
        metric_type="specific",
        metric_name="swing_control",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        swing_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "swing_control"
    )

    # Generate feedback
    feedback = {}
    if final_score >= 85:
        feedback["swing_control"] = "Excellent body control during pull-up."
    elif final_score >= 70:
        feedback["swing_control"] = "Good body control with slight swaying."
    else:
        feedback["swing_control"] = (
            "Too much body swaying. Keep core engaged for greater stability."
        )

    metrics = {
        "user_swing_x": user_swing_x,
        "user_swing_z": user_swing_z,
        "user_total_swing": user_total_swing,
        "expert_swing_x": expert_swing_x,
        "expert_swing_z": expert_swing_z,
        "expert_total_swing": expert_total_swing,
        "swing_ratio": swing_ratio,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_scapular_retraction_pullup(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    PULL-UP: Scapular retraction analysis at movement start.
    """
    exercise_name = "pull_up"

    # Get sensitivity factor using config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "scapular_retraction", exercise_name, config_path
    )

    # Analyze shoulder separation (less separation = more retraction)
    user_shoulder_separation = abs(
        user_data["landmark_left_shoulder_x"].values
        - user_data["landmark_right_shoulder_x"].values
    )
    expert_shoulder_separation = abs(
        expert_data["landmark_left_shoulder_x"].values
        - expert_data["landmark_right_shoulder_x"].values
    )

    # Analyze initial frames (starting position)
    initial_frames = min(10, len(user_shoulder_separation) // 4)
    user_initial_separation = np.mean(user_shoulder_separation[:initial_frames])
    expert_initial_separation = np.mean(expert_shoulder_separation[:initial_frames])

    # Initial separation ratio
    separation_ratio = (
        user_initial_separation / expert_initial_separation
        if expert_initial_separation > 0
        else 1
    )

    # Calculate score using config_manager
    max_penalty = config_manager.get_penalty_config(
        exercise_name=exercise_name,
        metric_type="specific",
        metric_name="scapular_retraction",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        separation_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "scapular_retraction"
    )

    # Generate feedback
    feedback = {}
    if final_score >= 85:
        feedback["scapular_retraction"] = "Excellent scapular retraction at start."
    elif final_score >= 70:
        feedback["scapular_retraction"] = (
            "Good scapular retraction with slight variations."
        )
    else:
        if separation_ratio > 1.1:
            feedback["scapular_retraction"] = (
                "Need more scapular retraction. Pull shoulder blades together more at start."
            )
        else:
            feedback["scapular_retraction"] = "Scapular retraction needs work."

    metrics = {
        "user_initial_separation": user_initial_separation,
        "expert_initial_separation": expert_initial_separation,
        "separation_ratio": separation_ratio,
        "frames_analyzed": initial_frames,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================


def _calculate_angle_3points(p1, p2, p3):
    """Calculate angle between 3 points (p2 is vertex)"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def _calculate_angle_2vectors(v1, v2):
    """Calculate angle between 2 vectors"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


# =============================================================================
# FACTORY FOR SPECIFIC METRICS
# =============================================================================


def get_specific_metrics_for_exercise(exercise_name):
    """
    Factory that returns specific metric functions for each exercise.

    Returns:
        dict: Dictionary with exercise-specific functions

    Raises:
        ValueError: If exercise is not supported
    """
    exercise_name = exercise_name.lower().replace(" ", "_")

    if exercise_name == "military_press_dumbbell":
        return {
            "specific_metric_a": analyze_elbow_abduction_angle_press,
            "specific_metric_b": analyze_scapular_stability_press,
        }
    elif exercise_name == "squat":
        return {
            "specific_metric_a": analyze_squat_depth,
            "specific_metric_b": analyze_knee_tracking_squat,
        }
    elif exercise_name == "pull_up":
        return {
            "specific_metric_a": analyze_body_swing_control_pullup,
            "specific_metric_b": analyze_scapular_retraction_pullup,
        }
    else:
        raise ValueError(f"Unsupported exercise: {exercise_name}")


def get_specific_metric_names_for_exercise(exercise_name):
    """
    Returns the names of specific metrics for an exercise.

    Returns:
        dict: Names of specific metrics

    Raises:
        ValueError: If exercise is not supported
    """
    exercise_name = exercise_name.lower().replace(" ", "_")

    if exercise_name == "military_press_dumbbell":
        return {
            "specific_metric_a": "abduction_score",
            "specific_metric_b": "scapular_score",
        }
    elif exercise_name == "squat":
        return {
            "specific_metric_a": "depth_score",
            "specific_metric_b": "knee_score",
        }
    elif exercise_name == "pull_up":
        return {
            "specific_metric_a": "swing_score",
            "specific_metric_b": "retraction_score",
        }
    else:
        raise ValueError(f"Unsupported exercise: {exercise_name}")
