# backend/src/utils/analysis_utils.py - UNIFIED VERSION
import numpy as np
import logging
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


def apply_unified_sensitivity(base_score, sensitivity_factor, metric_name="unknown"):
    """
    Applies sensitivity in a unified and bidirectional way to all metrics.

    Args:
        base_score: Base score calculated without sensitivity
        sensitivity_factor: Sensitivity factor (0.5=permissive, 1.0=normal, 2.0=strict)
        metric_name: Metric name for logging

    Returns:
        Score adjusted with applied sensitivity
    """
    if sensitivity_factor <= 0:
        logger.warning(
            f"Invalid sensitivity factor for {metric_name}: {sensitivity_factor}. Using 1.0"
        )
        sensitivity_factor = 1.0

    # Calculate adjustment based on deviation from normal factor (1.0)
    if sensitivity_factor < 1.0:
        # Low factor = More permissive = BONUS (score goes up)
        bonus_factor = 1.0 - sensitivity_factor  # 0.5 → 0.5, 0.8 → 0.2
        bonus = min(20, bonus_factor * 40)  # Maximum bonus of 20 points
        adjusted_score = min(100, base_score + bonus)
        logger.debug(
            f"{metric_name}: Factor {sensitivity_factor:.2f} → Bonus +{bonus:.1f} → {base_score:.1f}→{adjusted_score:.1f}"
        )

    elif sensitivity_factor > 1.0:
        # High factor = More strict = PENALTY (score goes down)
        penalty_factor = sensitivity_factor - 1.0  # 2.0 → 1.0, 1.5 → 0.5
        # Limit maximum penalty to avoid collapses
        max_penalty = 30 if sensitivity_factor > 2.0 else 25
        penalty = min(max_penalty, penalty_factor * 30)
        adjusted_score = max(10, base_score - penalty)  # Minimum score of 10
        logger.debug(
            f"{metric_name}: Factor {sensitivity_factor:.2f} → Penalty -{penalty:.1f} → {base_score:.1f}→{adjusted_score:.1f}"
        )

    else:
        # Normal factor = No changes
        adjusted_score = base_score
        logger.debug(
            f"{metric_name}: Factor {sensitivity_factor:.2f} → No changes → {adjusted_score:.1f}"
        )

    return adjusted_score


def calculate_deviation_score(
    actual_value, ideal_value, max_penalty=30, metric_type="linear"
):
    """
    Calculates base score based on deviation from ideal value.

    Args:
        actual_value: Actual measured value
        ideal_value: Ideal/target value
        max_penalty: Maximum penalty for complete deviation
        metric_type: "linear", "ratio", or "logarithmic"

    Returns:
        Base score (before applying sensitivity)
    """
    if metric_type == "ratio" and ideal_value != 0:
        # For ratios (e.g.: user/expert), ideal is normally 1.0
        deviation = abs(actual_value - ideal_value) / abs(ideal_value)
    elif metric_type == "logarithmic":
        # For metrics that need smooth changes in wide ranges
        deviation = abs(
            np.log(max(0.01, actual_value)) - np.log(max(0.01, ideal_value))
        )
    else:
        # Linear - for absolute differences
        deviation = abs(actual_value - ideal_value)

    # Convert deviation to penalty (normalized)
    # We assume 100% deviation = maximum penalty
    penalty = min(max_penalty, deviation * max_penalty)
    base_score = max(20, 100 - penalty)  # Minimum base score of 20

    return base_score


def get_exercise_config(
    exercise_name="military_press_dumbbell", config_path="config.json"
):
    """
    Gets specific configuration for the exercise from config.json.
    USES SINGLETON + READS CONFIGURATION FROM JSON FILE.
    """
    try:
        # Use config_manager singleton
        exercise_config = config_manager.get_exercise_config(exercise_name, config_path)

        # Load global configurations if not loaded
        if config_path not in config_manager._loaded_files:
            config_manager.load_config_file(config_path)
        config_data = config_manager._loaded_files[config_path]

        # Get exercise-specific analysis configuration
        analysis_config = exercise_config.get("analysis_config", {})

        # Get global analysis configuration
        global_analysis = config_data.get("global_analysis_config", {})

        # Combine: global first, then exercise-specific (priority)
        final_config = {}
        final_config.update(global_analysis)
        final_config.update(analysis_config)

        # Add additional configurations from global level
        final_config.update(
            {
                "scoring_weights": config_data.get("scoring_weights", {}),
                "analysis_ratios": config_data.get("analysis_ratios", {}),
                "feedback_multipliers": config_data.get("feedback_multipliers", {}),
                "skill_levels": config_data.get("skill_levels", {}),
                "signal_processing": config_data.get("signal_processing", {}),
            }
        )

        # If there's not enough configuration, complete with default values
        default_config = {
            "min_elbow_angle": 45,
            "max_elbow_angle": 175,
            "rom_threshold": 0.85,
            "bottom_diff_threshold": 0.2,
            "abduction_angle_threshold": 15,
            "symmetry_threshold": 0.15,
            "lateral_dev_threshold": 0.2,
            "frontal_dev_threshold": 0.15,
            "velocity_ratio_threshold": 0.3,
            "scapular_stability_threshold": 1.5,
            "sensitivity_factors": {
                "amplitude": 3.0,
                "elbow_abduction": 3.0,
                "symmetry": 1.0,
                "trajectory": 1.0,
                "speed": 1.0,
                "scapular_stability": 1.0,
            },
            "scoring_weights": {
                "rom_score": 0.20,
                "abduction_score": 0.20,
                "sym_score": 0.15,
                "path_score": 0.20,
                "speed_score": 0.15,
                "scapular_score": 0.10,
            },
        }

        # Complete missing values with defaults
        for key, value in default_config.items():
            if key not in final_config:
                final_config[key] = value

        logger.info(f"Analysis configuration loaded for {exercise_name}")
        return final_config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.warning("Using complete default configuration")
        # Complete default values
        return {
            "min_elbow_angle": 45,
            "max_elbow_angle": 175,
            "rom_threshold": 0.85,
            "bottom_diff_threshold": 0.2,
            "abduction_angle_threshold": 15,
            "symmetry_threshold": 0.15,
            "lateral_dev_threshold": 0.2,
            "frontal_dev_threshold": 0.15,
            "velocity_ratio_threshold": 0.3,
            "scapular_stability_threshold": 1.5,
            "sensitivity_factors": {
                "amplitude": 3.0,
                "elbow_abduction": 3.0,
                "symmetry": 1.0,
                "trajectory": 1.0,
                "speed": 1.0,
                "scapular_stability": 1.0,
            },
            "scoring_weights": {
                "rom_score": 0.20,
                "abduction_score": 0.20,
                "sym_score": 0.15,
                "path_score": 0.20,
                "speed_score": 0.15,
                "scapular_score": 0.10,
            },
        }


def calculate_elbow_abduction_angle(shoulder_point, elbow_point):
    """
    Calculates lateral abduction angle of elbow.
    """
    shoulder = np.array(shoulder_point)
    elbow = np.array(elbow_point)

    vector_shoulder_to_elbow = elbow - shoulder
    horizontal_projection = np.array(
        [
            vector_shoulder_to_elbow[0],
            vector_shoulder_to_elbow[2],
        ]
    )

    x_axis = np.array([1.0, 0.0])
    projection_magnitude = np.linalg.norm(horizontal_projection)

    if projection_magnitude < 1e-6:
        return 90.0

    normalized_projection = horizontal_projection / projection_magnitude
    dot_product = np.dot(normalized_projection, x_axis)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(abs(dot_product))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def apply_sensitivity_to_threshold(threshold, sensitivity_factor):
    """Applies sensitivity factor to a threshold (LEGACY METHOD - maintain compatibility)."""
    if sensitivity_factor <= 0:
        logger.warning(f"Invalid sensitivity factor: {sensitivity_factor}. Using 1.0")
        sensitivity_factor = 1.0

    adjusted_threshold = threshold / sensitivity_factor
    logger.debug(
        f"Adjusted threshold: {threshold} / {sensitivity_factor} = {adjusted_threshold}"
    )
    return adjusted_threshold


def determine_skill_level(overall_score, exercise_config=None):
    """Determines skill level based on score."""
    # Use configuration if available, otherwise original values
    if exercise_config and "skill_levels" in exercise_config:
        skill_levels = exercise_config["skill_levels"]
        if overall_score >= skill_levels.get("excellent", 90):
            return "Excellent"
        elif overall_score >= skill_levels.get("very_good", 80):
            return "Very Good"
        elif overall_score >= skill_levels.get("good", 70):
            return "Good"
        elif overall_score >= skill_levels.get("acceptable", 60):
            return "Acceptable"
        elif overall_score >= skill_levels.get("needs_improvement", 50):
            return "Needs Improvement"
        else:
            return "Beginner"
    else:
        # Original hardcoded values to maintain compatibility
        if overall_score >= 90:
            return "Excellent"
        elif overall_score >= 80:
            return "Very Good"
        elif overall_score >= 70:
            return "Good"
        elif overall_score >= 60:
            return "Acceptable"
        elif overall_score >= 50:
            return "Needs Improvement"
        else:
            return "Beginner"


def generate_recommendations(all_feedback, overall_score):
    """Generates specific recommendations based on feedback."""
    recommendations = []

    recommendation_map = {
        "insufficient": "Practice the complete movement with less weight to improve amplitude.",
        "low_position": "Work on bringing elbows to shoulder height when lowering.",
        "open": "Perform body awareness exercises in front of a mirror to correct elbow abduction.",
        "closed": "Try to keep elbows in an intermediate position, neither too open nor too closed.",
        "asymmetry": "Perform unilateral exercises (one arm at a time) to balance strength between both sides.",
        "deviates": "Practice in front of a mirror with a light bar or no weight to correct trajectory.",
        "lateral": "Focus on maintaining a vertical movement, avoiding lateral deviations.",
        "frontal": "Avoid pushing weights forward or backward, maintain a vertical plane.",
        "slow": "Incorporate some sets with less weight but greater controlled speed in the upward phase.",
        "fast": "Count mentally during the descent to ensure controlled lowering (approx. 2-3 seconds).",
        "instability": "Strengthen scapular belt muscles with specific exercises like scapular retractions.",
        "move": "Practice keeping shoulders fixed throughout the entire press movement.",
    }

    for category, message in all_feedback.items():
        for problem, recommendation in recommendation_map.items():
            if problem in message.lower():
                recommendations.append(recommendation)
                break

    if not recommendations and overall_score < 80:
        recommendations.extend(
            [
                "Record yourself performing the exercise regularly to review your technique.",
                "Consider performing the exercise with less weight to focus on technique.",
            ]
        )

    if len(recommendations) < 2:
        if overall_score < 70:
            recommendations.append(
                "Consider some sessions with a personal trainer to perfect your technique."
            )
        if overall_score < 60:
            recommendations.append(
                "Start with simpler variants of the military press, like seated press with back support."
            )

    return recommendations
