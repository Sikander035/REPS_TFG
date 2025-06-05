# backend/src/feedback/analysis_report.py - UNIFIED VERSION
import sys
import numpy as np
import pandas as pd
import os
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# IMPORT UNIVERSAL AND SPECIFIC FUNCTIONS
from src.feedback.universal_metrics import (
    analyze_movement_amplitude_universal,
    analyze_symmetry_universal,
    analyze_movement_trajectory_3d_universal,
    analyze_speed_universal,
)
from src.feedback.specific_metrics import (
    get_specific_metrics_for_exercise,
    get_specific_metric_names_for_exercise,
)

# KEEP ORIGINAL IMPORTS
from src.utils.analysis_utils import (
    determine_skill_level,
    generate_recommendations,
    apply_unified_sensitivity,
    calculate_deviation_score,
    calculate_elbow_abduction_angle,
    apply_sensitivity_to_threshold,
)

from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


def run_exercise_analysis(
    user_data,
    expert_data,
    exercise_name="press_militar",
    config_path="config.json",
    user_repetitions=None,
    expert_repetitions=None,
):
    """
    MAIN FUNCTION: Complete analysis using config_manager strictly.
    """
    logger.info(f"Starting strict analysis for: {exercise_name}")

    # Load configuration using config_manager - MANDATORY
    try:
        exercise_config = config_manager.get_exercise_config(exercise_name, config_path)
        # Add exercise_name to config for universal metrics
        exercise_config["_exercise_name"] = exercise_name
    except Exception as e:
        logger.error(
            f"Failed to load configuration for exercise '{exercise_name}': {e}"
        )
        raise ValueError(
            f"Configuration loading failed for exercise '{exercise_name}': {e}"
        )

    # Get landmarks configuration from config_manager
    try:
        landmarks_config = config_manager.get_exercise_landmarks_config(
            exercise_name, config_path
        )
    except Exception as e:
        logger.error(
            f"Failed to load landmarks configuration for exercise '{exercise_name}': {e}"
        )
        raise ValueError(
            f"Landmarks configuration loading failed for exercise '{exercise_name}': {e}"
        )

    # Get specific functions for this exercise
    try:
        specific_functions = get_specific_metrics_for_exercise(exercise_name)
        specific_names = get_specific_metric_names_for_exercise(exercise_name)
    except ValueError as e:
        logger.error(f"Exercise '{exercise_name}' not supported: {e}")
        raise

    # =================================================================
    # EXECUTE THE 4 UNIVERSAL METRICS (PASSING config_path)
    # =================================================================

    try:
        # 1. AMPLITUDE (universal)
        amplitude_result = analyze_movement_amplitude_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["amplitude"],
            config_path,
        )

        # 2. SYMMETRY (universal)
        symmetry_result = analyze_symmetry_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["symmetry"],
            config_path,
        )

        # 3. TRAJECTORY (universal)
        trajectory_result = analyze_movement_trajectory_3d_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["trajectory"],
            config_path,
        )

        # 4. SPEED (universal)
        speed_result = analyze_speed_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["speed"],
            config_path,
            user_repetitions=user_repetitions,
            expert_repetitions=expert_repetitions,
        )

    except Exception as e:
        logger.error(f"Error in universal metrics analysis: {e}")
        raise ValueError(f"Universal metrics analysis failed: {e}")

    # =================================================================
    # EXECUTE THE 2 SPECIFIC METRICS (PASSING config_path)
    # =================================================================

    try:
        # 5. SPECIFIC METRIC A (abduction/depth/swing)
        specific_a_result = specific_functions["specific_metric_a"](
            user_data, expert_data, exercise_config, config_path
        )

        # 6. SPECIFIC METRIC B (stability/tracking/retraction)
        specific_b_result = specific_functions["specific_metric_b"](
            user_data, expert_data, exercise_config, config_path
        )

    except Exception as e:
        logger.error(f"Error in specific metrics analysis: {e}")
        raise ValueError(f"Specific metrics analysis failed: {e}")

    # =================================================================
    # COMBINE RESULTS
    # =================================================================

    # Combine metrics
    all_metrics = {
        "amplitude": amplitude_result["metrics"],
        specific_names["specific_metric_a"]: specific_a_result["metrics"],
        "symmetry": symmetry_result["metrics"],
        "trajectory": trajectory_result["metrics"],
        "speed": speed_result["metrics"],
        specific_names["specific_metric_b"]: specific_b_result["metrics"],
    }

    # Combine feedback
    all_feedback = {
        **amplitude_result["feedback"],
        **specific_a_result["feedback"],
        **symmetry_result["feedback"],
        **trajectory_result["feedback"],
        **speed_result["feedback"],
        **specific_b_result["feedback"],
    }

    # Use original score names for compatibility
    individual_scores = {}

    if exercise_name.lower() == "military_press_dumbbell":
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "abduction_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "scapular_score": specific_b_result["score"],
        }
    elif exercise_name.lower() == "squat":
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "depth_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "knee_score": specific_b_result["score"],
        }
    elif exercise_name.lower() == "pull_up":
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "swing_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "retraction_score": specific_b_result["score"],
        }
    else:
        # Fallback for unrecognized exercises
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "specific_a_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "specific_b_score": specific_b_result["score"],
        }

    # VALIDATE SCORES
    for key, score in individual_scores.items():
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            logger.error(f"Invalid score detected: {key}={score}")
            raise ValueError(
                f"Invalid score for {key}: {score}. Must be between 0 and 100"
            )

    # =================================================================
    # CALCULATE GLOBAL SCORE USING CONFIG_MANAGER
    # =================================================================

    try:
        # Get weights using config_manager
        weights = config_manager.get_scoring_weights(exercise_name, config_path)

        # Verify that all weight keys exist in individual_scores
        valid_weights = {}
        total_weight = 0

        for weight_key, weight_value in weights.items():
            if weight_key in individual_scores:
                valid_weights[weight_key] = weight_value
                total_weight += weight_value
            else:
                logger.warning(
                    f"Score {weight_key} not found in individual_scores. Available: {list(individual_scores.keys())}"
                )

        if total_weight <= 0:
            raise ValueError("No valid scoring weights found or total weight is zero")

        # Normalize weights to sum 1
        valid_weights = {k: v / total_weight for k, v in valid_weights.items()}

        # Calculate global score
        overall_score = sum(
            individual_scores[key] * valid_weights[key] for key in valid_weights.keys()
        )
        overall_score = max(0, min(100, overall_score))

    except Exception as e:
        logger.error(f"Error calculating overall score: {e}")
        raise ValueError(f"Failed to calculate overall score: {e}")

    # Determine skill level using configuration if available
    try:
        # Try to get skill levels from global configuration
        if config_path not in config_manager._loaded_files:
            config_manager.load_config_file(config_path)

        config_data = config_manager._loaded_files[config_path]
        skill_levels_config = config_data.get("skill_levels")
        skill_level = determine_skill_level(overall_score, skill_levels_config)

    except Exception as e:
        logger.warning(f"Error loading skill levels from config, using defaults: {e}")
        skill_level = determine_skill_level(overall_score)

    logger.info(
        f"Strict analysis completed - Score: {overall_score:.1f}/100 - Level: {skill_level}"
    )
    logger.info(
        f"Individual scores: {[f'{k}={v:.1f}' for k,v in individual_scores.items()]}"
    )

    # RETURN SAME FORMAT AS BEFORE
    return {
        "metrics": all_metrics,
        "feedback": all_feedback,
        "score": overall_score,
        "level": skill_level,
        "individual_scores": individual_scores,
        "exercise_config": exercise_config,
        "sensitivity_factors": exercise_config.get("analysis_config", {}).get(
            "sensitivity_factors", {}
        ),
    }


def generate_analysis_report(analysis_results, exercise_name, output_path=None):
    """
    KEEP ORIGINAL FUNCTION EXACT - no changes.
    """
    if not isinstance(analysis_results, dict):
        raise ValueError("analysis_results must be a dictionary")

    if not isinstance(exercise_name, str) or not exercise_name.strip():
        raise ValueError("exercise_name must be a non-empty string")

    # Validate required fields in analysis_results
    required_fields = ["score", "level", "individual_scores", "feedback"]
    for field in required_fields:
        if field not in analysis_results:
            raise ValueError(f"Missing required field '{field}' in analysis_results")

    report = {
        "exercise": exercise_name,
        "overall_score": round(analysis_results["score"], 1),
        "individual_scores": {
            k: round(v, 1) for k, v in analysis_results["individual_scores"].items()
        },
        "level": analysis_results["level"],
        "improvement_areas": [],
        "strengths": [],
        "detailed_feedback": analysis_results["feedback"],
        "metrics": analysis_results["metrics"],
        "recommendations": generate_recommendations(
            analysis_results["feedback"], analysis_results["score"]
        ),
        "sensitivity_factors": analysis_results.get("sensitivity_factors", {}),
        "analysis_version": "unified_config_manager_v1.0",
    }

    # Identify improvement areas and strengths
    for category, message in analysis_results["feedback"].items():
        if "Excellent" in message or "Good" in message or "Great" in message:
            report["strengths"].append(message)
        else:
            report["improvement_areas"].append(message)

    # Save report if path specified
    if output_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            logger.info(f"Unified analysis report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise ValueError(f"Failed to save report to {output_path}: {e}")

    return report


# =============================================================================
# ORIGINAL SPECIFIC FUNCTIONS FOR MILITARY PRESS (maintain compatibility)
# =============================================================================


def analyze_movement_amplitude(user_data, expert_data, exercise_config):
    """WRAPPER: Maintains compatibility with original military press code."""
    landmarks_config = {
        "left_landmark": "landmark_left_elbow",
        "right_landmark": "landmark_right_elbow",
        "axis": "y",
        "movement_direction": "up",
        "feedback_context": "elbows",
    }
    return analyze_movement_amplitude_universal(
        user_data, expert_data, exercise_config, landmarks_config
    )


def analyze_elbow_abduction_angle(user_data, expert_data, exercise_config):
    """WRAPPER: Maintains compatibility with original military press code."""
    from src.feedback.specific_metrics import analyze_elbow_abduction_angle_press

    return analyze_elbow_abduction_angle_press(user_data, expert_data, exercise_config)


def analyze_symmetry(user_data, expert_data, exercise_config):
    """WRAPPER: Maintains compatibility with original military press code."""
    landmarks_config = {
        "left_landmark": "landmark_left_elbow",
        "right_landmark": "landmark_right_elbow",
        "axis": "y",
        "feedback_context": "elbow movement",
    }
    return analyze_symmetry_universal(
        user_data, expert_data, exercise_config, landmarks_config
    )


def analyze_movement_trajectory_3d(user_data, expert_data, exercise_config):
    """WRAPPER: Maintains compatibility with original military press code."""
    landmarks_config = {
        "left_landmark": "landmark_left_wrist",
        "right_landmark": "landmark_right_wrist",
    }
    return analyze_movement_trajectory_3d_universal(
        user_data, expert_data, exercise_config, landmarks_config
    )


def analyze_speed(user_data, expert_data, exercise_config):
    """WRAPPER: Maintains compatibility with original military press code."""
    landmarks_config = {
        "left_landmark": "landmark_left_elbow",
        "right_landmark": "landmark_right_elbow",
        "axis": "y",
        "movement_direction": "up",
    }
    return analyze_speed_universal(
        user_data, expert_data, exercise_config, landmarks_config
    )


def analyze_scapular_stability(user_data, expert_data, exercise_config):
    """WRAPPER: Maintains compatibility with original military press code."""
    from src.feedback.specific_metrics import analyze_scapular_stability_press

    return analyze_scapular_stability_press(user_data, expert_data, exercise_config)
