# src/feedback/analysis_graphics.py - THREAD-SAFE VERSION
import sys
import numpy as np
import pandas as pd
import os
import logging
import json

# CRÍTICO: Configurar matplotlib ANTES que cualquier otro import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.matplotlib_config import (
    ThreadSafeMatplotlib,
    ensure_matplotlib_thread_safety,
)

# Ahora importar matplotlib
import matplotlib.pyplot as plt

# IMPORT ONLY WHAT'S NECESSARY
from src.utils.analysis_utils import (
    calculate_elbow_abduction_angle,
    generate_recommendations,
)

from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


def visualize_analysis_results(
    analysis_results,
    user_data,
    expert_data,
    exercise_name,
    output_dir=None,
    config_path="config.json",
):
    """
    Creates visualizations of analysis results using unified scores.
    THREAD-SAFE VERSION with proper matplotlib cleanup.
    """
    # Asegurar thread safety al inicio
    ensure_matplotlib_thread_safety()

    if not isinstance(analysis_results, dict):
        raise ValueError("analysis_results must be a dictionary")

    if not isinstance(exercise_name, str) or not exercise_name.strip():
        raise ValueError("exercise_name must be a non-empty string")

    if user_data is None or user_data.empty:
        raise ValueError("user_data cannot be None or empty")

    if expert_data is None or expert_data.empty:
        raise ValueError("expert_data cannot be None or empty")

    # Validate required fields
    required_fields = ["metrics", "individual_scores"]
    for field in required_fields:
        if field not in analysis_results:
            raise ValueError(f"Missing required field '{field}' in analysis_results")

    metrics = analysis_results["metrics"]
    individual_scores = analysis_results["individual_scores"]

    visualizations = []

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            raise ValueError(f"Cannot create output directory: {e}")

    # Get landmarks configuration
    try:
        if not os.path.isabs(config_path):
            base_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            full_config_path = os.path.join(base_dir, "src", "config", config_path)
        else:
            full_config_path = config_path

        if not os.path.exists(full_config_path):
            logger.warning(
                f"Config file not found at {full_config_path}, using fallback"
            )
            landmarks_config = _get_fallback_landmarks_config(exercise_name)
        else:
            landmarks_config = config_manager.get_exercise_landmarks_config(
                exercise_name, full_config_path
            )
    except Exception as e:
        logger.error(f"Failed to get landmarks configuration for {exercise_name}: {e}")
        landmarks_config = _get_fallback_landmarks_config(exercise_name)

    # Create all visualizations with proper cleanup
    visualization_functions = [
        (
            "amplitude_chart",
            _create_amplitude_chart,
            (
                metrics,
                user_data,
                expert_data,
                exercise_name,
                output_dir,
                landmarks_config,
            ),
        ),
        (
            "abduction_chart",
            _create_abduction_chart,
            (user_data, expert_data, exercise_name, output_dir),
        ),
        (
            "trajectory_chart",
            _create_trajectory_chart,
            (user_data, expert_data, exercise_name, output_dir, landmarks_config),
        ),
        (
            "symmetry_chart",
            _create_symmetry_chart,
            (
                user_data,
                analysis_results,
                exercise_name,
                output_dir,
                full_config_path,
                landmarks_config,
            ),
        ),
        (
            "velocity_chart",
            _create_velocity_chart,
            (user_data, expert_data, exercise_name, output_dir, landmarks_config),
        ),
        (
            "scores_chart",
            _create_scores_chart,
            (analysis_results, exercise_name, output_dir),
        ),
        (
            "radar_chart",
            _create_radar_chart,
            (analysis_results, exercise_name, output_dir),
        ),
        (
            "summary_chart",
            _create_summary_chart,
            (analysis_results, exercise_name, output_dir),
        ),
    ]

    for viz_name, viz_func, viz_args in visualization_functions:
        try:
            with ThreadSafeMatplotlib():  # Context manager para thread safety
                viz_path = viz_func(*viz_args)
                if viz_path:
                    visualizations.append(viz_path)
                    logger.debug(f"✅ {viz_name} created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating {viz_name}: {e}")

    return visualizations


def _create_amplitude_chart(
    metrics, user_data, expert_data, exercise_name, output_dir, landmarks_config
):
    """Creates amplitude chart using landmarks from configuration."""
    plt.figure(figsize=(10, 6))

    # Get landmarks from configuration
    amplitude_config = landmarks_config.get("amplitude", {})
    landmark_left = amplitude_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = amplitude_config.get("right_landmark", "landmark_right_elbow")
    landmark_name = amplitude_config.get("feedback_context", "Joints")

    # Use configured landmarks
    user_signal = (
        user_data[f"{landmark_left}_y"] + user_data[f"{landmark_right}_y"]
    ) / 2
    expert_signal = (
        expert_data[f"{landmark_left}_y"] + expert_data[f"{landmark_right}_y"]
    ) / 2

    plt.plot(user_signal, label="User", color="blue")
    plt.plot(expert_signal, label="Expert", color="red")

    # Get amplitude metrics
    amplitude_metrics = metrics.get("amplitude", {})
    if amplitude_metrics:
        # Reference lines
        plt.axhline(
            amplitude_metrics.get("user_highest_point", 0),
            linestyle="--",
            color="blue",
            alpha=0.7,
        )
        plt.axhline(
            amplitude_metrics.get("user_lowest_point", 0),
            linestyle="--",
            color="blue",
            alpha=0.7,
        )
        plt.axhline(
            amplitude_metrics.get("expert_highest_point", 0),
            linestyle="--",
            color="red",
            alpha=0.7,
        )
        plt.axhline(
            amplitude_metrics.get("expert_lowest_point", 0),
            linestyle="--",
            color="red",
            alpha=0.7,
        )

    plt.title(f"Range of Motion ({landmark_name}) - {exercise_name}")
    plt.xlabel("Frame")
    plt.ylabel("Y Coordinate (MediaPipe)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "range_of_motion.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_abduction_chart(user_data, expert_data, exercise_name, output_dir):
    """Creates abduction chart (only for relevant exercises)."""

    # Only create for exercises that use elbows
    exercise_name_clean = exercise_name.lower().replace(" ", "_")
    if exercise_name_clean not in ["military_press", "pull_up"]:
        logger.info(f"Skipping abduction chart for exercise {exercise_name_clean}")
        return None

    plt.figure(figsize=(10, 6))

    user_abduction_angles = []
    expert_abduction_angles = []

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

            # === EXPERT - BOTH ELBOWS ===
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

            # Check for NaN
            if (
                not np.isnan(user_right_shoulder).any()
                and not np.isnan(user_right_elbow).any()
                and not np.isnan(user_left_shoulder).any()
                and not np.isnan(user_left_elbow).any()
                and not np.isnan(expert_right_shoulder).any()
                and not np.isnan(expert_right_elbow).any()
                and not np.isnan(expert_left_shoulder).any()
                and not np.isnan(expert_left_elbow).any()
            ):

                # Use lateral abduction function
                user_right_angle = calculate_elbow_abduction_angle(
                    user_right_shoulder, user_right_elbow
                )
                user_left_angle = calculate_elbow_abduction_angle(
                    user_left_shoulder, user_left_elbow
                )
                expert_right_angle = calculate_elbow_abduction_angle(
                    expert_right_shoulder, expert_right_elbow
                )
                expert_left_angle = calculate_elbow_abduction_angle(
                    expert_left_shoulder, expert_left_elbow
                )

                # Average of both elbows
                user_avg_angle = (user_right_angle + user_left_angle) / 2
                expert_avg_angle = (expert_right_angle + expert_left_angle) / 2

                user_abduction_angles.append(user_avg_angle)
                expert_abduction_angles.append(expert_avg_angle)

        except Exception as e:
            logger.warning(f"Error calculating abduction angles at frame {i}: {e}")
            pass

    # Plot signals if we have data
    if user_abduction_angles and expert_abduction_angles:
        plt.plot(user_abduction_angles, label="User", color="blue", linewidth=2)
        plt.plot(expert_abduction_angles, label="Expert", color="red", linewidth=2)

        plt.title(f"Lateral Elbow Abduction - {exercise_name}")
        plt.xlabel("Frame")
        plt.ylabel("Lateral Abduction Angle (degrees)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add reference lines for interpretation
        plt.axhline(
            y=30, color="green", linestyle="--", alpha=0.5, label="Very Open (30°)"
        )
        plt.axhline(
            y=60, color="orange", linestyle="--", alpha=0.5, label="Moderate (60°)"
        )
        plt.axhline(
            y=80, color="red", linestyle="--", alpha=0.5, label="Very Closed (80°)"
        )

        # Update legend
        plt.legend()

        # Show statistics on chart
        user_avg = np.mean(user_abduction_angles)
        expert_avg = np.mean(expert_abduction_angles)
        user_min = np.min(user_abduction_angles)
        expert_min = np.min(expert_abduction_angles)

        plt.text(
            0.02,
            0.98,
            f"User - Average: {user_avg:.1f}°, Minimum: {user_min:.1f}°\n"
            f"Expert - Average: {expert_avg:.1f}°, Minimum: {expert_min:.1f}°\n"
            f"Average Difference: {user_avg - expert_avg:.1f}°",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        if output_dir:
            path = os.path.join(output_dir, "lateral_elbow_abduction.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return path
    else:
        logger.warning("Could not calculate abduction angles for chart")

    plt.close()
    return None


def _create_trajectory_chart(
    user_data, expert_data, exercise_name, output_dir, landmarks_config
):
    """Creates trajectory chart using appropriate landmarks."""
    plt.figure(figsize=(10, 8))

    # Get landmarks from configuration
    trajectory_config = landmarks_config.get("trajectory", {})
    landmark_left = trajectory_config.get("left_landmark", "landmark_left_wrist")
    landmark_right = trajectory_config.get("right_landmark", "landmark_right_wrist")

    # Determine landmark name for display
    if "wrist" in landmark_left:
        landmark_name = "Right Wrist"
    elif "hip" in landmark_left:
        landmark_name = "Right Hip"
    elif "elbow" in landmark_left:
        landmark_name = "Right Elbow"
    else:
        landmark_name = "Right Joint"

    user_x = user_data[f"{landmark_right}_x"].values
    user_y = user_data[f"{landmark_right}_y"].values
    expert_x = expert_data[f"{landmark_right}_x"].values
    expert_y = expert_data[f"{landmark_right}_y"].values

    plt.scatter(user_x, user_y, s=10, alpha=0.7, color="blue", label="User")
    plt.plot(user_x, user_y, color="blue", alpha=0.4)

    plt.scatter(expert_x, expert_y, s=10, alpha=0.7, color="red", label="Expert")
    plt.plot(expert_x, expert_y, color="red", alpha=0.4)

    plt.title(f"{landmark_name} Trajectory - {exercise_name}")
    plt.xlabel("X (lateral)")
    plt.ylabel("Y (vertical)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "frontal_trajectory.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_symmetry_chart(
    user_data,
    analysis_results,
    exercise_name,
    output_dir,
    config_path,
    landmarks_config,
):
    """Creates bilateral symmetry chart using appropriate landmarks."""
    plt.figure(figsize=(10, 6))

    # Get landmarks from configuration
    symmetry_config = landmarks_config.get("symmetry", {})
    landmark_left = symmetry_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = symmetry_config.get("right_landmark", "landmark_right_elbow")

    # Determine landmark name for display
    if "elbow" in landmark_left:
        landmark_name = "elbows"
    elif "knee" in landmark_left:
        landmark_name = "knees"
    elif "hip" in landmark_left:
        landmark_name = "hips"
    else:
        landmark_name = "joints"

    diff_y = abs(
        user_data[f"{landmark_right}_y"].values - user_data[f"{landmark_left}_y"].values
    )

    plt.plot(diff_y, label=f"Difference between {landmark_name}", color="purple")

    # Get threshold using config_manager with better error handling
    try:
        symmetry_threshold = config_manager.get_analysis_threshold(
            "symmetry_threshold", exercise_name, config_path
        )

        plt.axhline(
            y=symmetry_threshold,
            linestyle="--",
            color="red",
            label=f"Asymmetry threshold ({symmetry_threshold})",
        )
        logger.debug(f"Successfully got symmetry threshold: {symmetry_threshold}")
    except Exception as e:
        logger.warning(f"Could not get symmetry threshold from config: {e}")
        # Use fallback threshold
        symmetry_threshold = 0.15
        plt.axhline(
            y=symmetry_threshold,
            linestyle="--",
            color="red",
            label=f"Asymmetry threshold (fallback: {symmetry_threshold})",
        )

    plt.title(f"Bilateral Symmetry ({landmark_name.title()}) - {exercise_name}")
    plt.xlabel("Frame")
    plt.ylabel("Height Difference (absolute value)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "bilateral_symmetry.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_velocity_chart(
    user_data, expert_data, exercise_name, output_dir, landmarks_config
):
    """Creates velocity chart using appropriate landmarks."""
    plt.figure(figsize=(10, 6))

    # Get landmarks from configuration
    speed_config = landmarks_config.get("speed", {})
    landmark_left = speed_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = speed_config.get("right_landmark", "landmark_right_elbow")

    # Determine landmark name for display
    if "elbow" in landmark_left:
        landmark_name = "Elbows"
    elif "hip" in landmark_left:
        landmark_name = "Hips"
    elif "wrist" in landmark_left:
        landmark_name = "Wrists"
    else:
        landmark_name = "Joints"

    user_signal = (
        user_data[f"{landmark_right}_y"] + user_data[f"{landmark_left}_y"]
    ) / 2
    expert_signal = (
        expert_data[f"{landmark_right}_y"] + expert_data[f"{landmark_left}_y"]
    ) / 2

    user_velocity = np.gradient(user_signal.values)
    expert_velocity = np.gradient(expert_signal.values)

    plt.plot(user_velocity, label="User", color="blue")
    plt.plot(expert_velocity, label="Expert", color="red")

    plt.title(f"Vertical Velocity ({landmark_name}) - {exercise_name}")
    plt.xlabel("Frame")
    plt.ylabel("Velocity (units/frame)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "velocity.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_radar_chart(analysis_results, exercise_name, output_dir):
    """
    Creates radar chart USING DYNAMIC SCORES according to exercise.
    """
    try:
        plt.figure(figsize=(10, 8))

        individual_scores = analysis_results.get("individual_scores", {})

        if not individual_scores:
            logger.error("No individual scores found in analysis_results")
            return None

        # Dynamic categories and scores according to exercise
        exercise_name_clean = exercise_name.lower().replace(" ", "_")

        if exercise_name_clean == "military_press":
            categories_radar = [
                "Amplitude",
                "Elbow\nAbduction",
                "Symmetry",
                "Trajectory",
                "Speed",
                "Scapular\nStability",
            ]
            score_keys = [
                "rom_score",
                "abduction_score",
                "sym_score",
                "path_score",
                "speed_score",
                "scapular_score",
            ]
        elif exercise_name_clean == "squat":
            categories_radar = [
                "Amplitude",
                "Depth",
                "Symmetry",
                "Trajectory",
                "Speed",
                "Knee\nTracking",
            ]
            score_keys = [
                "rom_score",
                "depth_score",
                "sym_score",
                "path_score",
                "speed_score",
                "knee_score",
            ]
        elif exercise_name_clean == "pull_up":
            categories_radar = [
                "Amplitude",
                "Swing\nControl",
                "Symmetry",
                "Trajectory",
                "Speed",
                "Scapular\nRetraction",
            ]
            score_keys = [
                "rom_score",
                "swing_score",
                "sym_score",
                "path_score",
                "speed_score",
                "retraction_score",
            ]
        else:
            # Fallback for unknown exercises
            score_keys = list(individual_scores.keys())
            categories_radar = [
                key.replace("_score", "").replace("_", " ").title()
                for key in score_keys
            ]

        # Extract scores dynamically
        scores_normalized = []
        scores_raw = []

        for score_key in score_keys:
            if score_key not in individual_scores:
                logger.error(f"Score key '{score_key}' not found in individual_scores")
                return None

            score_value = individual_scores[score_key]
            scores_normalized.append(score_value / 100)
            scores_raw.append(score_value)

        # DEBUG: Print values for verification
        logger.info("=== DEBUG RADAR CHART UNIFIED ===")
        for i, (cat, score_norm, score_raw) in enumerate(
            zip(categories_radar, scores_normalized, scores_raw)
        ):
            logger.info(
                f"{i}: {cat.replace(chr(10), ' ')} = {score_raw:.1f} ({score_norm:.3f})"
            )

        # Calculate angles BEFORE closing polygon
        angles = np.linspace(
            0, 2 * np.pi, len(categories_radar), endpoint=False
        ).tolist()

        # Close polygon by duplicating first element
        scores_normalized = np.concatenate((scores_normalized, [scores_normalized[0]]))
        angles = angles + [angles[0]]

        # Create radar chart
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, scores_normalized, color="blue", alpha=0.25)
        ax.plot(angles, scores_normalized, color="blue", linewidth=2)

        # Use original angles (without duplicate) for labels
        original_angles = np.linspace(
            0, 2 * np.pi, len(categories_radar), endpoint=False
        ).tolist()
        ax.set_xticks(original_angles)
        ax.set_xticklabels(categories_radar)

        # Configure radial scale
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25", "50", "75", "100"])

        plt.title(f"Technique Analysis - {exercise_name}", size=15, y=1.1)

        if output_dir:
            path = os.path.join(output_dir, "radar_analysis.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return path

        plt.close()
        return None

    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        return None


def generate_radar_data(analysis_results, exercise_name, output_dir):
    """
    Genera datos JSON para el radar chart del frontend.
    Reemplaza la generación de imagen PNG.
    """
    try:
        individual_scores = analysis_results.get("individual_scores", {})

        if not individual_scores:
            logger.error("No individual scores found in analysis_results")
            return None

        # Misma lógica que tenías en _create_radar_chart para categorías dinámicas
        exercise_name_clean = exercise_name.lower().replace(" ", "_")

        if exercise_name_clean == "military_press":
            categories_data = [
                {"metric": "Amplitud", "score": individual_scores.get("rom_score", 0)},
                {
                    "metric": "Abducción Codo",
                    "score": individual_scores.get("abduction_score", 0),
                },
                {"metric": "Simetría", "score": individual_scores.get("sym_score", 0)},
                {
                    "metric": "Trayectoria",
                    "score": individual_scores.get("path_score", 0),
                },
                {
                    "metric": "Velocidad",
                    "score": individual_scores.get("speed_score", 0),
                },
                {
                    "metric": "Estabilidad",
                    "score": individual_scores.get("scapular_score", 0),
                },
            ]
        elif exercise_name_clean == "squat":
            categories_data = [
                {"metric": "Amplitud", "score": individual_scores.get("rom_score", 0)},
                {
                    "metric": "Profundidad",
                    "score": individual_scores.get("depth_score", 0),
                },
                {"metric": "Simetría", "score": individual_scores.get("sym_score", 0)},
                {
                    "metric": "Trayectoria",
                    "score": individual_scores.get("path_score", 0),
                },
                {
                    "metric": "Velocidad",
                    "score": individual_scores.get("speed_score", 0),
                },
                {
                    "metric": "Seguimiento",
                    "score": individual_scores.get("knee_score", 0),
                },
            ]
        elif exercise_name_clean == "pull_up":
            categories_data = [
                {"metric": "Amplitud", "score": individual_scores.get("rom_score", 0)},
                {
                    "metric": "Control Balanceo",
                    "score": individual_scores.get("swing_score", 0),
                },
                {"metric": "Simetría", "score": individual_scores.get("sym_score", 0)},
                {
                    "metric": "Trayectoria",
                    "score": individual_scores.get("path_score", 0),
                },
                {
                    "metric": "Velocidad",
                    "score": individual_scores.get("speed_score", 0),
                },
                {
                    "metric": "Retracción",
                    "score": individual_scores.get("retraction_score", 0),
                },
            ]
        else:
            # Fallback para ejercicios desconocidos
            score_keys = list(individual_scores.keys())
            categories_data = []
            for score_key in score_keys:
                metric_name = score_key.replace("_score", "").replace("_", " ").title()
                categories_data.append(
                    {"metric": metric_name, "score": individual_scores[score_key]}
                )

        # DEBUG: Log para verificación
        logger.info("=== DEBUG RADAR DATA GENERATION ===")
        for item in categories_data:
            logger.info(f"{item['metric']}: {item['score']:.1f}")

        # Guardar como JSON
        if output_dir:
            radar_data_path = os.path.join(
                output_dir, f"{exercise_name}_radar_data.json"
            )
            with open(radar_data_path, "w", encoding="utf-8") as f:
                json.dump(categories_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Radar data JSON generated: {radar_data_path}")
            return radar_data_path

        return None

    except Exception as e:
        logger.error(f"Error generating radar data: {e}")
        return None


def _create_scores_chart(analysis_results, exercise_name, output_dir):
    """
    Creates scores by category chart USING DYNAMIC SCORES.
    """
    plt.figure(figsize=(10, 6))

    individual_scores = analysis_results.get("individual_scores", {})

    if not individual_scores:
        logger.error("No individual scores found in analysis_results")
        return None

    # Dynamic categories according to exercise
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean == "military_press":
        categories = [
            "Amplitude",
            "Elbow\nAbduction",
            "Symmetry",
            "Trajectory",
            "Speed",
            "Scapular\nStability",
            "Overall",
        ]
        score_keys = [
            "rom_score",
            "abduction_score",
            "sym_score",
            "path_score",
            "speed_score",
            "scapular_score",
        ]
    elif exercise_name_clean == "squat":
        categories = [
            "Amplitude",
            "Depth",
            "Symmetry",
            "Trajectory",
            "Speed",
            "Knee\nTracking",
            "Overall",
        ]
        score_keys = [
            "rom_score",
            "depth_score",
            "sym_score",
            "path_score",
            "speed_score",
            "knee_score",
        ]
    elif exercise_name_clean == "pull_up":
        categories = [
            "Amplitude",
            "Swing\nControl",
            "Symmetry",
            "Trajectory",
            "Speed",
            "Scapular\nRetraction",
            "Overall",
        ]
        score_keys = [
            "rom_score",
            "swing_score",
            "sym_score",
            "path_score",
            "speed_score",
            "retraction_score",
        ]
    else:
        # Fallback for unknown exercises
        score_keys = list(individual_scores.keys())
        categories = [
            key.replace("_score", "").replace("_", " ").title() for key in score_keys
        ]
        categories.append("Overall")

    # Extract scores dynamically
    scores_list = []
    for score_key in score_keys:
        if score_key not in individual_scores:
            logger.error(f"Score key '{score_key}' not found in individual_scores")
            return None
        scores_list.append(individual_scores[score_key])

    # Add overall score at end
    scores_list.append(analysis_results["score"])

    # DEBUG: Print values for verification
    logger.info("=== DEBUG SCORES CHART UNIFIED ===")
    for cat, score in zip(categories, scores_list):
        logger.info(f"{cat.replace(chr(10), ' ')}: {score:.1f}")

    # Colors according to score
    colors = []
    for score in scores_list:
        if score >= 90:
            colors.append("#27ae60")
        elif score >= 70:
            colors.append("#2ecc71")
        elif score >= 50:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    bars = plt.bar(categories, scores_list, color=colors)

    # Labels with values
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.axhline(y=50, color="r", linestyle="-", alpha=0.3, label="Poor")
    plt.axhline(y=70, color="y", linestyle="-", alpha=0.3, label="Acceptable")
    plt.axhline(y=90, color="g", linestyle="-", alpha=0.3, label="Excellent")

    plt.title(f"Score by Category - {exercise_name}")
    plt.ylabel("Score (0-100)")
    plt.ylim(0, 105)
    plt.legend(loc="lower right")

    if output_dir:
        path = os.path.join(output_dir, "category_scores.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_summary_chart(analysis_results, exercise_name, output_dir):
    """Creates visual text summary."""
    try:
        plt.figure(figsize=(12, 8))

        plt.text(
            0.5,
            0.95,
            f"TECHNIQUE ANALYSIS - {exercise_name.upper()}",
            fontsize=18,
            ha="center",
            va="top",
            fontweight="bold",
        )

        plt.text(
            0.5,
            0.88,
            f"Overall Score: {analysis_results['score']:.1f}/100 - Level: {analysis_results['level']}",
            fontsize=16,
            ha="center",
            va="top",
        )

        # Improvement areas
        improvement_areas = [
            msg
            for msg in analysis_results["feedback"].values()
            if "Good" not in msg and "Great" not in msg and "Excellent" not in msg
        ]

        plt.text(0.05, 0.8, "IMPROVEMENT AREAS:", fontsize=14, fontweight="bold")
        y_pos = 0.75
        for i, area in enumerate(improvement_areas[:5]):
            plt.text(0.07, y_pos - i * 0.05, f"• {area}", fontsize=12)

        # Recommendations
        plt.text(0.05, 0.5, "RECOMMENDATIONS:", fontsize=14, fontweight="bold")
        y_pos = 0.45
        recommendations = generate_recommendations(
            analysis_results["feedback"], analysis_results["score"]
        )
        for i, rec in enumerate(recommendations[:5]):
            plt.text(0.07, y_pos - i * 0.05, f"• {rec}", fontsize=12)

        # Strengths
        strengths = [
            msg
            for msg in analysis_results["feedback"].values()
            if "Good" in msg or "Great" in msg or "Excellent" in msg
        ]

        plt.text(0.05, 0.2, "STRENGTHS:", fontsize=14, fontweight="bold")
        y_pos = 0.15
        for i, strength in enumerate(strengths[:3]):
            plt.text(0.07, y_pos - i * 0.05, f"• {strength}", fontsize=12)

        plt.axis("off")

        if output_dir:
            path = os.path.join(output_dir, "visual_summary.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return path

        plt.close()
        return None

    except Exception as e:
        logger.error(f"Error creating summary chart: {e}")
        return None


def _get_fallback_landmarks_config(exercise_name):
    """Fallback landmarks configuration if config_manager fails."""
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean == "military_press":
        return {
            "amplitude": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "feedback_context": "Elbows",
            },
            "symmetry": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
            },
            "trajectory": {
                "left_landmark": "landmark_left_wrist",
                "right_landmark": "landmark_right_wrist",
            },
            "speed": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
            },
        }
    elif exercise_name_clean == "squat":
        return {
            "amplitude": {
                "left_landmark": "landmark_left_hip",
                "right_landmark": "landmark_right_hip",
                "feedback_context": "Hips",
            },
            "symmetry": {
                "left_landmark": "landmark_left_knee",
                "right_landmark": "landmark_right_knee",
            },
            "trajectory": {
                "left_landmark": "landmark_left_hip",
                "right_landmark": "landmark_right_hip",
            },
            "speed": {
                "left_landmark": "landmark_left_hip",
                "right_landmark": "landmark_right_hip",
            },
        }
    elif exercise_name_clean == "pull_up":
        return {
            "amplitude": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "feedback_context": "Elbows",
            },
            "symmetry": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
            },
            "trajectory": {
                "left_landmark": "landmark_left_wrist",
                "right_landmark": "landmark_right_wrist",
            },
            "speed": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
            },
        }
    else:
        # Default fallback
        return {
            "amplitude": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "feedback_context": "Elbows",
            },
            "symmetry": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
            },
            "trajectory": {
                "left_landmark": "landmark_left_wrist",
                "right_landmark": "landmark_right_wrist",
            },
            "speed": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
            },
        }
