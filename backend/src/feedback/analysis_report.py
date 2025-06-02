# backend/src/feedback/analysis_report.py - VERSIÓN SIN DEFAULTS CON CONFIG_MANAGER
import sys
import numpy as np
import pandas as pd
import os
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# IMPORTAR FUNCIONES UNIVERSALES Y ESPECÍFICAS
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

# MANTENER IMPORTACIONES ORIGINALES
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
    user_data, expert_data, exercise_name="press_militar", config_path="config.json"
):
    """
    FUNCIÓN PRINCIPAL: Análisis completo usando config_manager estrictamente.
    """
    logger.info(f"Starting strict analysis for: {exercise_name}")

    # Cargar configuración usando config_manager - OBLIGATORIO
    try:
        exercise_config = config_manager.get_exercise_config(exercise_name, config_path)
        # Añadir exercise_name al config para las métricas universales
        exercise_config["_exercise_name"] = exercise_name
    except Exception as e:
        logger.error(
            f"Failed to load configuration for exercise '{exercise_name}': {e}"
        )
        raise ValueError(
            f"Configuration loading failed for exercise '{exercise_name}': {e}"
        )

    # Obtener configuración de landmarks por ejercicio
    landmarks_config = _get_landmarks_config_for_exercise(exercise_name)

    # Obtener funciones específicas para este ejercicio
    try:
        specific_functions = get_specific_metrics_for_exercise(exercise_name)
        specific_names = get_specific_metric_names_for_exercise(exercise_name)
    except ValueError as e:
        logger.error(f"Exercise '{exercise_name}' not supported: {e}")
        raise

    # =================================================================
    # EJECUTAR LAS 4 MÉTRICAS UNIVERSALES (PASANDO config_path)
    # =================================================================

    try:
        # 1. AMPLITUD (universal)
        amplitude_result = analyze_movement_amplitude_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["amplitude"],
            config_path,
        )

        # 2. SIMETRÍA (universal)
        symmetry_result = analyze_symmetry_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["symmetry"],
            config_path,
        )

        # 3. TRAYECTORIA (universal)
        trajectory_result = analyze_movement_trajectory_3d_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["trajectory"],
            config_path,
        )

        # 4. VELOCIDAD (universal)
        speed_result = analyze_speed_universal(
            user_data,
            expert_data,
            exercise_config,
            landmarks_config["speed"],
            config_path,
        )

    except Exception as e:
        logger.error(f"Error in universal metrics analysis: {e}")
        raise ValueError(f"Universal metrics analysis failed: {e}")

    # =================================================================
    # EJECUTAR LAS 2 MÉTRICAS ESPECÍFICAS (PASANDO config_path)
    # =================================================================

    try:
        # 5. MÉTRICA ESPECÍFICA A (abducción/profundidad/swing)
        specific_a_result = specific_functions["metrica_especifica_a"](
            user_data, expert_data, exercise_config, config_path
        )

        # 6. MÉTRICA ESPECÍFICA B (estabilidad/tracking/retracción)
        specific_b_result = specific_functions["metrica_especifica_b"](
            user_data, expert_data, exercise_config, config_path
        )

    except Exception as e:
        logger.error(f"Error in specific metrics analysis: {e}")
        raise ValueError(f"Specific metrics analysis failed: {e}")

    # =================================================================
    # COMBINAR RESULTADOS
    # =================================================================

    # Combinar métricas
    all_metrics = {
        "amplitud": amplitude_result["metrics"],
        specific_names["metrica_especifica_a"]: specific_a_result["metrics"],
        "simetria": symmetry_result["metrics"],
        "trayectoria": trajectory_result["metrics"],
        "velocidad": speed_result["metrics"],
        specific_names["metrica_especifica_b"]: specific_b_result["metrics"],
    }

    # Combinar feedback
    all_feedback = {
        **amplitude_result["feedback"],
        **specific_a_result["feedback"],
        **symmetry_result["feedback"],
        **trajectory_result["feedback"],
        **speed_result["feedback"],
        **specific_b_result["feedback"],
    }

    # Usar los nombres originales de scores para compatibilidad
    individual_scores = {}

    if exercise_name.lower() == "press_militar":
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "abduction_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "scapular_score": specific_b_result["score"],
        }
    elif exercise_name.lower() == "sentadilla":
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "depth_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "knee_score": specific_b_result["score"],
        }
    elif exercise_name.lower() == "dominada":
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "swing_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "retraction_score": specific_b_result["score"],
        }
    else:
        # Fallback para ejercicios no reconocidos
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "specific_a_score": specific_a_result["score"],
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "specific_b_score": specific_b_result["score"],
        }

    # VALIDAR SCORES
    for key, score in individual_scores.items():
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            logger.error(f"Invalid score detected: {key}={score}")
            raise ValueError(
                f"Invalid score for {key}: {score}. Must be between 0 and 100"
            )

    # =================================================================
    # CALCULAR SCORE GLOBAL USANDO CONFIG_MANAGER
    # =================================================================

    try:
        # Obtener pesos usando config_manager
        weights = config_manager.get_scoring_weights(exercise_name, config_path)

        # Verificar que todas las claves de weights existen en individual_scores
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

        # Normalizar weights para que sumen 1
        valid_weights = {k: v / total_weight for k, v in valid_weights.items()}

        # Calcular score global
        overall_score = sum(
            individual_scores[key] * valid_weights[key] for key in valid_weights.keys()
        )
        overall_score = max(0, min(100, overall_score))

    except Exception as e:
        logger.error(f"Error calculating overall score: {e}")
        raise ValueError(f"Failed to calculate overall score: {e}")

    # Determinar skill level usando configuración si está disponible
    try:
        # Intentar obtener skill levels de la configuración global
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

    # RETORNAR MISMO FORMATO QUE ANTES
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
    MANTENER FUNCIÓN ORIGINAL EXACTA - sin cambios.
    """
    if not isinstance(analysis_results, dict):
        raise ValueError("analysis_results must be a dictionary")

    if not isinstance(exercise_name, str) or not exercise_name.strip():
        raise ValueError("exercise_name must be a non-empty string")

    # Validar campos requeridos en analysis_results
    required_fields = ["score", "level", "individual_scores", "feedback"]
    for field in required_fields:
        if field not in analysis_results:
            raise ValueError(f"Missing required field '{field}' in analysis_results")

    report = {
        "ejercicio": exercise_name,
        "puntuacion_global": round(analysis_results["score"], 1),
        "puntuaciones_individuales": {
            k: round(v, 1) for k, v in analysis_results["individual_scores"].items()
        },
        "nivel": analysis_results["level"],
        "areas_mejora": [],
        "puntos_fuertes": [],
        "feedback_detallado": analysis_results["feedback"],
        "metricas": analysis_results["metrics"],
        "recomendaciones": generate_recommendations(
            analysis_results["feedback"], analysis_results["score"]
        ),
        "sensitivity_factors": analysis_results.get("sensitivity_factors", {}),
        "version_analisis": "strict_config_manager_v1.0",
    }

    # Identificar áreas de mejora y puntos fuertes
    for category, message in analysis_results["feedback"].items():
        if "Excelente" in message or "Buen" in message or "Buena" in message:
            report["puntos_fuertes"].append(message)
        else:
            report["areas_mejora"].append(message)

    # Guardar informe si se especificó una ruta
    if output_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            logger.info(f"Strict analysis report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise ValueError(f"Failed to save report to {output_path}: {e}")

    return report


# =============================================================================
# FUNCIONES AUXILIARES PARA CONFIGURACIÓN POR EJERCICIO
# =============================================================================


def _get_landmarks_config_for_exercise(exercise_name):
    """
    Configura landmarks para las 4 métricas universales según el ejercicio.

    Args:
        exercise_name: Nombre del ejercicio

    Returns:
        dict: Configuración de landmarks por métrica

    Raises:
        ValueError: Si el ejercicio no está soportado
    """
    exercise_name = exercise_name.lower().replace(" ", "_")

    if exercise_name == "press_militar":
        return {
            "amplitude": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "axis": "y",
                "movement_direction": "up",
                "feedback_context": "codos",
            },
            "symmetry": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "axis": "y",
                "feedback_context": "movimiento de codos",
            },
            "trajectory": {
                "left_landmark": "landmark_left_wrist",
                "right_landmark": "landmark_right_wrist",
            },
            "speed": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "axis": "y",
                "movement_direction": "up",
            },
        }

    elif exercise_name == "sentadilla":
        return {
            "amplitude": {
                "left_landmark": "landmark_left_hip",
                "right_landmark": "landmark_right_hip",
                "axis": "y",
                "movement_direction": "down",
                "feedback_context": "caderas",
            },
            "symmetry": {
                "left_landmark": "landmark_left_knee",
                "right_landmark": "landmark_right_knee",
                "axis": "y",
                "feedback_context": "movimiento de rodillas",
            },
            "trajectory": {
                "left_landmark": "landmark_left_hip",
                "right_landmark": "landmark_right_hip",
            },
            "speed": {
                "left_landmark": "landmark_left_hip",
                "right_landmark": "landmark_right_hip",
                "axis": "y",
                "movement_direction": "down",
            },
        }

    elif exercise_name == "dominada":
        return {
            "amplitude": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "axis": "y",
                "movement_direction": "up",
                "feedback_context": "codos",
            },
            "symmetry": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "axis": "y",
                "feedback_context": "movimiento de codos",
            },
            "trajectory": {
                "left_landmark": "landmark_left_wrist",
                "right_landmark": "landmark_right_wrist",
            },
            "speed": {
                "left_landmark": "landmark_left_elbow",
                "right_landmark": "landmark_right_elbow",
                "axis": "y",
                "movement_direction": "up",
            },
        }

    else:
        raise ValueError(
            f"Unsupported exercise for landmarks configuration: {exercise_name}"
        )


# =============================================================================
# FUNCIONES ORIGINALES ESPECÍFICAS PARA PRESS MILITAR (mantener compatibilidad)
# =============================================================================


def analyze_movement_amplitude(user_data, expert_data, exercise_config):
    """WRAPPER: Mantiene compatibilidad con código original del press militar."""
    landmarks_config = {
        "left_landmark": "landmark_left_elbow",
        "right_landmark": "landmark_right_elbow",
        "axis": "y",
        "movement_direction": "up",
        "feedback_context": "codos",
    }
    return analyze_movement_amplitude_universal(
        user_data, expert_data, exercise_config, landmarks_config
    )


def analyze_elbow_abduction_angle(user_data, expert_data, exercise_config):
    """WRAPPER: Mantiene compatibilidad con código original del press militar."""
    from src.feedback.specific_metrics import analyze_elbow_abduction_angle_press

    return analyze_elbow_abduction_angle_press(user_data, expert_data, exercise_config)


def analyze_symmetry(user_data, expert_data, exercise_config):
    """WRAPPER: Mantiene compatibilidad con código original del press militar."""
    landmarks_config = {
        "left_landmark": "landmark_left_elbow",
        "right_landmark": "landmark_right_elbow",
        "axis": "y",
        "feedback_context": "movimiento de codos",
    }
    return analyze_symmetry_universal(
        user_data, expert_data, exercise_config, landmarks_config
    )


def analyze_movement_trajectory_3d(user_data, expert_data, exercise_config):
    """WRAPPER: Mantiene compatibilidad con código original del press militar."""
    landmarks_config = {
        "left_landmark": "landmark_left_wrist",
        "right_landmark": "landmark_right_wrist",
    }
    return analyze_movement_trajectory_3d_universal(
        user_data, expert_data, exercise_config, landmarks_config
    )


def analyze_speed(user_data, expert_data, exercise_config):
    """WRAPPER: Mantiene compatibilidad con código original del press militar."""
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
    """WRAPPER: Mantiene compatibilidad con código original del press militar."""
    from src.feedback.specific_metrics import analyze_scapular_stability_press

    return analyze_scapular_stability_press(user_data, expert_data, exercise_config)
