# backend/src/feedback/analysis_report.py - VERSIÓN ACTUALIZADA CON SISTEMA MODULAR
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
    get_exercise_config,
    determine_skill_level,
    generate_recommendations,
    apply_unified_sensitivity,
    calculate_deviation_score,
    calculate_elbow_abduction_angle,
    apply_sensitivity_to_threshold,
)

logger = logging.getLogger(__name__)


def run_exercise_analysis(
    user_data, expert_data, exercise_name="press_militar", config_path="config.json"
):
    """
    FUNCIÓN PRINCIPAL: Mantiene exactamente la misma interfaz que antes.
    Ahora usa el sistema modular internamente pero la API es idéntica.
    """
    logger.info(f"Iniciando análisis modular para: {exercise_name}")

    # Cargar configuración (IGUAL QUE ANTES)
    exercise_config = get_exercise_config(exercise_name, config_path)

    # Obtener configuración de landmarks por ejercicio
    landmarks_config = _get_landmarks_config_for_exercise(exercise_name)

    # Obtener funciones específicas para este ejercicio
    specific_functions = get_specific_metrics_for_exercise(exercise_name)
    specific_names = get_specific_metric_names_for_exercise(exercise_name)

    # =================================================================
    # EJECUTAR LAS 4 MÉTRICAS UNIVERSALES
    # =================================================================

    # 1. AMPLITUD (universal)
    amplitude_result = analyze_movement_amplitude_universal(
        user_data, expert_data, exercise_config, landmarks_config["amplitude"]
    )

    # 2. SIMETRÍA (universal)
    symmetry_result = analyze_symmetry_universal(
        user_data, expert_data, exercise_config, landmarks_config["symmetry"]
    )

    # 3. TRAYECTORIA (universal)
    trajectory_result = analyze_movement_trajectory_3d_universal(
        user_data, expert_data, exercise_config, landmarks_config["trajectory"]
    )

    # 4. VELOCIDAD (universal)
    speed_result = analyze_speed_universal(
        user_data, expert_data, exercise_config, landmarks_config["speed"]
    )

    # =================================================================
    # EJECUTAR LAS 2 MÉTRICAS ESPECÍFICAS
    # =================================================================

    # 5. MÉTRICA ESPECÍFICA A (abducción/profundidad/swing)
    specific_a_result = specific_functions["metrica_especifica_a"](
        user_data, expert_data, exercise_config
    )

    # 6. MÉTRICA ESPECÍFICA B (estabilidad/tracking/retracción)
    specific_b_result = specific_functions["metrica_especifica_b"](
        user_data, expert_data, exercise_config
    )

    # =================================================================
    # COMBINAR RESULTADOS - CORREGIDO: Usar nombres originales
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

    # CORREGIDO: Usar los nombres originales de scores para compatibilidad
    individual_scores = {}

    if exercise_name.lower() == "press_militar":
        individual_scores = {
            "rom_score": amplitude_result["score"],
            "abduction_score": specific_a_result[
                "score"
            ],  # ← CORREGIDO: Usar nombre original
            "sym_score": symmetry_result["score"],
            "path_score": trajectory_result["score"],
            "speed_score": speed_result["score"],
            "scapular_score": specific_b_result[
                "score"
            ],  # ← CORREGIDO: Usar nombre original
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

    # VALIDAR SCORES (IGUAL QUE ANTES)
    for key, score in individual_scores.items():
        if score < 0 or score > 100:
            logger.warning(
                f"Score fuera de rango detectado: {key}={score:.1f}. Corrigiendo..."
            )
            individual_scores[key] = max(0, min(100, score))

    # CALCULAR SCORE GLOBAL - CORREGIDO: Sin necesidad de adaptar nombres
    weights = _get_weights_for_exercise(exercise_name, exercise_config)

    # Verificar que todas las claves de weights existen en individual_scores
    valid_weights = {}
    total_weight = 0

    for weight_key, weight_value in weights.items():
        if weight_key in individual_scores:
            valid_weights[weight_key] = weight_value
            total_weight += weight_value
        else:
            logger.warning(
                f"Score {weight_key} no encontrado en individual_scores. Disponibles: {list(individual_scores.keys())}"
            )

    # Normalizar weights para que sumen 1
    if total_weight > 0:
        valid_weights = {k: v / total_weight for k, v in valid_weights.items()}
    else:
        logger.error("No se encontraron weights válidos. Usando pesos uniformes.")
        valid_weights = {
            k: 1.0 / len(individual_scores) for k in individual_scores.keys()
        }

    # Calcular score global
    overall_score = sum(
        individual_scores[key] * valid_weights[key] for key in valid_weights.keys()
    )
    overall_score = max(0, min(100, overall_score))

    skill_level = determine_skill_level(overall_score, exercise_config)

    logger.info(
        f"Análisis modular completado - Puntuación: {overall_score:.1f}/100 - Nivel: {skill_level}"
    )
    logger.info(
        f"Scores individuales: {[f'{k}={v:.1f}' for k,v in individual_scores.items()]}"
    )

    # RETORNAR MISMO FORMATO QUE ANTES
    return {
        "metrics": all_metrics,
        "feedback": all_feedback,
        "score": overall_score,
        "level": skill_level,
        "individual_scores": individual_scores,
        "exercise_config": exercise_config,
        "sensitivity_factors": exercise_config.get("sensitivity_factors", {}),
    }


def generate_analysis_report(analysis_results, exercise_name, output_path=None):
    """
    MANTENER FUNCIÓN ORIGINAL EXACTA - sin cambios.
    """
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
        "version_analisis": "modular_system_v1.0",
    }

    # Identificar áreas de mejora y puntos fuertes
    for category, message in analysis_results["feedback"].items():
        if "Excelente" in message or "Buen" in message or "Buena" in message:
            report["puntos_fuertes"].append(message)
        else:
            report["areas_mejora"].append(message)

    # Guardar informe si se especificó una ruta
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        logger.info(f"Informe modular guardado en: {output_path}")

    return report


# =============================================================================
# FUNCIONES AUXILIARES PARA CONFIGURACIÓN POR EJERCICIO
# =============================================================================


def _get_landmarks_config_for_exercise(exercise_name):
    """
    Configura landmarks para las 4 métricas universales según el ejercicio.
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
        raise ValueError(f"Ejercicio no soportado: {exercise_name}")


def _get_weights_for_exercise(exercise_name, exercise_config):
    """
    Obtiene los pesos para el cálculo del score global según el ejercicio.
    CORREGIDO: Prioriza weights específicos del ejercicio sobre los globales.
    """
    # PRIMERA PRIORIDAD: Weights específicos del ejercicio en su analysis_config
    if "scoring_weights" in exercise_config:
        weights = exercise_config["scoring_weights"]
        if weights:
            logger.info(f"Usando weights específicos para {exercise_name}: {weights}")
            return weights

    # SEGUNDA PRIORIDAD: Weights globales del config
    weights = exercise_config.get("scoring_weights", {})
    if weights:
        logger.info(f"Usando weights globales para {exercise_name}: {weights}")
        return weights

    # TERCERA PRIORIDAD: Pesos por defecto según ejercicio
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean == "press_militar":
        default_weights = {
            "rom_score": 0.20,
            "abduction_score": 0.20,
            "sym_score": 0.15,
            "path_score": 0.20,
            "speed_score": 0.15,
            "scapular_score": 0.10,
        }
    elif exercise_name_clean == "sentadilla":
        default_weights = {
            "rom_score": 0.15,
            "depth_score": 0.25,
            "sym_score": 0.15,
            "path_score": 0.15,
            "speed_score": 0.10,
            "knee_score": 0.20,
        }
    elif exercise_name_clean == "dominada":
        default_weights = {
            "rom_score": 0.25,
            "swing_score": 0.20,
            "sym_score": 0.15,
            "path_score": 0.15,
            "speed_score": 0.10,
            "retraction_score": 0.15,
        }
    else:
        # Pesos uniformes por defecto para ejercicios desconocidos
        default_weights = {
            "rom_score": 0.20,
            "specific_a_score": 0.20,
            "sym_score": 0.15,
            "path_score": 0.20,
            "speed_score": 0.15,
            "specific_b_score": 0.10,
        }

    logger.warning(
        f"Usando weights por defecto para {exercise_name}: {default_weights}"
    )
    return default_weights


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
    """MANTENER FUNCIÓN ORIGINAL EXACTA - COPIAR DEL CÓDIGO ACTUAL"""
    # TODO: AQUÍ DEBE IR LA FUNCIÓN COMPLETA ACTUAL DE analysis_report.py
    # Por ahora placeholder para mantener estructura
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
    """MANTENER FUNCIÓN ORIGINAL EXACTA - COPIAR DEL CÓDIGO ACTUAL"""
    # TODO: AQUÍ DEBE IR LA FUNCIÓN COMPLETA ACTUAL DE analysis_report.py
    # Por ahora placeholder para mantener estructura
    from src.feedback.specific_metrics import analyze_scapular_stability_press

    return analyze_scapular_stability_press(user_data, expert_data, exercise_config)
