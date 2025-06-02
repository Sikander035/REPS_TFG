# backend/src/feedback/specific_metrics.py - MÉTRICAS ESPECÍFICAS SIN DEFAULTS
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
    calculate_elbow_abduction_angle,  # Mantener función original
    apply_sensitivity_to_threshold,
)
from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


# =============================================================================
# PRESS MILITAR - MÉTRICAS ESPECÍFICAS (usando config_manager)
# =============================================================================


def analyze_elbow_abduction_angle_press(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    PRESS MILITAR: Análisis de abducción de codos usando config_manager.
    """
    exercise_name = "press_militar"

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "abduccion_codos", exercise_name, config_path
    )

    user_right_abduction = []
    user_left_abduction = []
    expert_right_abduction = []
    expert_left_abduction = []

    # Calcular ángulo de abducción para TODOS los frames
    logger.info("Calculando ángulos de abducción lateral (proyección XZ con eje X)")

    for i in range(len(user_data)):
        try:
            # === USUARIO - AMBOS CODOS ===
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

            # Verificar que no hay NaN en las coordenadas
            if (
                not np.isnan(user_right_shoulder).any()
                and not np.isnan(user_right_elbow).any()
                and not np.isnan(user_left_shoulder).any()
                and not np.isnan(user_left_elbow).any()
            ):
                # Calcular ángulos de abducción del usuario
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
            logger.warning(
                f"Error al calcular ángulo de abducción del usuario en frame {i}: {e}"
            )
            user_right_abduction.append(np.nan)
            user_left_abduction.append(np.nan)

    # Calcular ángulos del experto (similar al usuario)
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
                f"Error al calcular ángulo de abducción del experto en frame {i}: {e}"
            )
            expert_right_abduction.append(np.nan)
            expert_left_abduction.append(np.nan)

    if not user_right_abduction or not expert_right_abduction:
        return {
            "metrics": {},
            "feedback": {
                "elbow_abduction": "No se pudo analizar la abducción de los codos."
            },
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
            "feedback": {
                "elbow_abduction": "Datos insuficientes para analizar abducción."
            },
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
        logger.error(f"Error en detección de mínimos: {e}")
        user_min_abduction = np.mean(user_clean_signal)
        expert_min_abduction = np.mean(expert_clean_signal)
        abduction_diff = user_min_abduction - expert_min_abduction
        absolute_diff = abduction_diff
        user_valley_values = user_clean_signal
        expert_valley_values = expert_clean_signal

    # Calcular score base usando config_manager
    max_penalty = config_manager.get_penalty_config(
        exercise_name=exercise_name,
        metric_type="specific",
        metric_name="elbow_abduction",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        abs(abduction_diff), 0, max_penalty=max_penalty, metric_type="linear"
    )

    # Aplicar sensibilidad de manera unificada
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "abduccion_codos"
    )

    # Feedback basado en score final para consistencia CON KEY EN INGLÉS
    feedback = {}

    if final_score >= 80:
        feedback["elbow_abduction"] = "Excelente posición lateral de codos."
    elif final_score >= 60:
        if abduction_diff > 0:  # Usuario tiene ángulo mayor = más cerrado
            feedback["elbow_abduction"] = (
                "Tus codos están ligeramente más cerrados que el experto. "
                "Sepáralos un poco más del cuerpo."
            )
        else:  # Usuario tiene ángulo menor = más abierto
            feedback["elbow_abduction"] = (
                "Tus codos están ligeramente más abiertos que el experto. "
                "Acércalos un poco más al cuerpo."
            )
    elif final_score >= 40:
        # Casos moderados
        if abduction_diff > 0:
            feedback["elbow_abduction"] = (
                "Tus codos están moderadamente más cerrados que el experto. "
                "Sepáralos más del cuerpo para mejor mecánica."
            )
        else:
            feedback["elbow_abduction"] = (
                "Tus codos se abren moderadamente durante el ejercicio. "
                "Acércalos más al cuerpo para mayor estabilidad."
            )
    else:
        # Casos críticos
        if abduction_diff > 0:
            feedback["elbow_abduction"] = (
                "Tus codos están significativamente más cerrados que el experto. "
                "Es importante separarlos más del cuerpo para mejor mecánica."
            )
        else:
            feedback["elbow_abduction"] = (
                "Tus codos se abren excesivamente durante el ejercicio. "
                "Es crítico acercarlos más al cuerpo para mayor seguridad."
            )

    metrics = {
        "abduccion_lateral_minima_usuario": user_min_abduction,
        "abduccion_lateral_minima_experto": expert_min_abduction,
        "diferencia_abduccion": abduction_diff,
        "min_absoluto_usuario": user_absolute_min,
        "min_absoluto_experto": expert_absolute_min,
        "diferencia_absoluta": absolute_diff,
        "num_minimos_detectados": len(user_valley_values),
        "frames_totales_usuario": len(user_data),
        "frames_totales_experto": len(expert_data),
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_scapular_stability_press(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    PRESS MILITAR: Análisis de estabilidad escapular usando config_manager.
    """
    exercise_name = "press_militar"

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "estabilidad_escapular", exercise_name, config_path
    )

    try:
        # Calcular centro de hombros
        user_r_shoulder_y = user_data["landmark_right_shoulder_y"].values
        user_l_shoulder_y = user_data["landmark_left_shoulder_y"].values
        expert_r_shoulder_y = expert_data["landmark_right_shoulder_y"].values
        expert_l_shoulder_y = expert_data["landmark_left_shoulder_y"].values

        user_shoulder_center_y = (user_r_shoulder_y + user_l_shoulder_y) / 2
        expert_shoulder_center_y = (expert_r_shoulder_y + expert_l_shoulder_y) / 2

        # Analizar estabilidad como variabilidad del centro de hombros
        user_shoulder_movement = float(np.std(user_shoulder_center_y))
        expert_shoulder_movement = float(np.std(expert_shoulder_center_y))
        movement_ratio = (
            user_shoulder_movement / expert_shoulder_movement
            if expert_shoulder_movement > 0
            else 1.0
        )

        # Analizar simetría de hombros
        user_shoulder_asymmetry = float(np.std(user_r_shoulder_y - user_l_shoulder_y))
        expert_shoulder_asymmetry = float(
            np.std(expert_r_shoulder_y - expert_l_shoulder_y)
        )
        asymmetry_ratio = (
            user_shoulder_asymmetry / expert_shoulder_asymmetry
            if expert_shoulder_asymmetry > 0
            else 1.0
        )

        # Calcular score base usando el peor ratio
        worst_stability_ratio = max(movement_ratio, asymmetry_ratio)
        # Obtener penalty de configuración
        max_penalty = config_manager.get_penalty_config(
            exercise_name=exercise_name,
            metric_type="specific",
            metric_name="scapular_stability",
            config_path=config_path,
        )
        base_score = calculate_deviation_score(
            worst_stability_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
        )

        # Aplicar sensibilidad de manera unificada (SUAVIZADA para evitar colapsos)
        # Limitar el impacto de la sensibilidad para esta métrica específica
        capped_sensitivity = (
            min(sensitivity_factor, 2.0)
            if sensitivity_factor > 1.5
            else sensitivity_factor
        )
        final_score = apply_unified_sensitivity(
            base_score, capped_sensitivity, "estabilidad_escapular"
        )

        # Obtener umbral usando config_manager
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
                    "Tus hombros se mueven excesivamente durante el press. "
                    "Es crítico mantener una posición mucho más estable de la cintura escapular."
                )
            else:
                feedback["scapular_stability"] = (
                    "Tus hombros se mueven excesivamente durante el press. "
                    "Mantén una posición más estable de la cintura escapular."
                )
        elif asymmetry_ratio > (2.0 / sensitivity_factor):
            feedback["scapular_stability"] = (
                "Se detecta asimetría en el movimiento de tus hombros. "
                "Concéntrate en mantener ambos hombros equilibrados."
            )
        elif movement_ratio > stability_threshold_adj:
            if sensitivity_factor > 1.5:
                feedback["scapular_stability"] = (
                    "Se detecta inestabilidad notable en tu cintura escapular. "
                    "Es importante practicar mantener los hombros en una posición más fija."
                )
            else:
                feedback["scapular_stability"] = (
                    "Se detecta cierta inestabilidad en tu cintura escapular. "
                    "Practica mantener los hombros en una posición más fija."
                )
        else:
            feedback["scapular_stability"] = (
                "Buena estabilidad de la cintura escapular."
            )

        metrics = {
            "movimiento_hombros_usuario": user_shoulder_movement,
            "movimiento_hombros_experto": expert_shoulder_movement,
            "ratio_movimiento": float(movement_ratio),
            "asimetria_hombros_usuario": user_shoulder_asymmetry,
            "asimetria_hombros_experto": expert_shoulder_asymmetry,
            "ratio_asimetria": float(asymmetry_ratio),
        }

        return {"metrics": metrics, "feedback": feedback, "score": final_score}

    except Exception as e:
        logger.error(f"Error en análisis de estabilidad escapular: {e}")
        return {
            "metrics": {
                "movimiento_hombros_usuario": 0.0,
                "movimiento_hombros_experto": 0.0,
                "ratio_movimiento": 1.0,
                "asimetria_hombros_usuario": 0.0,
                "asimetria_hombros_experto": 0.0,
                "ratio_asimetria": 1.0,
            },
            "feedback": {
                "scapular_stability": "Análisis de estabilidad escapular no disponible."
            },
            "score": 50,
        }


# =============================================================================
# SENTADILLA - MÉTRICAS ESPECÍFICAS (usando config_manager)
# =============================================================================


def analyze_squat_depth(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    SENTADILLA: Análisis específico de profundidad usando ángulo de rodillas.
    """
    exercise_name = "sentadilla"

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "profundidad", exercise_name, config_path
    )

    user_knee_angles = []
    expert_knee_angles = []

    for i in range(len(user_data)):
        try:
            # USUARIO - calcular ángulo de rodilla (cadera-rodilla-tobillo)
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

            # EXPERTO - similar
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
            logger.warning(f"Error calculando ángulo de rodilla en frame {i}: {e}")
            continue

    if not user_knee_angles or not expert_knee_angles:
        return {
            "metrics": {},
            "feedback": {
                "squat_depth": "No se pudo analizar la profundidad de la sentadilla."
            },
            "score": 50,
        }

    # Calcular ángulo mínimo (máxima flexión)
    user_min_angle = np.min(user_knee_angles)
    expert_min_angle = np.min(expert_knee_angles)
    angle_diff = abs(user_min_angle - expert_min_angle)

    # Calcular score usando config_manager
    max_penalty = config_manager.get_penalty_config(
        exercise_name=exercise_name,
        metric_type="specific",
        metric_name="squat_depth",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        angle_diff, 0, max_penalty=max_penalty, metric_type="linear"
    )
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "profundidad"
    )

    # Generar feedback CON KEY EN INGLÉS
    feedback = {}
    if final_score >= 85:
        feedback["squat_depth"] = "Excelente profundidad en la sentadilla."
    elif final_score >= 70:
        if user_min_angle > expert_min_angle + 10:
            feedback["squat_depth"] = (
                "Tu sentadilla podría ser más profunda. Baja más las caderas."
            )
        else:
            feedback["squat_depth"] = "Buena profundidad en la sentadilla."
    else:
        if user_min_angle > expert_min_angle + 15:
            feedback["squat_depth"] = (
                "Tu sentadilla es muy superficial. Es importante bajar más para una técnica correcta."
            )
        else:
            feedback["squat_depth"] = (
                "La profundidad de tu sentadilla necesita trabajo."
            )

    metrics = {
        "angulo_minimo_usuario": user_min_angle,
        "angulo_minimo_experto": expert_min_angle,
        "diferencia_angulo": angle_diff,
        "frames_analizados": len(user_knee_angles),
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_knee_tracking_squat(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    SENTADILLA: Análisis del tracking de rodillas usando config_manager.
    """
    exercise_name = "sentadilla"

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "tracking_rodillas", exercise_name, config_path
    )

    # Calcular separación entre rodillas a lo largo del movimiento
    user_knee_separation = []
    expert_knee_separation = []

    for i in range(len(user_data)):
        try:
            # Separación entre rodillas (distancia en X)
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
            logger.warning(f"Error calculando separación de rodillas en frame {i}: {e}")
            continue

    if not user_knee_separation or not expert_knee_separation:
        return {
            "metrics": {},
            "feedback": {
                "knee_tracking": "No se pudo analizar el tracking de rodillas."
            },
            "score": 50,
        }

    # Analizar variabilidad de la separación (menor variabilidad = mejor tracking)
    user_knee_stability = np.std(user_knee_separation)
    expert_knee_stability = np.std(expert_knee_separation)
    stability_ratio = (
        user_knee_stability / expert_knee_stability if expert_knee_stability > 0 else 1
    )

    # Calcular score usando config_manager
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
        base_score, sensitivity_factor, "tracking_rodillas"
    )

    # Generar feedback CON KEY EN INGLÉS
    feedback = {}
    if final_score >= 85:
        feedback["knee_tracking"] = "Excelente control del tracking de rodillas."
    elif final_score >= 70:
        feedback["knee_tracking"] = (
            "Buen control general de las rodillas con ligeras variaciones."
        )
    else:
        feedback["knee_tracking"] = (
            "Las rodillas tienden a moverse hacia adentro. Mantén las rodillas alineadas con los pies."
        )

    metrics = {
        "estabilidad_rodillas_usuario": user_knee_stability,
        "estabilidad_rodillas_experto": expert_knee_stability,
        "ratio_estabilidad": stability_ratio,
        "separacion_promedio_usuario": np.mean(user_knee_separation),
        "separacion_promedio_experto": np.mean(expert_knee_separation),
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


# =============================================================================
# DOMINADA - MÉTRICAS ESPECÍFICAS (usando config_manager)
# =============================================================================


def analyze_body_swing_control_pullup(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    DOMINADA: Análisis específico de control de swing del cuerpo.
    """
    exercise_name = "dominada"

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "control_swing", exercise_name, config_path
    )

    # Analizar movimiento de caderas como indicador de swing
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

    # Calcular variabilidad del movimiento de caderas
    user_swing_x = np.std(user_hip_center_x)
    user_swing_z = np.std(user_hip_center_z)
    expert_swing_x = np.std(expert_hip_center_x)
    expert_swing_z = np.std(expert_hip_center_z)

    # Calcular ratio de swing total
    user_total_swing = np.sqrt(user_swing_x**2 + user_swing_z**2)
    expert_total_swing = np.sqrt(expert_swing_x**2 + expert_swing_z**2)
    swing_ratio = user_total_swing / expert_total_swing if expert_total_swing > 0 else 1

    # Calcular score usando config_manager
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
        base_score, sensitivity_factor, "control_swing"
    )

    # Generar feedback CON KEY EN INGLÉS
    feedback = {}
    if final_score >= 85:
        feedback["swing_control"] = "Excelente control del cuerpo durante la dominada."
    elif final_score >= 70:
        feedback["swing_control"] = "Buen control del cuerpo con ligero balanceo."
    else:
        feedback["swing_control"] = (
            "Hay demasiado balanceo del cuerpo. Mantén el core contraído para mayor estabilidad."
        )

    metrics = {
        "swing_usuario_x": user_swing_x,
        "swing_usuario_z": user_swing_z,
        "swing_total_usuario": user_total_swing,
        "swing_experto_x": expert_swing_x,
        "swing_experto_z": expert_swing_z,
        "swing_total_experto": expert_total_swing,
        "ratio_swing": swing_ratio,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_scapular_retraction_pullup(
    user_data, expert_data, exercise_config, config_path="config.json"
):
    """
    DOMINADA: Análisis de retracción escapular al inicio del movimiento.
    """
    exercise_name = "dominada"

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "retraccion_escapular", exercise_name, config_path
    )

    # Analizar separación entre hombros (menor separación = mayor retracción)
    user_shoulder_separation = abs(
        user_data["landmark_left_shoulder_x"].values
        - user_data["landmark_right_shoulder_x"].values
    )
    expert_shoulder_separation = abs(
        expert_data["landmark_left_shoulder_x"].values
        - expert_data["landmark_right_shoulder_x"].values
    )

    # Analizar primeros frames (posición inicial)
    initial_frames = min(10, len(user_shoulder_separation) // 4)
    user_initial_separation = np.mean(user_shoulder_separation[:initial_frames])
    expert_initial_separation = np.mean(expert_shoulder_separation[:initial_frames])

    # Ratio de separación inicial
    separation_ratio = (
        user_initial_separation / expert_initial_separation
        if expert_initial_separation > 0
        else 1
    )

    # Calcular score usando config_manager
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
        base_score, sensitivity_factor, "retraccion_escapular"
    )

    # Generar feedback CON KEY EN INGLÉS
    feedback = {}
    if final_score >= 85:
        feedback["scapular_retraction"] = "Excelente retracción escapular al inicio."
    elif final_score >= 70:
        feedback["scapular_retraction"] = (
            "Buena retracción escapular con ligeras variaciones."
        )
    else:
        if separation_ratio > 1.1:
            feedback["scapular_retraction"] = (
                "Necesitas mayor retracción escapular. Junta más las escápulas al inicio."
            )
        else:
            feedback["scapular_retraction"] = (
                "La retracción escapular necesita trabajo."
            )

    metrics = {
        "separacion_inicial_usuario": user_initial_separation,
        "separacion_inicial_experto": expert_initial_separation,
        "ratio_separacion": separation_ratio,
        "frames_analizados": initial_frames,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def _calculate_angle_3points(p1, p2, p3):
    """Calcula ángulo entre 3 puntos (p2 es el vértice)"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def _calculate_angle_2vectors(v1, v2):
    """Calcula ángulo entre 2 vectores"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


# =============================================================================
# FACTORY PARA MÉTRICAS ESPECÍFICAS
# =============================================================================


def get_specific_metrics_for_exercise(exercise_name):
    """
    Factory que retorna las funciones de métricas específicas para cada ejercicio.

    Returns:
        dict: Diccionario con las funciones específicas del ejercicio

    Raises:
        ValueError: Si el ejercicio no está soportado
    """
    exercise_name = exercise_name.lower().replace(" ", "_")

    if exercise_name == "press_militar":
        return {
            "metrica_especifica_a": analyze_elbow_abduction_angle_press,
            "metrica_especifica_b": analyze_scapular_stability_press,
        }
    elif exercise_name == "sentadilla":
        return {
            "metrica_especifica_a": analyze_squat_depth,
            "metrica_especifica_b": analyze_knee_tracking_squat,
        }
    elif exercise_name == "dominada":
        return {
            "metrica_especifica_a": analyze_body_swing_control_pullup,
            "metrica_especifica_b": analyze_scapular_retraction_pullup,
        }
    else:
        raise ValueError(f"Unsupported exercise: {exercise_name}")


def get_specific_metric_names_for_exercise(exercise_name):
    """
    Retorna los nombres de las métricas específicas para un ejercicio.

    Returns:
        dict: Nombres de las métricas específicas

    Raises:
        ValueError: Si el ejercicio no está soportado
    """
    exercise_name = exercise_name.lower().replace(" ", "_")

    if exercise_name == "press_militar":
        return {
            "metrica_especifica_a": "abduction_score",
            "metrica_especifica_b": "scapular_score",
        }
    elif exercise_name == "sentadilla":
        return {
            "metrica_especifica_a": "depth_score",
            "metrica_especifica_b": "knee_score",
        }
    elif exercise_name == "dominada":
        return {
            "metrica_especifica_a": "swing_score",
            "metrica_especifica_b": "retraction_score",
        }
    else:
        raise ValueError(f"Unsupported exercise: {exercise_name}")
