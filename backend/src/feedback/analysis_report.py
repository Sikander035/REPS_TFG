# backend/src/feedback/analysis_report.py - VERSIÓN UNIFICADA
import sys
import numpy as np
import pandas as pd
import os
import json
import logging
from scipy.signal import find_peaks, savgol_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.analysis_utils import (
    get_exercise_config,
    calculate_elbow_abduction_angle,
    determine_skill_level,
    generate_recommendations,
    apply_sensitivity_to_threshold,
)

logger = logging.getLogger(__name__)


def analyze_movement_amplitude(user_data, expert_data, exercise_config):
    """
    UNIFICADO: Analiza amplitud y calcula tanto feedback como score en el mismo lugar.
    """
    sensitivity_factor = exercise_config.get("sensitivity_factors", {}).get(
        "amplitud", 1.0
    )

    # Usar CODOS (promedio de ambos) en lugar de muñecas
    user_elbow_y = (
        user_data["landmark_right_elbow_y"].values
        + user_data["landmark_left_elbow_y"].values
    ) / 2
    expert_elbow_y = (
        expert_data["landmark_right_elbow_y"].values
        + expert_data["landmark_left_elbow_y"].values
    ) / 2

    # En MediaPipe: Y=0 arriba, Y=1 abajo
    user_highest_point = np.min(user_elbow_y)
    user_lowest_point = np.max(user_elbow_y)
    expert_highest_point = np.min(expert_elbow_y)
    expert_lowest_point = np.max(expert_elbow_y)

    # Calcular rango de movimiento (positivo)
    user_rom = user_lowest_point - user_highest_point
    expert_rom = expert_lowest_point - expert_highest_point
    rom_ratio = user_rom / expert_rom if expert_rom > 0 else 0

    # Analizar diferencia en el punto más bajo
    bottom_diff = (
        abs(user_lowest_point - expert_lowest_point) / expert_rom
        if expert_rom > 0
        else 0
    )

    # Aplicar sensibilidad a umbrales
    rom_threshold = apply_sensitivity_to_threshold(
        exercise_config["rom_threshold"], sensitivity_factor
    )
    bottom_diff_threshold = apply_sensitivity_to_threshold(
        exercise_config["bottom_diff_threshold"], sensitivity_factor
    )

    # UNIFICADO: Evaluar y calcular feedback + score en el mismo lugar
    feedback = {}
    score = 50  # Score por defecto

    # CORREGIDO: Cálculo simétrico de score basado en desviación del ideal (rom_ratio = 1.0)
    feedback = {}

    # Calcular desviación del ideal (1.0 = 100% igual al experto)
    deviation_from_ideal = abs(rom_ratio - 1.0)

    # Score base: penalizar cualquier desviación del ideal proporcionalmente
    base_score = max(
        0, min(100, 100 - (deviation_from_ideal * 100))
    )  # Penalización simétrica

    # MANTENER lógica original de feedback con sensibilidad aplicada correctamente
    if rom_ratio > 1.15:  # Exceso de amplitud
        if sensitivity_factor > 1.5 and rom_ratio > 1.25:
            feedback["amplitud"] = (
                "Tu rango de movimiento es excesivamente amplio. Es crítico "
                "controlar la bajada para evitar hiperextensión de los hombros."
            )
            score = max(5, base_score - 20)  # Penalización extra por ser crítico
        else:
            feedback["amplitud"] = (
                "Tu rango de movimiento es excesivo. Controla la bajada para evitar "
                "hiperextensión de los hombros."
            )
            score = max(15, base_score - 10)  # Penalización moderada

    elif bottom_diff > bottom_diff_threshold:  # Problema específico de posición baja
        if sensitivity_factor > 1.5 and bottom_diff > bottom_diff_threshold * 1.5:
            feedback["posicion_baja"] = (
                "Tu rango de movimiento es insuficiente. Es importante bajar hasta que "
                "las mancuernas estén aproximadamente a la altura de los hombros para técnica correcta."
            )
            score = max(10, base_score - 25)  # Penalización extra por ser crítico
        else:
            feedback["posicion_baja"] = (
                "Tu rango de movimento podría ser más amplio. Baja hasta que las mancuernas "
                "estén aproximadamente a la altura de los hombros."
            )
            score = max(25, base_score - 15)  # Penalización moderada

    elif rom_ratio < rom_threshold:  # Defecto general de amplitud
        if sensitivity_factor > 1.5 and rom_ratio < rom_threshold * 0.8:
            feedback["amplitud"] = (
                "Tu rango de movimiento es significativamente limitado. Es crítico "
                "trabajar en la amplitud completa: baja más los codos y extiende completamente arriba."
            )
            score = max(5, base_score - 20)  # Penalización extra por ser crítico
        else:
            feedback["amplitud"] = (
                "Tu rango de movimiento es limitado. Baja más los codos para una flexión completa "
                "y extiende completamente arriba."
            )
            score = max(15, base_score - 10)  # Penalización moderada
    else:
        feedback["amplitud"] = "Excelente amplitud de movimiento en los codos."
        score = max(90, base_score)  # Score excelente si está cerca del ideal

    metrics = {
        "rom_usuario": user_rom,
        "rom_experto": expert_rom,
        "rom_ratio": rom_ratio,
        "diferencia_posicion_baja": bottom_diff,
        "punto_mas_alto_usuario": user_highest_point,
        "punto_mas_bajo_usuario": user_lowest_point,
        "punto_mas_alto_experto": expert_highest_point,
        "punto_mas_bajo_experto": expert_lowest_point,
    }

    return {"metrics": metrics, "feedback": feedback, "score": score}


def analyze_elbow_abduction_angle(user_data, expert_data, exercise_config):
    """
    UNIFICADO: Analiza abducción de codos y calcula tanto feedback como score.
    """
    sensitivity_factor = exercise_config.get("sensitivity_factors", {}).get(
        "abduccion_codos", 1.0
    )

    # [MANTENER TODA LA LÓGICA DE CÁLCULO DE ÁNGULOS EXISTENTE...]
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
                "abduccion_codos": "No se pudo analizar la abducción de los codos."
            },
            "score": 50,
        }

    # [MANTENER LÓGICA DE PROCESAMIENTO DE SEÑALES...]
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
                "abduccion_codos": "Datos insuficientes para analizar abducción."
            },
            "score": 50,
        }

    user_clean_signal = user_avg_signal[user_valid]
    expert_clean_signal = expert_avg_signal[expert_valid]

    # [MANTENER LÓGICA DE SUAVIZADO Y DETECCIÓN DE PICOS...]
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

    # UNIFICADO: Aplicar sensibilidad y calcular feedback + score
    abduction_threshold = apply_sensitivity_to_threshold(
        exercise_config.get("abduction_angle_threshold", 15), sensitivity_factor
    )

    feedback = {}
    score = 50  # Score por defecto

    # CORREGIDO: Cálculo simétrico basado en desviación del ideal (abduction_diff = 0)
    abduction_threshold = apply_sensitivity_to_threshold(
        exercise_config.get("abduction_angle_threshold", 15), sensitivity_factor
    )

    feedback = {}

    # Score base: penalizar cualquier desviación del ideal (0 diferencia con experto)
    deviation_magnitude = abs(abduction_diff)
    base_score = max(
        0, min(100, 100 - (deviation_magnitude * 3))
    )  # Penalización simétrica por grado de diferencia

    # MANTENER lógica original de feedback
    if abs(abduction_diff) > abduction_threshold:
        if abduction_diff > 0:  # Usuario tiene ángulo mayor = más cerrado
            if sensitivity_factor > 1.5 and abduction_diff > abduction_threshold * 1.5:
                feedback["abduccion_codos"] = (
                    f"Tus codos están significativamente más cerrados que el experto. "
                    f"Es importante separarlos más del cuerpo para mejor mecánica."
                )
                score = max(5, base_score - 25)  # Penalización extra por ser crítico
            else:
                feedback["abduccion_codos"] = (
                    f"Tus codos están ligeramente más cerrados que el experto. "
                    f"Sepáralos un poco más del cuerpo."
                )
                score = max(25, base_score - 15)  # Penalización moderada
        else:  # Usuario tiene ángulo menor = más abierto
            if (
                sensitivity_factor > 1.5
                and abs(abduction_diff) > abduction_threshold * 1.5
            ):
                feedback["abduccion_codos"] = (
                    f"Tus codos se abren excesivamente durante el ejercicio. "
                    f"Es crítico acercarlos más al cuerpo para mayor seguridad."
                )
                score = max(5, base_score - 25)  # Penalización extra por ser crítico
            else:
                feedback["abduccion_codos"] = (
                    f"Tus codos están ligeramente más abiertos que el experto. "
                    f"Acércalos un poco más al cuerpo."
                )
                score = max(25, base_score - 15)  # Penalización moderada
    else:
        feedback["abduccion_codos"] = f"Excelente posición lateral de codos."
        score = max(90, base_score)  # Score excelente si está cerca del ideal

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

    return {"metrics": metrics, "feedback": feedback, "score": score}


def analyze_symmetry(user_data, expert_data, exercise_config):
    """UNIFICADO: Analiza simetría y calcula feedback + score."""
    sensitivity_factor = exercise_config.get("sensitivity_factors", {}).get(
        "simetria", 1.0
    )

    # Extraer posiciones verticales de codos
    user_r_elbow_y = user_data["landmark_right_elbow_y"].values
    user_l_elbow_y = user_data["landmark_left_elbow_y"].values

    # Calcular diferencia promedio entre lados
    height_diff = np.mean(np.abs(user_r_elbow_y - user_l_elbow_y))

    # Normalizar diferencia respecto al rango de movimiento promedio
    user_range = np.max(user_r_elbow_y) - np.min(user_r_elbow_y)
    normalized_diff = height_diff / user_range if user_range > 0 else 0

    # Comparar con simetría del experto
    expert_r_elbow_y = expert_data["landmark_right_elbow_y"].values
    expert_l_elbow_y = expert_data["landmark_left_elbow_y"].values
    expert_height_diff = np.mean(np.abs(expert_r_elbow_y - expert_l_elbow_y))
    expert_range = np.max(expert_r_elbow_y) - np.min(expert_r_elbow_y)
    expert_normalized_diff = (
        expert_height_diff / expert_range if expert_range > 0 else 0
    )

    # Ratio de asimetría usuario vs experto
    asymmetry_ratio = (
        normalized_diff / expert_normalized_diff if expert_normalized_diff > 0 else 1
    )

    # Aplicar sensibilidad al umbral
    symmetry_threshold = apply_sensitivity_to_threshold(
        exercise_config["symmetry_threshold"], sensitivity_factor
    )

    # UNIFICADO: Evaluar y calcular feedback + score
    feedback = {}
    score = 50

    # CORREGIDO: Cálculo simétrico basado en desviaciones de los ideales
    symmetry_threshold = apply_sensitivity_to_threshold(
        exercise_config["symmetry_threshold"], sensitivity_factor
    )

    feedback = {}

    # Calcular desviaciones de los ideales
    # Ideal 1: normalized_diff = 0 (sin diferencia de altura entre lados)
    # Ideal 2: asymmetry_ratio = 1.0 (igual asimetría que el experto)
    normalized_deviation = normalized_diff  # Ya es desviación del ideal (0)
    asymmetry_deviation = abs(asymmetry_ratio - 1.0)  # Desviación del ideal (1.0)

    # Score base usando la peor desviación
    worst_deviation = max(
        normalized_deviation * 100, asymmetry_deviation * 100
    )  # Convertir a porcentaje
    base_score = max(0, min(100, 100 - worst_deviation))  # Penalización simétrica

    # MANTENER lógica original de feedback
    if asymmetry_ratio > (1.8 / sensitivity_factor):
        if sensitivity_factor > 1.5:
            feedback["simetria"] = (
                "Hay una asimetría muy notable entre tu lado derecho e izquierdo. "
                "Es prioritario trabajar en equilibrar ambos brazos."
            )
            score = max(5, base_score - 25)  # Penalización extra por ser crítico
        else:
            feedback["simetria"] = (
                "Hay una asimetría notable entre tu lado derecho e izquierdo. "
                "Enfócate en levantar ambos brazos por igual."
            )
            score = max(15, base_score - 15)  # Penalización moderada
    elif normalized_diff > symmetry_threshold:
        if sensitivity_factor > 1.5 and normalized_diff > symmetry_threshold * 1.5:
            feedback["simetria"] = (
                "Se detecta asimetría significativa en el movimiento. "
                "Es importante trabajar en mantener ambos codos a la misma altura."
            )
            score = max(10, base_score - 20)  # Penalización extra por ser crítico
        else:
            feedback["simetria"] = (
                "Se detecta cierta asimetría en el movimiento. "
                "Intenta mantener ambos codos a la misma altura."
            )
            score = max(30, base_score - 10)  # Penalización moderada
    else:
        feedback["simetria"] = "Excelente simetría bilateral en el movimiento."
        score = max(90, base_score)  # Score excelente si está cerca del ideal

    metrics = {
        "diferencia_altura": height_diff,
        "diferencia_normalizada": normalized_diff,
        "diferencia_experto_normalizada": expert_normalized_diff,
        "ratio_asimetria": asymmetry_ratio,
        "rango_movimiento_usuario": user_range,
    }

    return {"metrics": metrics, "feedback": feedback, "score": score}


def analyze_movement_trajectory_3d(user_data, expert_data, exercise_config):
    """UNIFICADO: Analiza trayectoria 3D y calcula feedback + score."""
    sensitivity_factor = exercise_config.get("sensitivity_factors", {}).get(
        "trayectoria", 1.0
    )

    # Usar promedio de ambas muñecas para mayor estabilidad
    user_x = (
        user_data["landmark_right_wrist_x"].values
        + user_data["landmark_left_wrist_x"].values
    ) / 2
    user_y = (
        user_data["landmark_right_wrist_y"].values
        + user_data["landmark_left_wrist_y"].values
    ) / 2
    user_z = (
        user_data["landmark_right_wrist_z"].values
        + user_data["landmark_left_wrist_z"].values
    ) / 2

    expert_x = (
        expert_data["landmark_right_wrist_x"].values
        + expert_data["landmark_left_wrist_x"].values
    ) / 2
    expert_y = (
        expert_data["landmark_right_wrist_y"].values
        + expert_data["landmark_left_wrist_y"].values
    ) / 2
    expert_z = (
        expert_data["landmark_right_wrist_z"].values
        + expert_data["landmark_left_wrist_z"].values
    ) / 2

    # Análisis de desviaciones
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

    # Diferencias directas con experto
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

    # Aplicar sensibilidad a los umbrales
    lateral_threshold = apply_sensitivity_to_threshold(
        exercise_config["lateral_dev_threshold"], sensitivity_factor
    )
    frontal_threshold = apply_sensitivity_to_threshold(
        exercise_config.get("frontal_dev_threshold", 0.15), sensitivity_factor
    )

    # UNIFICADO: Evaluar y calcular feedback + score
    feedback = {}
    lateral_score = 75
    frontal_score = 75

    # CORREGIDO: Cálculo simétrico basado en desviaciones de los ideales
    lateral_threshold = apply_sensitivity_to_threshold(
        exercise_config["lateral_dev_threshold"], sensitivity_factor
    )
    frontal_threshold = apply_sensitivity_to_threshold(
        exercise_config.get("frontal_dev_threshold", 0.15), sensitivity_factor
    )

    feedback = {}

    # Calcular desviaciones de los ideales (lateral_ratio = 1.0, frontal_ratio = 1.0)
    lateral_deviation = abs(lateral_deviation_ratio - 1.0)
    frontal_deviation = abs(frontal_deviation_ratio - 1.0)

    # Scores base para cada aspecto
    lateral_base_score = max(
        0, min(100, 100 - (lateral_deviation * 100))
    )  # Penalización simétrica
    frontal_base_score = max(
        0, min(100, 100 - (frontal_deviation * 100))
    )  # Penalización simétrica

    # MANTENER lógica original de feedback
    # Evaluar lateral
    if lateral_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trayectoria_lateral"] = (
            "Tu movimiento se desvía excesivamente en dirección lateral. "
            "Concéntrate urgentemente en mantener las muñecas en línea vertical."
        )
        lateral_score = max(
            5, lateral_base_score - 30
        )  # Penalización extra por ser crítico
    elif normalized_trajectory_diff_x > lateral_threshold:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_x > lateral_threshold * 1.5
        ):
            feedback["trayectoria_lateral"] = (
                "Se detecta desviación lateral significativa en tu trayectoria. "
                "Es importante corregir para mantener un movimiento más vertical."
            )
            lateral_score = max(
                10, lateral_base_score - 20
            )  # Penalización extra por ser crítico
        else:
            feedback["trayectoria_lateral"] = (
                "Se detecta cierta desviación lateral en tu trayectoria. "
                "Intenta mantener un movimiento más vertical."
            )
            lateral_score = max(30, lateral_base_score - 10)  # Penalización moderada
    else:
        feedback["trayectoria_lateral"] = "Excelente control lateral del movimiento."
        lateral_score = max(90, lateral_base_score)  # Score excelente

    # Evaluar frontal
    if frontal_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trayectoria_frontal"] = (
            "Tu movimiento se desvía hacia adelante/atrás significativamente. "
            "Mantén las muñecas en un plano vertical consistente."
        )
        frontal_score = max(
            5, frontal_base_score - 30
        )  # Penalización extra por ser crítico
    elif normalized_trajectory_diff_z > frontal_threshold:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_z > frontal_threshold * 1.5
        ):
            feedback["trayectoria_frontal"] = (
                "Se detecta desviación frontal significativa en tu movimiento. "
                "Es importante mantener un plano vertical más consistente."
            )
            frontal_score = max(
                10, frontal_base_score - 20
            )  # Penalización extra por ser crítico
        else:
            feedback["trayectoria_frontal"] = (
                "Se detecta cierta desviación frontal en tu movimiento."
            )
            frontal_score = max(30, frontal_base_score - 10)  # Penalización moderada
    else:
        feedback["trayectoria_frontal"] = "Buen control frontal del movimiento."
        frontal_score = max(75, frontal_base_score)  # Score bueno

    # Score combinado (usar el peor)
    combined_score = min(lateral_score, frontal_score)

    # Feedback general
    if max(lateral_deviation_ratio, frontal_deviation_ratio) < (
        1.5 / sensitivity_factor
    ):
        feedback["trayectoria"] = "Excelente trayectoria 3D del movimiento."
        combined_score = max(combined_score, 90)
    else:
        feedback["trayectoria"] = "La trayectoria del movimiento puede mejorarse."

    metrics = {
        "desviacion_lateral_usuario": lateral_deviation_user,
        "desviacion_lateral_experto": lateral_deviation_expert,
        "ratio_desviacion_lateral": lateral_deviation_ratio,
        "desviacion_frontal_usuario": frontal_deviation_user,
        "desviacion_frontal_experto": frontal_deviation_expert,
        "ratio_desviacion_frontal": frontal_deviation_ratio,
        "diferencia_trayectoria_x": normalized_trajectory_diff_x,
        "diferencia_trayectoria_z": normalized_trajectory_diff_z,
    }

    return {"metrics": metrics, "feedback": feedback, "score": combined_score}


def analyze_speed(user_data, expert_data, exercise_config):
    """UNIFICADO: Analiza velocidad y calcula feedback + score."""
    sensitivity_factor = exercise_config.get("sensitivity_factors", {}).get(
        "velocidad", 1.0
    )

    # Usar promedio de codos para velocidad
    user_elbow_y = (
        user_data["landmark_right_elbow_y"].values
        + user_data["landmark_left_elbow_y"].values
    ) / 2
    expert_elbow_y = (
        expert_data["landmark_right_elbow_y"].values
        + expert_data["landmark_left_elbow_y"].values
    ) / 2

    # Calcular velocidades
    user_velocity = np.gradient(user_elbow_y)
    expert_velocity = np.gradient(expert_elbow_y)

    # Separar fases
    user_concentric = user_velocity[user_velocity < 0]
    user_eccentric = user_velocity[user_velocity > 0]
    expert_concentric = expert_velocity[expert_velocity < 0]
    expert_eccentric = expert_velocity[expert_velocity > 0]

    # Calcular velocidades promedio
    user_concentric_avg = (
        np.mean(np.abs(user_concentric)) if len(user_concentric) > 0 else 0
    )
    user_eccentric_avg = (
        np.mean(np.abs(user_eccentric)) if len(user_eccentric) > 0 else 0
    )
    expert_concentric_avg = (
        np.mean(np.abs(expert_concentric)) if len(expert_concentric) > 0 else 0
    )
    expert_eccentric_avg = (
        np.mean(np.abs(expert_eccentric)) if len(expert_eccentric) > 0 else 0
    )

    # Calcular ratios
    concentric_ratio = (
        user_concentric_avg / expert_concentric_avg if expert_concentric_avg > 0 else 1
    )
    eccentric_ratio = (
        user_eccentric_avg / expert_eccentric_avg if expert_eccentric_avg > 0 else 1
    )

    # Aplicar sensibilidad al umbral
    velocity_threshold = apply_sensitivity_to_threshold(
        exercise_config["velocity_ratio_threshold"], sensitivity_factor
    )

    # UNIFICADO: Evaluar y calcular feedback + score
    feedback = {}
    concentric_score = 75
    eccentric_score = 75

    # CORREGIDO: Cálculo simétrico basado en desviaciones de los ideales
    velocity_threshold = apply_sensitivity_to_threshold(
        exercise_config["velocity_ratio_threshold"], sensitivity_factor
    )

    feedback = {}

    # Calcular desviaciones de los ideales (concentric_ratio = 1.0, eccentric_ratio = 1.0)
    concentric_deviation = abs(concentric_ratio - 1.0)
    eccentric_deviation = abs(eccentric_ratio - 1.0)

    # Scores base para cada fase
    concentric_base_score = max(
        0, min(100, 100 - (concentric_deviation * 100))
    )  # Penalización simétrica
    eccentric_base_score = max(
        0, min(100, 100 - (eccentric_deviation * 100))
    )  # Penalización simétrica

    # MANTENER lógica original de feedback
    # Evaluar fase concéntrica
    if concentric_ratio < (1 - velocity_threshold):
        if sensitivity_factor > 1.5:
            feedback["velocidad_subida"] = (
                "La fase de subida es significativamente muy lenta comparada con el experto. "
                "Es importante ser más explosivo en la fase concéntrica."
            )
            concentric_score = max(
                5, concentric_base_score - 25
            )  # Penalización extra por ser crítico
        else:
            feedback["velocidad_subida"] = (
                "La fase de subida es demasiado lenta comparada con el experto. "
                "Intenta ser más explosivo en la fase concéntrica."
            )
            concentric_score = max(
                15, concentric_base_score - 15
            )  # Penalización moderada
    elif concentric_ratio > (1 + velocity_threshold):
        feedback["velocidad_subida"] = (
            "La fase de subida es demasiado rápida. Controla más el movimiento."
        )
        concentric_score = max(20, concentric_base_score - 10)  # Penalización moderada
    else:
        feedback["velocidad_subida"] = "Excelente velocidad en la fase de subida."
        concentric_score = max(90, concentric_base_score)  # Score excelente

    # Evaluar fase excéntrica
    if eccentric_ratio < (1 - velocity_threshold):
        feedback["velocidad_bajada"] = (
            "La fase de bajada es demasiado lenta. Controla el descenso pero no lo ralentices en exceso."
        )
        eccentric_score = max(30, eccentric_base_score - 10)  # Penalización leve
    elif eccentric_ratio > (1 + velocity_threshold):
        if sensitivity_factor > 1.5:
            feedback["velocidad_bajada"] = (
                "La fase de bajada es significativamente muy rápida. "
                "Es crítico controlar más el descenso para técnica segura."
            )
            eccentric_score = max(
                5, eccentric_base_score - 25
            )  # Penalización extra por ser crítico
        else:
            feedback["velocidad_bajada"] = (
                "La fase de bajada es demasiado rápida. Intenta controlar más el descenso."
            )
            eccentric_score = max(
                15, eccentric_base_score - 15
            )  # Penalización moderada
    else:
        feedback["velocidad_bajada"] = "Excelente control en la fase de bajada."
        eccentric_score = max(90, eccentric_base_score)  # Score excelente

    # Score combinado (promedio)
    combined_score = (concentric_score + eccentric_score) / 2

    metrics = {
        "velocidad_subida_usuario": user_concentric_avg,
        "velocidad_subida_experto": expert_concentric_avg,
        "ratio_subida": concentric_ratio,
        "velocidad_bajada_usuario": user_eccentric_avg,
        "velocidad_bajada_experto": expert_eccentric_avg,
        "ratio_bajada": eccentric_ratio,
    }

    return {"metrics": metrics, "feedback": feedback, "score": combined_score}


def analyze_scapular_stability(user_data, expert_data, exercise_config):
    """UNIFICADO: Analiza estabilidad escapular y calcula feedback + score."""
    sensitivity_factor = exercise_config.get("sensitivity_factors", {}).get(
        "estabilidad_escapular", 1.0
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

        # Aplicar sensibilidad al umbral
        stability_threshold = apply_sensitivity_to_threshold(
            exercise_config.get("scapular_stability_threshold", 1.5), sensitivity_factor
        )

        # UNIFICADO: Evaluar y calcular feedback + score
        feedback = {}
        score = 75

        # CORREGIDO: Cálculo simétrico basado en desviaciones de los ideales
        stability_threshold = apply_sensitivity_to_threshold(
            exercise_config.get("scapular_stability_threshold", 1.5), sensitivity_factor
        )

        feedback = {}

        # Calcular desviaciones de los ideales (movement_ratio = 1.0, asymmetry_ratio = 1.0)
        movement_deviation = abs(movement_ratio - 1.0)
        asymmetry_deviation = abs(asymmetry_ratio - 1.0)

        # Score base usando la peor desviación
        worst_deviation = max(movement_deviation, asymmetry_deviation)
        base_score = max(
            0, min(100, 100 - (worst_deviation * 100))
        )  # Penalización simétrica

        # MANTENER lógica original de feedback
        if movement_ratio > (2.0 / sensitivity_factor):
            if sensitivity_factor > 1.5:
                feedback["estabilidad_escapular"] = (
                    "Tus hombros se mueven excesivamente durante el press. "
                    "Es crítico mantener una posición mucho más estable de la cintura escapular."
                )
                score = max(5, base_score - 30)  # Penalización extra por ser crítico
            else:
                feedback["estabilidad_escapular"] = (
                    "Tus hombros se mueven excesivamente durante el press. "
                    "Mantén una posición más estable de la cintura escapular."
                )
                score = max(15, base_score - 20)  # Penalización moderada
        elif asymmetry_ratio > (2.0 / sensitivity_factor):
            feedback["estabilidad_escapular"] = (
                "Se detecta asimetría en el movimiento de tus hombros. "
                "Concéntrate en mantener ambos hombros equilibrados."
            )
            score = max(20, base_score - 15)  # Penalización moderada
        elif movement_ratio > stability_threshold:
            if sensitivity_factor > 1.5:
                feedback["estabilidad_escapular"] = (
                    "Se detecta inestabilidad notable en tu cintura escapular. "
                    "Es importante practicar mantener los hombros en una posición más fija."
                )
                score = max(10, base_score - 25)  # Penalización extra por ser crítico
            else:
                feedback["estabilidad_escapular"] = (
                    "Se detecta cierta inestabilidad en tu cintura escapular. "
                    "Practica mantener los hombros en una posición más fija."
                )
                score = max(30, base_score - 10)  # Penalización moderada
        else:
            feedback["estabilidad_escapular"] = (
                "Buena estabilidad de la cintura escapular."
            )
            score = max(75, base_score)  # Score bueno si está cerca del ideal

        metrics = {
            "movimiento_hombros_usuario": user_shoulder_movement,
            "movimiento_hombros_experto": expert_shoulder_movement,
            "ratio_movimiento": float(movement_ratio),
            "asimetria_hombros_usuario": user_shoulder_asymmetry,
            "asimetria_hombros_experto": expert_shoulder_asymmetry,
            "ratio_asimetria": float(asymmetry_ratio),
        }

        return {"metrics": metrics, "feedback": feedback, "score": score}

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
                "estabilidad_escapular": "Análisis de estabilidad escapular no disponible."
            },
            "score": 50,
        }


def run_exercise_analysis(
    user_data, expert_data, exercise_name="press_militar", config_path="config.json"
):
    """UNIFICADO: Ejecuta análisis completo y calcula scores directamente."""
    logger.info(f"Iniciando análisis unificado: {exercise_name}")

    # Cargar configuración
    exercise_config = get_exercise_config(exercise_name, config_path)

    # Ejecutar todos los análisis (ahora cada uno retorna metrics, feedback Y score)
    amplitude_result = analyze_movement_amplitude(
        user_data, expert_data, exercise_config
    )
    abduction_result = analyze_elbow_abduction_angle(
        user_data, expert_data, exercise_config
    )
    symmetry_result = analyze_symmetry(user_data, expert_data, exercise_config)
    trajectory_result = analyze_movement_trajectory_3d(
        user_data, expert_data, exercise_config
    )
    speed_result = analyze_speed(user_data, expert_data, exercise_config)
    scapular_result = analyze_scapular_stability(
        user_data, expert_data, exercise_config
    )

    # Combinar métricas
    all_metrics = {
        "amplitud": amplitude_result["metrics"],
        "abduccion_codos": abduction_result["metrics"],
        "simetria": symmetry_result["metrics"],
        "trayectoria": trajectory_result["metrics"],
        "velocidad": speed_result["metrics"],
        "estabilidad_escapular": scapular_result["metrics"],
    }

    # Combinar feedback
    all_feedback = {
        **amplitude_result["feedback"],
        **abduction_result["feedback"],
        **symmetry_result["feedback"],
        **trajectory_result["feedback"],
        **speed_result["feedback"],
        **scapular_result["feedback"],
    }

    # UNIFICADO: Calcular score global usando scores individuales de cada análisis
    individual_scores = {
        "rom_score": amplitude_result["score"],
        "abduction_score": abduction_result["score"],
        "sym_score": symmetry_result["score"],
        "path_score": trajectory_result["score"],
        "speed_score": speed_result["score"],
        "scapular_score": scapular_result["score"],
    }

    # CORREGIDO: Validar que todos los scores estén entre 0 y 100
    for key, score in individual_scores.items():
        if score < 0 or score > 100:
            logger.warning(
                f"Score fuera de rango detectado: {key}={score:.1f}. Corrigiendo..."
            )
            individual_scores[key] = max(0, min(100, score))

    # Pesos de configuración
    weights = exercise_config.get(
        "scoring_weights",
        {
            "rom_score": 0.20,
            "abduction_score": 0.20,
            "sym_score": 0.15,
            "path_score": 0.20,
            "speed_score": 0.15,
            "scapular_score": 0.10,
        },
    )

    overall_score = sum(individual_scores[key] * weights[key] for key in weights.keys())

    # CORREGIDO: Asegurar que el score global también esté en rango
    overall_score = max(0, min(100, overall_score))

    skill_level = determine_skill_level(overall_score, exercise_config)

    logger.info(
        f"Análisis unificado completado - Puntuación: {overall_score:.1f}/100 - Nivel: {skill_level}"
    )
    logger.info(
        f"Scores individuales: {[f'{k}={v:.1f}' for k,v in individual_scores.items()]}"
    )

    return {
        "metrics": all_metrics,
        "feedback": all_feedback,
        "score": overall_score,
        "level": skill_level,
        "individual_scores": individual_scores,  # NUEVO: scores individuales
        "exercise_config": exercise_config,
        "sensitivity_factors": exercise_config.get("sensitivity_factors", {}),
    }


def generate_analysis_report(analysis_results, exercise_name, output_path=None):
    """Genera un informe completo con los resultados del análisis UNIFICADO."""
    report = {
        "ejercicio": exercise_name,
        "puntuacion_global": round(analysis_results["score"], 1),
        "puntuaciones_individuales": {  # NUEVO: scores individuales
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
        "version_analisis": "unified_system_v2.0",  # Actualizado
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
        logger.info(f"Informe unificado guardado en: {output_path}")

    return report
