# backend/src/feedback/analysis_report.py - MEJORA: Detección de repeticiones interna
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
    calculate_overall_score,
    determine_skill_level,
    generate_recommendations,
    apply_sensitivity_to_threshold,
)

logger = logging.getLogger(__name__)


def analyze_movement_amplitude(user_data, expert_data, exercise_config):
    """
    Analiza la amplitud del movimiento usando los CODOS como referencia principal.
    CORREGIDO: Lógica de umbrales y orden de evaluación.
    """
    # Obtener factor de sensibilidad para amplitud
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

    # CORREGIDO: Aplicar sensibilidad solo al umbral inferior, no al superior
    rom_threshold = apply_sensitivity_to_threshold(
        exercise_config["rom_threshold"], sensitivity_factor
    )
    bottom_diff_threshold = apply_sensitivity_to_threshold(
        exercise_config["bottom_diff_threshold"], sensitivity_factor
    )

    # CORREGIDO: Generar feedback con lógica y orden correctos
    feedback = {}

    # CORREGIDO: Evaluar en orden correcto - casos más específicos primero
    if rom_ratio > 1.15:  # ✅ FIJO: Sin división por sensibilidad
        if sensitivity_factor > 1.5 and rom_ratio > 1.25:
            feedback["amplitud"] = (
                "Tu rango de movimiento es excesivamente amplio. Es crítico "
                "controlar la bajada para evitar hiperextensión de los hombros."
            )
        else:
            feedback["amplitud"] = (
                "Tu rango de movimiento es excesivo. Controla la bajada para evitar "
                "hiperextensión de los hombros."
            )
    elif bottom_diff > bottom_diff_threshold:
        if sensitivity_factor > 1.5 and bottom_diff > bottom_diff_threshold * 1.5:
            feedback["posicion_baja"] = (
                "Tu rango de movimiento es insuficiente. Es importante bajar hasta que "
                "las mancuernas estén aproximadamente a la altura de los hombros para técnica correcta."
            )
        else:
            feedback["posicion_baja"] = (
                "Tu rango de movimento es podría ser mas amplio. Baja hasta que las mancuernas "
                "estén aproximadamente a la altura de los hombros."
            )
    elif rom_ratio < rom_threshold:  # CORREGIDO: Orden correcto
        if sensitivity_factor > 1.5 and rom_ratio < rom_threshold * 0.8:
            feedback["amplitud"] = (
                "Tu rango de movimiento es significativamente limitado. Es crítico "
                "trabajar en la amplitud completa: baja más los codos y extiende completamente arriba."
            )
        else:
            feedback["amplitud"] = (
                "Tu rango de movimiento es limitado. Baja más los codos para una flexión completa "
                "y extiende completamente arriba."
            )
    else:
        feedback["amplitud"] = "Excelente amplitud de movimiento en los codos."

    # Log de sensibilidad aplicada
    if sensitivity_factor != 1.0:
        logger.debug(
            f"Amplitud - Sensibilidad {sensitivity_factor}: ROM threshold {rom_threshold:.3f} (base: {exercise_config['rom_threshold']})"
        )

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

    return {"metrics": metrics, "feedback": feedback}


def analyze_elbow_abduction_angle(user_data, expert_data, exercise_config):
    """
    CORREGIDO: Analiza el ángulo de abducción de los codos midiendo el ángulo
    del vector codo-hombro respecto al plano horizontal usando proyección.
    """
    # Obtener factor de sensibilidad para abducción
    sensitivity_factor = exercise_config.get("sensitivity_factors", {}).get(
        "abduccion_codos", 1.0
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

    # Calcular ángulos del experto para todos los frames
    for i in range(len(expert_data)):
        try:
            # === EXPERTO - AMBOS CODOS ===
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

            # Verificar que no hay NaN en las coordenadas
            if (
                not np.isnan(expert_right_shoulder).any()
                and not np.isnan(expert_right_elbow).any()
                and not np.isnan(expert_left_shoulder).any()
                and not np.isnan(expert_left_elbow).any()
            ):

                # Calcular ángulos de abducción del experto
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
        }

    # Convertir a arrays y limpiar NaN
    user_right_abduction = np.array(user_right_abduction)
    user_left_abduction = np.array(user_left_abduction)
    expert_right_abduction = np.array(expert_right_abduction)
    expert_left_abduction = np.array(expert_left_abduction)

    # Calcular señal promedio de abducción (ambos codos)
    user_avg_signal = (user_right_abduction + user_left_abduction) / 2
    expert_avg_signal = (expert_right_abduction + expert_left_abduction) / 2

    # Limpiar NaN para detección de picos
    user_valid = ~np.isnan(user_avg_signal)
    expert_valid = ~np.isnan(expert_avg_signal)

    if np.sum(user_valid) < 10 or np.sum(expert_valid) < 10:
        return {
            "metrics": {},
            "feedback": {
                "abduccion_codos": "Datos insuficientes para analizar abducción."
            },
        }

    user_clean_signal = user_avg_signal[user_valid]
    expert_clean_signal = expert_avg_signal[expert_valid]

    # Para el press militar, queremos analizar los momentos de MÍNIMA abducción
    # (cuando los codos están más abiertos = ángulos menores)
    try:
        # Suavizar ambas señales
        window_length = min(9, len(user_clean_signal) // 4)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(3, window_length)

        user_smooth = savgol_filter(user_clean_signal, window_length, 2)
        expert_smooth = savgol_filter(expert_clean_signal, window_length, 2)

        logger.info(
            f"Ángulos de abducción lateral - Usuario promedio: {np.mean(user_clean_signal):.1f}°"
        )
        logger.info(
            f"Ángulos de abducción lateral - Experto promedio: {np.mean(expert_clean_signal):.1f}°"
        )
        logger.info(
            f"Ángulos mínimos (máxima abducción lateral) - Usuario: {np.min(user_clean_signal):.1f}°, Experto: {np.min(expert_clean_signal):.1f}°"
        )

        # Detectar mínimos (máxima abducción) usando find_peaks en señal invertida
        prominence = max(2, np.std(user_smooth) * 0.3)
        distance = max(15, len(user_smooth) // 12)
        height = np.percentile(user_smooth, 30)  # Percentil 30 para capturar mínimos

        logger.info(
            f"Parámetros para detección de mínimos - Prominence: {prominence:.1f}°, Distance: {distance}, Height máxima: {height:.1f}°"
        )

        # Detectar mínimos (máxima abducción)
        user_valleys, _ = find_peaks(
            -user_smooth,  # Invertir la señal para encontrar mínimos
            prominence=prominence,
            distance=distance,
            height=-height,  # Altura negativa porque invertimos la señal
        )

        # Si no detecta suficientes mínimos, ser más permisivo
        if len(user_valleys) < 2:
            logger.info("Pocos mínimos detectados, reduciendo exigencia...")
            prominence = max(1, np.std(user_smooth) * 0.2)
            height = np.percentile(user_smooth, 40)

            user_valleys, _ = find_peaks(
                -user_smooth, prominence=prominence, distance=distance, height=-height
            )

        logger.info(
            f"Mínimos (máxima abducción) detectados: {len(user_valleys)} en posiciones {user_valleys}"
        )

        # Si aún no detecta suficientes mínimos, usar estrategia de percentil
        if len(user_valleys) < 2:
            logger.info("Usando estrategia de percentiles para mínimos")

            # Usar percentil 10 para capturar los valores más bajos (máxima abducción)
            user_threshold = np.percentile(user_smooth, 10)
            user_low_indices = np.where(user_smooth <= user_threshold)[0]

            if len(user_low_indices) < 3:
                user_threshold = np.percentile(user_smooth, 15)
                user_low_indices = np.where(user_smooth <= user_threshold)[0]

            user_valley_values = user_clean_signal[user_low_indices]
            expert_valley_values = expert_clean_signal[
                user_low_indices
            ]  # Mismos índices

            logger.info(
                f"Usando percentiles - Umbral: {user_threshold:.1f}°, {len(user_valley_values)} valores"
            )
        else:
            # Extraer valores en las posiciones de los mínimos detectados
            user_valley_values = user_clean_signal[user_valleys]
            expert_valley_values = expert_clean_signal[
                user_valleys
            ]  # Mismos índices temporales

        logger.info(f"Valores en mínimos detectados - Usuario: {user_valley_values}")
        logger.info(f"Valores en mínimos detectados - Experto: {expert_valley_values}")

        # Calcular métricas de los mínimos (máxima abducción)
        user_min_abduction = np.mean(user_valley_values)
        expert_min_abduction = np.mean(expert_valley_values)

        # También calcular el mínimo absoluto
        user_absolute_min = np.min(user_valley_values)
        expert_absolute_min = np.min(expert_valley_values)

        # Diferencia en abducción mínima (ángulos menores = más abierto)
        abduction_diff = user_min_abduction - expert_min_abduction
        absolute_diff = user_absolute_min - expert_absolute_min

        logger.info(
            f"Ángulos promedio en máxima abducción lateral - Usuario: {user_min_abduction:.1f}°, Experto: {expert_min_abduction:.1f}°"
        )
        logger.info(f"Diferencia en ángulos: {abduction_diff:.1f}°")

    except Exception as e:
        logger.error(f"Error en detección de mínimos: {e}")
        # Fallback: usar promedio general
        user_min_abduction = np.mean(user_clean_signal)
        expert_min_abduction = np.mean(expert_clean_signal)
        abduction_diff = user_min_abduction - expert_min_abduction
        absolute_diff = abduction_diff
        user_valley_values = user_clean_signal
        expert_valley_values = expert_clean_signal

    # APLICAR SENSIBILIDAD al umbral
    abduction_threshold = apply_sensitivity_to_threshold(
        exercise_config.get("abduction_angle_threshold", 15), sensitivity_factor
    )

    # Generar feedback basado en diferencia en MÍNIMOS (máxima abducción)
    feedback = {}

    # Interpretar resultados: ángulo menor = más abierto (más abducción)
    if abs(abduction_diff) > abduction_threshold:
        if abduction_diff > 0:  # Usuario tiene ángulo mayor = más cerrado
            if sensitivity_factor > 1.5 and abduction_diff > abduction_threshold * 1.5:
                feedback["abduccion_codos"] = (
                    f"Tus codos están significativamente más cerrados que el experto. "
                    f"Es importante separarlos más del cuerpo para mejor mecánica."
                )
            else:
                feedback["abduccion_codos"] = (
                    f"Tus codos están ligeramente más cerrados que el experto. "
                    f"Sepáralos un poco más del cuerpo."
                )
        else:  # Usuario tiene ángulo menor = más abierto
            if (
                sensitivity_factor > 1.5
                and abs(abduction_diff) > abduction_threshold * 1.5
            ):
                feedback["abduccion_codos"] = (
                    f"Tus codos se abren excesivamente durante el ejercicio. "
                    f"Es crítico acercarlos más al cuerpo para mayor seguridad."
                )
            else:
                feedback["abduccion_codos"] = (
                    f"Tus codos están ligeramente más abiertos que el experto. "
                    f"Acércalos un poco más al cuerpo."
                )
    else:
        feedback["abduccion_codos"] = f"Excelente posición lateral de codos. "

    # Log de sensibilidad aplicada
    if sensitivity_factor != 1.0:
        logger.debug(
            f"Abducción lateral - Sensibilidad {sensitivity_factor}: threshold {abduction_threshold:.1f}° (base: {exercise_config.get('abduction_angle_threshold', 15)}°)"
        )

    # Métricas actualizadas
    metrics = {
        "abduccion_lateral_minima_usuario": user_min_abduction,
        "abduccion_lateral_minima_experto": expert_min_abduction,
        "diferencia_abduccion_lateral": abduction_diff,
        "min_absoluto_usuario": user_absolute_min,
        "min_absoluto_experto": expert_absolute_min,
        "diferencia_absoluta": absolute_diff,
        "num_minimos_detectados": len(user_valley_values),
        "frames_totales_usuario": len(user_data),
        "frames_totales_experto": len(expert_data),
        "analisis_tipo": "proyeccion_XZ_angulo_con_eje_X",
        "metrica_abduccion": "angulo_lateral_proyeccion_hombro_codo",
        "interpretacion": "0°=máxima_abducción_lateral, 90°=mínima_abducción_lateral",
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_symmetry(user_data, expert_data, exercise_config):
    """Analiza la simetría entre el lado izquierdo y derecho. CORREGIDO."""
    # Obtener factor de sensibilidad para simetría
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

    # APLICAR SENSIBILIDAD al umbral
    symmetry_threshold = apply_sensitivity_to_threshold(
        exercise_config["symmetry_threshold"], sensitivity_factor
    )

    # CORREGIDO: Generar feedback con lógica correcta
    feedback = {}

    # CORREGIDO: Evaluar ratio y diferencia absoluta por separado
    if asymmetry_ratio > (
        1.8 / sensitivity_factor
    ):  # Usuario significativamente más asimétrico que experto
        if sensitivity_factor > 1.5:
            feedback["simetria"] = (
                "Hay una asimetría muy notable entre tu lado derecho e izquierdo. "
                "Es prioritario trabajar en equilibrar ambos brazos."
            )
        else:
            feedback["simetria"] = (
                "Hay una asimetría notable entre tu lado derecho e izquierdo. "
                "Enfócate en levantar ambos brazos por igual."
            )
    elif normalized_diff > symmetry_threshold:  # Asimetría absoluta alta
        if sensitivity_factor > 1.5 and normalized_diff > symmetry_threshold * 1.5:
            feedback["simetria"] = (
                "Se detecta asimetría significativa en el movimiento. "
                "Es importante trabajar en mantener ambos codos a la misma altura."
            )
        else:
            feedback["simetria"] = (
                "Se detecta cierta asimetría en el movimiento. "
                "Intenta mantener ambos codos a la misma altura."
            )
    else:
        feedback["simetria"] = "Excelente simetría bilateral en el movimiento."

    # Log de sensibilidad aplicada
    if sensitivity_factor != 1.0:
        logger.debug(
            f"Simetría - Sensibilidad {sensitivity_factor}: threshold {symmetry_threshold:.3f} (base: {exercise_config['symmetry_threshold']})"
        )

    metrics = {
        "diferencia_altura": height_diff,
        "diferencia_normalizada": normalized_diff,
        "diferencia_experto_normalizada": expert_normalized_diff,
        "ratio_asimetria": asymmetry_ratio,
        "rango_movimiento_usuario": user_range,
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_movement_trajectory_3d(user_data, expert_data, exercise_config):
    """
    Analiza la trayectoria 3D completa del movimiento. REVISADO.
    """
    # Obtener factor de sensibilidad para trayectoria
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

    # APLICAR SENSIBILIDAD a los umbrales
    lateral_threshold = apply_sensitivity_to_threshold(
        exercise_config["lateral_dev_threshold"], sensitivity_factor
    )
    frontal_threshold = apply_sensitivity_to_threshold(
        exercise_config.get("frontal_dev_threshold", 0.15), sensitivity_factor
    )

    # Generar feedback - SIMPLIFICADO para evitar confusión
    feedback = {}

    # Evaluar lateral
    if lateral_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trayectoria_lateral"] = (
            "Tu movimiento se desvía excesivamente en dirección lateral. "
            "Concéntrate urgentemente en mantener las muñecas en línea vertical."
        )
    elif normalized_trajectory_diff_x > lateral_threshold:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_x > lateral_threshold * 1.5
        ):
            feedback["trayectoria_lateral"] = (
                "Se detecta desviación lateral significativa en tu trayectoria. "
                "Es importante corregir para mantener un movimiento más vertical."
            )
        else:
            feedback["trayectoria_lateral"] = (
                "Se detecta cierta desviación lateral en tu trayectoria. "
                "Intenta mantener un movimiento más vertical."
            )
    else:
        feedback["trayectoria_lateral"] = "Excelente control lateral del movimiento."

    # Evaluar frontal
    if frontal_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trayectoria_frontal"] = (
            "Tu movimiento se desvía hacia adelante/atrás significativamente. "
            "Mantén las muñecas en un plano vertical consistente."
        )
    elif normalized_trajectory_diff_z > frontal_threshold:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_z > frontal_threshold * 1.5
        ):
            feedback["trayectoria_frontal"] = (
                "Se detecta desviación frontal significativa en tu movimiento. "
                "Es importante mantener un plano vertical más consistente."
            )
        else:
            feedback["trayectoria_frontal"] = (
                "Se detecta cierta desviación frontal en tu movimiento."
            )
    else:
        feedback["trayectoria_frontal"] = "Buen control frontal del movimiento."

    # Feedback general
    if max(lateral_deviation_ratio, frontal_deviation_ratio) < (
        1.5 / sensitivity_factor
    ):
        feedback["trayectoria"] = "Excelente trayectoria 3D del movimiento."
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

    return {"metrics": metrics, "feedback": feedback}


def analyze_speed(user_data, expert_data, exercise_config):
    """Analiza la velocidad de ejecución. REVISADO - Parece correcto."""
    # Obtener factor de sensibilidad para velocidad
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

    # APLICAR SENSIBILIDAD al umbral
    velocity_threshold = apply_sensitivity_to_threshold(
        exercise_config["velocity_ratio_threshold"], sensitivity_factor
    )

    # Generar feedback - LÓGICA CORRECTA
    feedback = {}

    if concentric_ratio < (1 - velocity_threshold):
        if sensitivity_factor > 1.5:
            feedback["velocidad_subida"] = (
                "La fase de subida es significativamente muy lenta comparada con el experto. "
                "Es importante ser más explosivo en la fase concéntrica."
            )
        else:
            feedback["velocidad_subida"] = (
                "La fase de subida es demasiado lenta comparada con el experto. "
                "Intenta ser más explosivo en la fase concéntrica."
            )
    elif concentric_ratio > (1 + velocity_threshold):
        feedback["velocidad_subida"] = (
            "La fase de subida es demasiado rápida. Controla más el movimiento."
        )
    else:
        feedback["velocidad_subida"] = "Excelente velocidad en la fase de subida."

    if eccentric_ratio < (1 - velocity_threshold):
        feedback["velocidad_bajada"] = (
            "La fase de bajada es demasiado lenta. Controla el descenso pero no lo ralentices en exceso."
        )
    elif eccentric_ratio > (1 + velocity_threshold):
        if sensitivity_factor > 1.5:
            feedback["velocidad_bajada"] = (
                "La fase de bajada es significativamente muy rápida. "
                "Es crítico controlar más el descenso para técnica segura."
            )
        else:
            feedback["velocidad_bajada"] = (
                "La fase de bajada es demasiado rápida. Intenta controlar más el descenso."
            )
    else:
        feedback["velocidad_bajada"] = "Excelente control en la fase de bajada."

    metrics = {
        "velocidad_subida_usuario": user_concentric_avg,
        "velocidad_subida_experto": expert_concentric_avg,
        "ratio_subida": concentric_ratio,
        "velocidad_bajada_usuario": user_eccentric_avg,
        "velocidad_bajada_experto": expert_eccentric_avg,
        "ratio_bajada": eccentric_ratio,
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_scapular_stability(user_data, expert_data, exercise_config):
    """
    Analiza la estabilidad de la cintura escapular. REVISADO - Parece correcto.
    """
    # Obtener factor de sensibilidad para estabilidad escapular
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

        # APLICAR SENSIBILIDAD al umbral
        stability_threshold = apply_sensitivity_to_threshold(
            exercise_config.get("scapular_stability_threshold", 1.5), sensitivity_factor
        )

        # Generar feedback - LÓGICA CORRECTA
        feedback = {}

        if movement_ratio > (2.0 / sensitivity_factor):
            if sensitivity_factor > 1.5:
                feedback["estabilidad_escapular"] = (
                    "Tus hombros se mueven excesivamente durante el press. "
                    "Es crítico mantener una posición mucho más estable de la cintura escapular."
                )
            else:
                feedback["estabilidad_escapular"] = (
                    "Tus hombros se mueven excesivamente durante el press. "
                    "Mantén una posición más estable de la cintura escapular."
                )
        elif asymmetry_ratio > (2.0 / sensitivity_factor):
            feedback["estabilidad_escapular"] = (
                "Se detecta asimetría en el movimiento de tus hombros. "
                "Concéntrate en mantener ambos hombros equilibrados."
            )
        elif movement_ratio > stability_threshold:
            if sensitivity_factor > 1.5:
                feedback["estabilidad_escapular"] = (
                    "Se detecta inestabilidad notable en tu cintura escapular. "
                    "Es importante practicar mantener los hombros en una posición más fija."
                )
            else:
                feedback["estabilidad_escapular"] = (
                    "Se detecta cierta inestabilidad en tu cintura escapular. "
                    "Practica mantener los hombros en una posición más fija."
                )
        else:
            feedback["estabilidad_escapular"] = (
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

        return {"metrics": metrics, "feedback": feedback}

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
        }


def run_exercise_analysis(user_data, expert_data, exercise_name="press_militar"):
    """MEJORA: Ejecuta análisis completo con detección interna de repeticiones."""
    logger.info(f"Iniciando análisis mejorado: {exercise_name}")

    # Obtener configuración del ejercicio
    exercise_config = get_exercise_config(exercise_name)

    # Log de factores de sensibilidad aplicados
    sensitivity_factors = exercise_config.get("sensitivity_factors", {})
    logger.info(f"Factores de sensibilidad: {sensitivity_factors}")

    # MEJORA: Ya no necesitamos pasar repeticiones como parámetro
    logger.info(
        "Usando detección interna de picos temporalmente sincronizada para análisis de abducción"
    )

    # Ejecutar todos los análisis - SIN PASAR REPETICIONES
    amplitude_result = analyze_movement_amplitude(
        user_data, expert_data, exercise_config
    )
    abduction_result = analyze_elbow_abduction_angle(
        user_data,
        expert_data,
        exercise_config,
        # REMOVIDO: user_repetitions, expert_repetitions
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

    # Calcular puntuación global y nivel
    overall_score = calculate_overall_score(all_metrics, exercise_config)
    skill_level = determine_skill_level(overall_score)

    logger.info(
        f"Análisis mejorado completado - Puntuación: {overall_score:.1f}/100 - Nivel: {skill_level}"
    )

    return {
        "metrics": all_metrics,
        "feedback": all_feedback,
        "score": overall_score,
        "level": skill_level,
        "exercise_config": exercise_config,
        "sensitivity_factors": sensitivity_factors,
    }


def generate_analysis_report(analysis_results, exercise_name, output_path=None):
    """Genera un informe completo con los resultados del análisis."""
    # Crear informe
    report = {
        "ejercicio": exercise_name,
        "puntuacion_global": round(analysis_results["score"], 1),
        "nivel": analysis_results["level"],
        "areas_mejora": [],
        "puntos_fuertes": [],
        "feedback_detallado": analysis_results["feedback"],
        "metricas": analysis_results["metrics"],
        "recomendaciones": generate_recommendations(
            analysis_results["feedback"], analysis_results["score"]
        ),
        "sensitivity_factors": analysis_results.get("sensitivity_factors", {}),
        "version_analisis": "deteccion_picos_sincronizada_v1.0",
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
        logger.info(f"Informe mejorado guardado en: {output_path}")

    return report
