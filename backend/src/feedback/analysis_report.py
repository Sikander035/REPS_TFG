# backend/src/feedback/analysis_report.py
import sys
import numpy as np
import pandas as pd
import os
import json
import logging
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.analysis_utils import (
    get_exercise_config,
    calculate_angle,
    calculate_overall_score,
    determine_skill_level,
    generate_recommendations,
)

logger = logging.getLogger(__name__)


def analyze_movement_amplitude(user_data, expert_data, exercise_config):
    """
    Analiza la amplitud del movimiento usando los CODOS como referencia principal.
    Compara el rango de movimiento promedio de ambos codos usuario vs experto.
    """
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
    user_highest_point = np.min(user_elbow_y)  # Posición más alta (extensión completa)
    user_lowest_point = np.max(user_elbow_y)  # Posición más baja (flexión máxima)
    expert_highest_point = np.min(expert_elbow_y)
    expert_lowest_point = np.max(expert_elbow_y)

    # Calcular rango de movimiento (positivo)
    user_rom = user_lowest_point - user_highest_point
    expert_rom = expert_lowest_point - expert_highest_point
    rom_ratio = user_rom / expert_rom if expert_rom > 0 else 0

    # Analizar diferencia en el punto más bajo (flexión máxima de codos)
    bottom_diff = (
        abs(user_lowest_point - expert_lowest_point) / expert_rom
        if expert_rom > 0
        else 0
    )

    # Generar feedback específico para press militar
    feedback = {}
    if rom_ratio < exercise_config["rom_threshold"]:
        feedback["amplitud"] = (
            "Tu rango de movimiento es limitado. Baja más los codos para una flexión completa "
            "y extiende completamente arriba."
        )
    elif bottom_diff > exercise_config["bottom_diff_threshold"]:
        feedback["posicion_baja"] = (
            "No estás flexionando suficientemente los codos. Baja hasta que los codos "
            "estén aproximadamente a la altura de los hombros."
        )
    elif rom_ratio > 1.15:
        feedback["amplitud"] = (
            "Tu rango de movimiento es excesivo. Controla la bajada para evitar "
            "hiperextensión de los hombros."
        )
    else:
        feedback["amplitud"] = "Excelente amplitud de movimiento en los codos."

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
    Analiza el ángulo de abducción de los codos (ángulo con la horizontal).
    Determina si los codos están más abiertos o cerrados que el experto.
    """
    user_right_abduction = []
    user_left_abduction = []
    expert_right_abduction = []
    expert_left_abduction = []

    # Calcular ángulo de abducción para cada frame
    for i in range(len(user_data)):
        try:
            # === USUARIO - CODO DERECHO ===
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

            # === USUARIO - CODO IZQUIERDO ===
            user_left_elbow = [
                user_data.iloc[i]["landmark_left_elbow_x"],
                user_data.iloc[i]["landmark_left_elbow_y"],
                user_data.iloc[i]["landmark_left_elbow_z"],
            ]

            # === EXPERTO - CODO DERECHO ===
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

            # === EXPERTO - CODO IZQUIERDO ===
            expert_left_elbow = [
                expert_data.iloc[i]["landmark_left_elbow_x"],
                expert_data.iloc[i]["landmark_left_elbow_y"],
                expert_data.iloc[i]["landmark_left_elbow_z"],
            ]

            # Calcular ángulos de abducción usando la línea de hombros como referencia horizontal
            # Ángulo entre vector hombro->codo y línea de hombros

            # USUARIO
            user_right_angle = calculate_angle(
                user_left_shoulder, user_right_shoulder, user_right_elbow
            )
            user_left_angle = calculate_angle(
                user_right_shoulder, user_left_shoulder, user_left_elbow
            )

            # EXPERTO
            expert_right_angle = calculate_angle(
                expert_left_shoulder, expert_right_shoulder, expert_right_elbow
            )
            expert_left_angle = calculate_angle(
                expert_right_shoulder, expert_left_shoulder, expert_left_elbow
            )

            # Verificar que no hay valores NaN
            if (
                not np.isnan(user_right_angle)
                and not np.isnan(user_left_angle)
                and not np.isnan(expert_right_angle)
                and not np.isnan(expert_left_angle)
            ):

                user_right_abduction.append(user_right_angle)
                user_left_abduction.append(user_left_angle)
                expert_right_abduction.append(expert_right_angle)
                expert_left_abduction.append(expert_left_angle)

        except Exception as e:
            logger.warning(f"Error al calcular ángulo de abducción en frame {i}: {e}")

    if (
        not user_right_abduction
        or not user_left_abduction
        or not expert_right_abduction
        or not expert_left_abduction
    ):
        logger.warning("No se pudieron calcular ángulos de abducción válidos")
        return {
            "metrics": {},
            "feedback": {
                "abduccion_codos": "No se pudo analizar la abducción de los codos."
            },
        }

    # Convertir a arrays
    user_right_abduction = np.array(user_right_abduction)
    user_left_abduction = np.array(user_left_abduction)
    expert_right_abduction = np.array(expert_right_abduction)
    expert_left_abduction = np.array(expert_left_abduction)

    # Calcular métricas promedio
    user_avg_right = np.mean(user_right_abduction)
    user_avg_left = np.mean(user_left_abduction)
    user_avg_abduction = (user_avg_right + user_avg_left) / 2

    expert_avg_right = np.mean(expert_right_abduction)
    expert_avg_left = np.mean(expert_left_abduction)
    expert_avg_abduction = (expert_avg_right + expert_avg_left) / 2

    # Diferencia en abducción
    abduction_diff = user_avg_abduction - expert_avg_abduction

    # Simetría en abducción (diferencia entre brazos)
    user_asymmetry = abs(user_avg_right - user_avg_left)
    expert_asymmetry = abs(expert_avg_right - expert_avg_left)
    asymmetry_ratio = user_asymmetry / expert_asymmetry if expert_asymmetry > 0 else 1

    # Generar feedback
    feedback = {}
    abduction_threshold = exercise_config.get("abduction_angle_threshold", 15)  # grados

    if abs(abduction_diff) > abduction_threshold:
        if abduction_diff > 0:
            feedback["abduccion_codos"] = (
                "Tus codos están demasiado abiertos respecto al cuerpo. "
                "Acércalos un poco más para una posición más segura."
            )
        else:
            feedback["abduccion_codos"] = (
                "Tus codos están muy cerrados respecto al cuerpo. "
                "Sepáralos un poco más para una mejor mecánica del movimiento."
            )
    elif asymmetry_ratio > 2.0:
        feedback["abduccion_codos"] = (
            "Hay asimetría en la abducción de tus codos. "
            "Intenta mantener ambos codos en la misma posición respecto al cuerpo."
        )
    else:
        feedback["abduccion_codos"] = "Excelente posición de codos respecto al cuerpo."

    metrics = {
        "abduccion_usuario_derecha": user_avg_right,
        "abduccion_usuario_izquierda": user_avg_left,
        "abduccion_usuario_promedio": user_avg_abduction,
        "abduccion_experto_derecha": expert_avg_right,
        "abduccion_experto_izquierda": expert_avg_left,
        "abduccion_experto_promedio": expert_avg_abduction,
        "diferencia_abduccion": abduction_diff,
        "asimetria_abduccion_usuario": user_asymmetry,
        "asimetria_abduccion_experto": expert_asymmetry,
        "ratio_asimetria_abduccion": asymmetry_ratio,
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_symmetry(user_data, expert_data, exercise_config):
    """Analiza la simetría entre el lado izquierdo y derecho."""
    # Extraer posiciones verticales de codos (mejorado de muñecas)
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

    # Generar feedback
    feedback = {}
    if (
        asymmetry_ratio > 2.0
    ):  # Usuario es significativamente más asimétrico que experto
        feedback["simetria"] = (
            "Hay una asimetría notable entre tu lado derecho e izquierdo. "
            "Enfócate en levantar ambos brazos por igual."
        )
    elif normalized_diff > exercise_config["symmetry_threshold"]:
        feedback["simetria"] = (
            "Se detecta cierta asimetría en el movimiento. "
            "Intenta mantener ambos codos a la misma altura."
        )
    else:
        feedback["simetria"] = "Excelente simetría bilateral en el movimiento."

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
    Analiza la trayectoria 3D completa del movimiento.
    Evalúa desviaciones laterales, frontales y consistencia del movimiento.
    """
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

    # === ANÁLISIS 1: DESVIACIÓN LATERAL (X) ===
    lateral_deviation_user = np.std(user_x)
    lateral_deviation_expert = np.std(expert_x)
    lateral_deviation_ratio = (
        lateral_deviation_user / lateral_deviation_expert
        if lateral_deviation_expert > 0
        else 1
    )

    # === ANÁLISIS 2: DESVIACIÓN FRONTAL/POSTERIOR (Z) ===
    frontal_deviation_user = np.std(user_z)
    frontal_deviation_expert = np.std(expert_z)
    frontal_deviation_ratio = (
        frontal_deviation_user / frontal_deviation_expert
        if frontal_deviation_expert > 0
        else 1
    )

    # === ANÁLISIS 3: VERTICALIDAD DEL MOVIMIENTO ===
    # Calcular cuánto se desvía de una línea perfectamente vertical
    try:
        # Pendiente de la trayectoria Y vs X (debería ser cercana a 0 para movimiento vertical)
        slope_x_user, _, r_value_x_user, _, _ = stats.linregress(user_y, user_x)
        slope_z_user, _, r_value_z_user, _, _ = stats.linregress(user_y, user_z)

        slope_x_expert, _, r_value_x_expert, _, _ = stats.linregress(expert_y, expert_x)
        slope_z_expert, _, r_value_z_expert, _, _ = stats.linregress(expert_y, expert_z)

        # Comparar verticalidad con experto
        verticality_x_ratio = (
            abs(slope_x_user) / abs(slope_x_expert) if abs(slope_x_expert) > 0 else 1
        )
        verticality_z_ratio = (
            abs(slope_z_user) / abs(slope_z_expert) if abs(slope_z_expert) > 0 else 1
        )

    except:
        verticality_x_ratio = verticality_z_ratio = 1
        r_value_x_user = r_value_z_user = 0

    # === ANÁLISIS 4: CONSISTENCIA DE LA TRAYECTORIA ===
    # Diferencias frame a frame con el experto
    trajectory_diff_x = np.mean(np.abs(user_x - expert_x))
    trajectory_diff_z = np.mean(np.abs(user_z - expert_z))

    # Normalizar por rango de movimiento
    shoulder_width = np.mean(
        np.abs(
            user_data["landmark_right_shoulder_x"].values
            - user_data["landmark_left_shoulder_x"].values
        )
    )
    normalized_trajectory_diff_x = (
        trajectory_diff_x / shoulder_width if shoulder_width > 0 else 0
    )
    normalized_trajectory_diff_z = (
        trajectory_diff_z / shoulder_width if shoulder_width > 0 else 0
    )

    # === GENERAR FEEDBACK ===
    feedback = {}

    # Feedback para desviación lateral
    if lateral_deviation_ratio > 2.0:
        feedback["trayectoria_lateral"] = (
            "Tu movimiento se desvía demasiado lateralmente. "
            "Concéntrate en mantener las muñecas en línea vertical."
        )
    elif normalized_trajectory_diff_x > exercise_config["lateral_dev_threshold"]:
        feedback["trayectoria_lateral"] = (
            "Se detecta cierta desviación lateral en tu trayectoria. "
            "Intenta mantener un movimiento más vertical."
        )
    else:
        feedback["trayectoria_lateral"] = "Excelente control lateral del movimiento."

    # Feedback para desviación frontal
    if frontal_deviation_ratio > 2.0:
        feedback["trayectoria_frontal"] = (
            "Tu movimiento se desvía hacia adelante/atrás. "
            "Mantén las muñecas en un plano vertical consistente."
        )
    elif normalized_trajectory_diff_z > exercise_config.get(
        "frontal_dev_threshold", 0.15
    ):
        feedback["trayectoria_frontal"] = (
            "Se detecta cierta desviación frontal en tu movimiento."
        )
    else:
        feedback["trayectoria_frontal"] = "Buen control frontal del movimiento."

    # Feedback general de trayectoria
    if max(lateral_deviation_ratio, frontal_deviation_ratio) < 1.5:
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
        "verticalidad_x_ratio": verticality_x_ratio,
        "verticalidad_z_ratio": verticality_z_ratio,
        "diferencia_trayectoria_x": normalized_trajectory_diff_x,
        "diferencia_trayectoria_z": normalized_trajectory_diff_z,
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_speed(user_data, expert_data, exercise_config):
    """Analiza la velocidad de ejecución en las fases concéntrica y excéntrica."""
    # Usar promedio de codos para velocidad (más relevante que muñecas)
    user_elbow_y = (
        user_data["landmark_right_elbow_y"].values
        + user_data["landmark_left_elbow_y"].values
    ) / 2
    expert_elbow_y = (
        expert_data["landmark_right_elbow_y"].values
        + expert_data["landmark_left_elbow_y"].values
    ) / 2

    # Calcular velocidades (derivada de la posición)
    user_velocity = np.gradient(user_elbow_y)
    expert_velocity = np.gradient(expert_elbow_y)

    # Separar fases concéntrica (valores negativos - subida) y excéntrica (valores positivos - bajada)
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

    # Generar feedback
    feedback = {}
    velocity_threshold = exercise_config["velocity_ratio_threshold"]

    if concentric_ratio < (1 - velocity_threshold):
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
    Analiza la estabilidad de la cintura escapular durante el movimiento.
    TEMPORALMENTE SIMPLIFICADO para evitar errores de NumPy.
    """
    try:
        # VERSIÓN SIMPLIFICADA - Solo analizar movimiento vertical de hombros
        user_r_shoulder_y = user_data["landmark_right_shoulder_y"].values
        user_l_shoulder_y = user_data["landmark_left_shoulder_y"].values
        expert_r_shoulder_y = expert_data["landmark_right_shoulder_y"].values
        expert_l_shoulder_y = expert_data["landmark_left_shoulder_y"].values

        # Calcular centro de hombros
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

        # Generar feedback basado en movimiento y asimetría
        feedback = {}

        if movement_ratio > 2.0:
            feedback["estabilidad_escapular"] = (
                "Tus hombros se mueven excesivamente durante el press. "
                "Mantén una posición más estable de la cintura escapular."
            )
        elif asymmetry_ratio > 2.0:
            feedback["estabilidad_escapular"] = (
                "Se detecta asimetría en el movimiento de tus hombros. "
                "Concéntrate en mantener ambos hombros equilibrados."
            )
        elif movement_ratio > 1.5:
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
        # Fallback completo - devolver valores por defecto
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
    """Ejecuta análisis completo del ejercicio con las mejoras implementadas."""
    logger.info(f"Iniciando análisis mejorado del ejercicio: {exercise_name}")

    # Obtener configuración específica del ejercicio
    exercise_config = get_exercise_config(exercise_name)

    # Ejecutar todos los análisis mejorados
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
    }


def generate_analysis_report(analysis_results, exercise_name, output_path=None):
    """Genera un informe completo con los resultados del análisis mejorado."""
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
        "version_analisis": "mejorada_v2.0",  # Indicador de la versión mejorada
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
