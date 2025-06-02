# backend/src/feedback/universal_metrics.py
import sys
import os
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.analysis_utils import (
    apply_unified_sensitivity,
    calculate_deviation_score,
    apply_sensitivity_to_threshold,
)
from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


def analyze_movement_amplitude_universal(
    user_data, expert_data, exercise_config, landmarks_config, config_path="config.json"
):
    """
    UNIVERSAL: Análisis de amplitud extraído del código actual del press militar.
    Solo cambian los landmarks, la lógica es EXACTAMENTE la misma.
    """
    # Obtener exercise_name del config_path (extraer de exercise_config si es posible)
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "amplitud", exercise_name, config_path
    )

    # Extraer landmarks de la configuración
    landmark_left = landmarks_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = landmarks_config.get("right_landmark", "landmark_right_elbow")
    axis = landmarks_config.get("axis", "y")
    movement_direction = landmarks_config.get("movement_direction", "up")
    feedback_context = landmarks_config.get("feedback_context", "codos")

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - calcular promedio de landmarks
    user_signal = (
        user_data[f"{landmark_left}_{axis}"].values
        + user_data[f"{landmark_right}_{axis}"].values
    ) / 2
    expert_signal = (
        expert_data[f"{landmark_left}_{axis}"].values
        + expert_data[f"{landmark_right}_{axis}"].values
    ) / 2

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - En MediaPipe: Y=0 arriba, Y=1 abajo
    user_highest_point = np.min(user_signal)
    user_lowest_point = np.max(user_signal)
    expert_highest_point = np.min(expert_signal)
    expert_lowest_point = np.max(expert_signal)

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Calcular rango de movimiento
    user_rom = user_lowest_point - user_highest_point
    expert_rom = expert_lowest_point - expert_highest_point
    rom_ratio = user_rom / expert_rom if expert_rom > 0 else 0

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Analizar diferencia en el punto más bajo
    bottom_diff = (
        abs(user_lowest_point - expert_lowest_point) / expert_rom
        if expert_rom > 0
        else 0
    )

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Score y sensibilidad
    # Obtener penalty de configuración
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="amplitude",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        rom_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(base_score, sensitivity_factor, "amplitud")

    # Obtener umbrales usando config_manager
    rom_threshold = config_manager.get_analysis_threshold(
        "rom_threshold", exercise_name, config_path
    )
    bottom_diff_threshold = config_manager.get_analysis_threshold(
        "bottom_diff_threshold", exercise_name, config_path
    )

    # FEEDBACK ADAPTADO - mantener lógica pero cambiar contexto y KEY EN INGLÉS
    feedback = {}
    if final_score >= 85:
        feedback["amplitude"] = (
            f"Excelente amplitud de movimiento en los {feedback_context}."
        )
    elif final_score >= 70:
        if rom_ratio > 1.15:
            if movement_direction == "down":  # Sentadilla
                feedback["amplitude"] = (
                    f"Tu rango de movimiento es excesivo. Controla la bajada para evitar "
                    f"hiperflexión de los {feedback_context}."
                )
            else:  # Press, dominada
                feedback["amplitude"] = (
                    f"Tu rango de movimiento es excesivo. Controla la bajada para evitar "
                    f"hiperextensión de los {feedback_context}."
                )
        else:
            if movement_direction == "down":
                feedback["amplitude"] = (
                    f"Tu rango de movimento podría ser más amplio. Baja más los {feedback_context}."
                )
            else:
                feedback["amplitude"] = (
                    f"Tu rango de movimento podría ser más amplio. Baja hasta que las mancuernas "
                    f"estén aproximadamente a la altura de los hombros."
                )
    elif final_score >= 50:
        # LÓGICA EXACTA DEL CÓDIGO ACTUAL
        rom_threshold_adj = apply_sensitivity_to_threshold(
            rom_threshold, sensitivity_factor
        )
        bottom_diff_threshold_adj = apply_sensitivity_to_threshold(
            bottom_diff_threshold, sensitivity_factor
        )

        if rom_ratio > 1.25:
            feedback["amplitude"] = (
                f"Tu rango de movimiento es excesivamente amplio. Es crítico "
                f"controlar la bajada para evitar hiperextensión de los {feedback_context}."
            )
        elif bottom_diff > bottom_diff_threshold_adj * 1.5:
            feedback["amplitude"] = (
                f"Tu rango de movimiento es insuficiente. Es importante bajar hasta que "
                f"los {feedback_context} lleguen a la posición correcta para técnica adecuada."
            )
        else:
            feedback["amplitude"] = (
                f"Tu rango de movimiento es limitado. Baja más los {feedback_context} para una flexión completa "
                f"y extiende completamente arriba."
            )
    else:
        # Casos críticos
        if rom_ratio > 1.25:
            feedback["amplitude"] = (
                f"Tu rango de movimiento es excesivamente amplio. Es crítico "
                f"controlar la bajada para evitar hiperextensión de los {feedback_context}."
            )
        else:
            feedback["amplitude"] = (
                f"Tu rango de movimiento es significativamente limitado. Es crítico "
                f"trabajar en la amplitud completa: baja más los {feedback_context} y extiende completamente arriba."
            )

    # MÉTRICAS EXACTAS DEL CÓDIGO ACTUAL
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

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_symmetry_universal(
    user_data, expert_data, exercise_config, landmarks_config, config_path="config.json"
):
    """
    UNIVERSAL: Análisis de simetría extraído del código actual del press militar.
    Solo cambian los landmarks, la lógica es EXACTAMENTE la misma.
    """
    # Obtener exercise_name del config_path (extraer de exercise_config si es posible)
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "simetria", exercise_name, config_path
    )

    # Extraer landmarks de la configuración
    landmark_left = landmarks_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = landmarks_config.get("right_landmark", "landmark_right_elbow")
    axis = landmarks_config.get("axis", "y")
    feedback_context = landmarks_config.get("feedback_context", "movimiento")

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Extraer posiciones verticales
    user_r_signal = user_data[f"{landmark_right}_{axis}"].values
    user_l_signal = user_data[f"{landmark_left}_{axis}"].values

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Calcular diferencia promedio entre lados
    height_diff = np.mean(np.abs(user_r_signal - user_l_signal))

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Normalizar diferencia respecto al rango de movimiento promedio
    user_range = np.max(user_r_signal) - np.min(user_r_signal)
    normalized_diff = height_diff / user_range if user_range > 0 else 0

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Comparar con simetría del experto
    expert_r_signal = expert_data[f"{landmark_right}_{axis}"].values
    expert_l_signal = expert_data[f"{landmark_left}_{axis}"].values
    expert_height_diff = np.mean(np.abs(expert_r_signal - expert_l_signal))
    expert_range = np.max(expert_r_signal) - np.min(expert_r_signal)
    expert_normalized_diff = (
        expert_height_diff / expert_range if expert_range > 0 else 0
    )

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Ratio de asimetría usuario vs experto
    asymmetry_ratio = (
        normalized_diff / expert_normalized_diff if expert_normalized_diff > 0 else 1
    )

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Score y sensibilidad
    # Obtener penalty de configuración
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="symmetry",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        asymmetry_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(base_score, sensitivity_factor, "simetria")

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Feedback con KEY EN INGLÉS
    symmetry_threshold = config_manager.get_analysis_threshold(
        "symmetry_threshold", exercise_name, config_path
    )
    symmetry_threshold_adj = apply_sensitivity_to_threshold(
        symmetry_threshold, sensitivity_factor
    )

    feedback = {}
    if asymmetry_ratio > (1.8 / sensitivity_factor):
        if sensitivity_factor > 1.5:
            feedback["symmetry"] = (
                f"Hay una asimetría muy notable entre tu lado derecho e izquierdo en el {feedback_context}. "
                f"Es prioritario trabajar en equilibrar ambos brazos."
            )
        else:
            feedback["symmetry"] = (
                f"Hay una asimetría notable entre tu lado derecho e izquierdo en el {feedback_context}. "
                f"Enfócate en levantar ambos brazos por igual."
            )
    elif normalized_diff > symmetry_threshold_adj:
        if sensitivity_factor > 1.5 and normalized_diff > symmetry_threshold_adj * 1.5:
            feedback["symmetry"] = (
                f"Se detecta asimetría significativa en el {feedback_context}. "
                f"Es importante trabajar en mantener ambos lados a la misma altura."
            )
        else:
            feedback["symmetry"] = (
                f"Se detecta cierta asimetría en el {feedback_context}. "
                f"Intenta mantener ambos lados a la misma altura."
            )
    else:
        feedback["symmetry"] = f"Excelente simetría bilateral en el {feedback_context}."

    # MÉTRICAS EXACTAS DEL CÓDIGO ACTUAL
    metrics = {
        "diferencia_altura": height_diff,
        "diferencia_normalizada": normalized_diff,
        "diferencia_experto_normalizada": expert_normalized_diff,
        "ratio_asimetria": asymmetry_ratio,
        "rango_movimiento_usuario": user_range,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_movement_trajectory_3d_universal(
    user_data, expert_data, exercise_config, landmarks_config, config_path="config.json"
):
    """
    UNIVERSAL: Análisis de trayectoria extraído del código actual del press militar.
    Solo cambian los landmarks, la lógica es EXACTAMENTE la misma.
    """
    # Obtener exercise_name del config_path (extraer de exercise_config si es posible)
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "trayectoria", exercise_name, config_path
    )

    # Extraer landmarks de la configuración
    landmark_left = landmarks_config.get("left_landmark", "landmark_left_wrist")
    landmark_right = landmarks_config.get("right_landmark", "landmark_right_wrist")

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Usar promedio de ambas muñecas/landmarks para mayor estabilidad
    user_x = (
        user_data[f"{landmark_right}_x"].values + user_data[f"{landmark_left}_x"].values
    ) / 2
    user_y = (
        user_data[f"{landmark_right}_y"].values + user_data[f"{landmark_left}_y"].values
    ) / 2
    user_z = (
        user_data[f"{landmark_right}_z"].values + user_data[f"{landmark_left}_z"].values
    ) / 2

    expert_x = (
        expert_data[f"{landmark_right}_x"].values
        + expert_data[f"{landmark_left}_x"].values
    ) / 2
    expert_y = (
        expert_data[f"{landmark_right}_y"].values
        + expert_data[f"{landmark_left}_y"].values
    ) / 2
    expert_z = (
        expert_data[f"{landmark_right}_z"].values
        + expert_data[f"{landmark_left}_z"].values
    ) / 2

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Análisis de desviaciones
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

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Score
    worst_deviation_ratio = max(lateral_deviation_ratio, frontal_deviation_ratio)
    # Obtener penalty de configuración
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="trajectory",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        worst_deviation_ratio, 1.0, max_penalty=max_penalty, metric_type="ratio"
    )
    final_score = apply_unified_sensitivity(
        base_score, sensitivity_factor, "trayectoria"
    )

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Feedback con KEYS EN INGLÉS
    lateral_threshold = config_manager.get_analysis_threshold(
        "lateral_dev_threshold", exercise_name, config_path
    )
    frontal_threshold = config_manager.get_analysis_threshold(
        "frontal_dev_threshold", exercise_name, config_path
    )

    lateral_threshold_adj = apply_sensitivity_to_threshold(
        lateral_threshold, sensitivity_factor
    )
    frontal_threshold_adj = apply_sensitivity_to_threshold(
        frontal_threshold, sensitivity_factor
    )

    feedback = {}

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Diferencias directas con experto
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

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Evaluar lateral
    if lateral_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trajectory_lateral"] = (
            "Tu movimiento se desvía excesivamente en dirección lateral. "
            "Concéntrate urgentemente en mantener las muñecas en línea vertical."
        )
    elif normalized_trajectory_diff_x > lateral_threshold_adj:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_x > lateral_threshold_adj * 1.5
        ):
            feedback["trajectory_lateral"] = (
                "Se detecta desviación lateral significativa en tu trayectoria. "
                "Es importante corregir para mantener un movimiento más vertical."
            )
        else:
            feedback["trajectory_lateral"] = (
                "Se detecta cierta desviación lateral en tu trayectoria. "
                "Intenta mantener un movimiento más vertical."
            )
    else:
        feedback["trajectory_lateral"] = "Excelente control lateral del movimiento."

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Evaluar frontal
    if frontal_deviation_ratio > (2.0 / sensitivity_factor):
        feedback["trajectory_frontal"] = (
            "Tu movimiento se desvía hacia adelante/atrás significativamente. "
            "Mantén las muñecas en un plano vertical consistente."
        )
    elif normalized_trajectory_diff_z > frontal_threshold_adj:
        if (
            sensitivity_factor > 1.5
            and normalized_trajectory_diff_z > frontal_threshold_adj * 1.5
        ):
            feedback["trajectory_frontal"] = (
                "Se detecta desviación frontal significativa en tu movimiento. "
                "Es importante mantener un plano vertical más consistente."
            )
        else:
            feedback["trajectory_frontal"] = (
                "Se detecta cierta desviación frontal en tu movimiento."
            )
    else:
        feedback["trajectory_frontal"] = "Buen control frontal del movimiento."

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Feedback general
    if max(lateral_deviation_ratio, frontal_deviation_ratio) < (
        1.5 / sensitivity_factor
    ):
        feedback["trajectory"] = "Excelente trayectoria 3D del movimiento."
    else:
        feedback["trajectory"] = "La trayectoria del movimiento puede mejorarse."

    # MÉTRICAS EXACTAS DEL CÓDIGO ACTUAL
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

    return {"metrics": metrics, "feedback": feedback, "score": final_score}


def analyze_speed_universal(
    user_data, expert_data, exercise_config, landmarks_config, config_path="config.json"
):
    """
    UNIVERSAL: Análisis de velocidad extraído del código actual del press militar.
    Solo cambian los landmarks, la lógica es EXACTAMENTE la misma.
    """
    # Obtener exercise_name del config_path (extraer de exercise_config si es posible)
    exercise_name = exercise_config.get("_exercise_name", "unknown")

    # Obtener factor de sensibilidad usando config_manager
    sensitivity_factor = config_manager.get_sensitivity_factor(
        "velocidad", exercise_name, config_path
    )

    # Extraer landmarks de la configuración
    landmark_left = landmarks_config.get("left_landmark", "landmark_left_elbow")
    landmark_right = landmarks_config.get("right_landmark", "landmark_right_elbow")
    axis = landmarks_config.get("axis", "y")
    movement_direction = landmarks_config.get("movement_direction", "up")

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Usar promedio de codos/landmarks para velocidad
    user_signal = (
        user_data[f"{landmark_right}_{axis}"].values
        + user_data[f"{landmark_left}_{axis}"].values
    ) / 2
    expert_signal = (
        expert_data[f"{landmark_right}_{axis}"].values
        + expert_data[f"{landmark_left}_{axis}"].values
    ) / 2

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Calcular velocidades
    user_velocity = np.gradient(user_signal)
    expert_velocity = np.gradient(expert_signal)

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Separar fases
    if movement_direction == "down":  # Sentadilla
        user_concentric = user_velocity[
            user_velocity > 0
        ]  # Subida = velocidad positiva
        user_eccentric = user_velocity[user_velocity < 0]  # Bajada = velocidad negativa
        expert_concentric = expert_velocity[expert_velocity > 0]
        expert_eccentric = expert_velocity[expert_velocity < 0]
    else:  # Press, dominada (movement_direction == "up")
        user_concentric = user_velocity[
            user_velocity < 0
        ]  # Subida = velocidad negativa
        user_eccentric = user_velocity[user_velocity > 0]  # Bajada = velocidad positiva
        expert_concentric = expert_velocity[expert_velocity < 0]
        expert_eccentric = expert_velocity[expert_velocity > 0]

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Calcular velocidades promedio
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

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Calcular ratios
    concentric_ratio = (
        user_concentric_avg / expert_concentric_avg if expert_concentric_avg > 0 else 1
    )
    eccentric_ratio = (
        user_eccentric_avg / expert_eccentric_avg if expert_eccentric_avg > 0 else 1
    )

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Score
    concentric_deviation = abs(concentric_ratio - 1.0)
    eccentric_deviation = abs(eccentric_ratio - 1.0)
    worst_velocity_deviation = max(concentric_deviation, eccentric_deviation)

    # Obtener penalty de configuración
    max_penalty = config_manager.get_penalty_config(
        exercise_name="",
        metric_type="universal",
        metric_name="speed",
        config_path=config_path,
    )
    base_score = calculate_deviation_score(
        worst_velocity_deviation, 0, max_penalty=max_penalty, metric_type="linear"
    )
    final_score = apply_unified_sensitivity(base_score, sensitivity_factor, "velocidad")

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Feedback con KEYS EN INGLÉS
    velocity_threshold = config_manager.get_analysis_threshold(
        "velocity_ratio_threshold", exercise_name, config_path
    )
    velocity_threshold_adj = apply_sensitivity_to_threshold(
        velocity_threshold, sensitivity_factor
    )

    feedback = {}

    # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Feedback unificado basado en score final
    if final_score >= 85:
        feedback["speed_concentric"] = "Excelente velocidad en la fase de subida."
        feedback["speed_eccentric"] = "Excelente control en la fase de bajada."
    elif final_score >= 70:
        # Determinar cuál fase es problemática para feedback específico
        if concentric_deviation > eccentric_deviation:
            # Problema principal en fase concéntrica
            if concentric_ratio < (1 - velocity_threshold_adj):
                feedback["speed_concentric"] = (
                    "La fase de subida es moderadamente lenta. "
                    "Intenta ser más explosivo en la fase concéntrica."
                )
            else:
                feedback["speed_concentric"] = (
                    "La fase de subida es moderadamente rápida. "
                    "Controla un poco más el movimiento."
                )
            feedback["speed_eccentric"] = "Buen control en la fase de bajada."
        else:
            # Problema principal en fase excéntrica
            feedback["speed_concentric"] = "Buena velocidad en la fase de subida."
            if eccentric_ratio > (1 + velocity_threshold_adj):
                feedback["speed_eccentric"] = (
                    "La fase de bajada es moderadamente rápida. "
                    "Intenta controlar más el descenso."
                )
            else:
                feedback["speed_eccentric"] = (
                    "La fase de bajada es moderadamente lenta. "
                    "Controla el descenso pero no lo ralentices en exceso."
                )
    elif final_score >= 50:
        # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Casos moderados-críticos
        if concentric_ratio < (1 - velocity_threshold_adj):
            feedback["speed_concentric"] = (
                "La fase de subida es demasiado lenta comparada con el experto. "
                "Intenta ser más explosivo en la fase concéntrica."
            )
        elif concentric_ratio > (1 + velocity_threshold_adj):
            feedback["speed_concentric"] = (
                "La fase de subida es demasiado rápida. "
                "Controla más el movimiento para mejor técnica."
            )
        else:
            feedback["speed_concentric"] = "Velocidad de subida aceptable."

        if eccentric_ratio < (1 - velocity_threshold_adj):
            feedback["speed_eccentric"] = (
                "La fase de bajada es demasiado lenta. "
                "Controla el descenso pero no lo ralentices en exceso."
            )
        elif eccentric_ratio > (1 + velocity_threshold_adj):
            feedback["speed_eccentric"] = (
                "La fase de bajada es demasiado rápida. "
                "Intenta controlar más el descenso para mejor técnica."
            )
        else:
            feedback["speed_eccentric"] = "Control de bajada aceptable."
    else:
        # LÓGICA EXACTA DEL CÓDIGO ACTUAL - Casos críticos
        if concentric_ratio < (1 - velocity_threshold_adj):
            if sensitivity_factor > 1.5:
                feedback["speed_concentric"] = (
                    "La fase de subida es significativamente muy lenta comparada con el experto. "
                    "Es importante ser más explosivo en la fase concéntrica."
                )
            else:
                feedback["speed_concentric"] = (
                    "La fase de subida es demasiado lenta comparada con el experto. "
                    "Intenta ser más explosivo en la fase concéntrica."
                )
        elif concentric_ratio > (1 + velocity_threshold_adj):
            feedback["speed_concentric"] = (
                "La fase de subida es excesivamente rápida. "
                "Es crítico controlar más el movimiento para técnica segura."
            )
        else:
            feedback["speed_concentric"] = "Velocidad de subida problemática."

        if eccentric_ratio > (1 + velocity_threshold_adj):
            if sensitivity_factor > 1.5:
                feedback["speed_eccentric"] = (
                    "La fase de bajada es significativamente muy rápida. "
                    "Es crítico controlar más el descenso para técnica segura."
                )
            else:
                feedback["speed_eccentric"] = (
                    "La fase de bajada es demasiado rápida. "
                    "Intenta controlar más el descenso."
                )
        elif eccentric_ratio < (1 - velocity_threshold_adj):
            feedback["speed_eccentric"] = (
                "La fase de bajada es excesivamente lenta. "
                "Encuentra un mejor equilibrio en el control del descenso."
            )
        else:
            feedback["speed_eccentric"] = "Control de bajada problemático."

    # MÉTRICAS EXACTAS DEL CÓDIGO ACTUAL
    metrics = {
        "velocidad_subida_usuario": user_concentric_avg,
        "velocidad_subida_experto": expert_concentric_avg,
        "ratio_subida": concentric_ratio,
        "velocidad_bajada_usuario": user_eccentric_avg,
        "velocidad_bajada_experto": expert_eccentric_avg,
        "ratio_bajada": eccentric_ratio,
    }

    return {"metrics": metrics, "feedback": feedback, "score": final_score}
