# backend/src/feedback/analysis_report.py
import sys
import numpy as np
import pandas as pd
import os
import json
import logging

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
    """Analiza la amplitud del movimiento, comparando puntos máximos y mínimos."""
    # Extraer posición vertical (y) de las muñecas
    user_r_wrist_y = user_data["landmark_right_wrist_y"].values
    expert_r_wrist_y = expert_data["landmark_right_wrist_y"].values

    # En MediaPipe: Y=0 arriba, Y=1 abajo
    user_highest_point = np.min(user_r_wrist_y)
    user_lowest_point = np.max(user_r_wrist_y)
    expert_highest_point = np.min(expert_r_wrist_y)
    expert_lowest_point = np.max(expert_r_wrist_y)

    # Calcular rango de movimiento (positivo)
    user_rom = user_lowest_point - user_highest_point
    expert_rom = expert_lowest_point - expert_highest_point
    rom_ratio = user_rom / expert_rom if expert_rom > 0 else 0

    # Analizar diferencia en el punto más bajo (normalizado)
    bottom_diff = (
        abs(user_lowest_point - expert_lowest_point) / expert_rom
        if expert_rom > 0
        else 0
    )

    # Generar feedback
    feedback = {}
    if rom_ratio < exercise_config["rom_threshold"]:
        feedback["amplitud"] = (
            "Tu rango de movimiento es insuficiente. Intenta bajar más las pesas y subirlas más arriba."
        )
    elif bottom_diff > exercise_config["bottom_diff_threshold"]:
        feedback["posicion_baja"] = (
            "No estás bajando las pesas lo suficiente. Asegúrate de llegar hasta la altura de los hombros."
        )
    elif rom_ratio > 1.15:
        feedback["amplitud"] = (
            "Tu rango de movimiento es excesivo, lo que puede causar estrés en los hombros."
        )
    else:
        feedback["amplitud"] = "Buen rango de movimiento."

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


def analyze_elbow_angles(user_data, expert_data, exercise_config):
    """Analiza los ángulos de los codos durante el movimiento."""
    user_angles = []
    expert_angles = []

    # Calcular ángulos para cada frame
    for i in range(len(user_data)):
        try:
            # Extraer coordenadas para el codo derecho (hombro-codo-muñeca)
            user_shoulder = [
                user_data.iloc[i]["landmark_right_shoulder_x"],
                user_data.iloc[i]["landmark_right_shoulder_y"],
                user_data.iloc[i]["landmark_right_shoulder_z"],
            ]
            user_elbow = [
                user_data.iloc[i]["landmark_right_elbow_x"],
                user_data.iloc[i]["landmark_right_elbow_y"],
                user_data.iloc[i]["landmark_right_elbow_z"],
            ]
            user_wrist = [
                user_data.iloc[i]["landmark_right_wrist_x"],
                user_data.iloc[i]["landmark_right_wrist_y"],
                user_data.iloc[i]["landmark_right_wrist_z"],
            ]

            expert_shoulder = [
                expert_data.iloc[i]["landmark_right_shoulder_x"],
                expert_data.iloc[i]["landmark_right_shoulder_y"],
                expert_data.iloc[i]["landmark_right_shoulder_z"],
            ]
            expert_elbow = [
                expert_data.iloc[i]["landmark_right_elbow_x"],
                expert_data.iloc[i]["landmark_right_elbow_y"],
                expert_data.iloc[i]["landmark_right_elbow_z"],
            ]
            expert_wrist = [
                expert_data.iloc[i]["landmark_right_wrist_x"],
                expert_data.iloc[i]["landmark_right_wrist_y"],
                expert_data.iloc[i]["landmark_right_wrist_z"],
            ]

            # Verificar que no hay valores NaN
            if (
                np.isnan(user_shoulder).any()
                or np.isnan(user_elbow).any()
                or np.isnan(user_wrist).any()
                or np.isnan(expert_shoulder).any()
                or np.isnan(expert_elbow).any()
                or np.isnan(expert_wrist).any()
            ):
                continue

            user_angle = calculate_angle(user_shoulder, user_elbow, user_wrist)
            expert_angle = calculate_angle(expert_shoulder, expert_elbow, expert_wrist)

            user_angles.append(user_angle)
            expert_angles.append(expert_angle)
        except Exception as e:
            logger.warning(f"Error al calcular ángulo en frame {i}: {e}")

    # Convertir a arrays
    user_angles = np.array(user_angles)
    expert_angles = np.array(expert_angles)

    # Encontrar ángulos mínimos (punto más bajo de press)
    user_min_angle = np.min(user_angles) if len(user_angles) > 0 else 0
    expert_min_angle = np.min(expert_angles) if len(expert_angles) > 0 else 0
    angle_diff = user_min_angle - expert_min_angle

    # Generar feedback
    feedback = {}
    if angle_diff > exercise_config["angle_diff_threshold"]:
        feedback["codos"] = (
            "Tus codos están demasiado abiertos en la posición baja. Intenta flexionarlos más para lograr una mejor mecánica."
        )
    elif angle_diff < -exercise_config["angle_diff_threshold"]:
        feedback["codos"] = (
            "Tus codos están demasiado cerrados. Asegúrate de mantener una flexión adecuada para proteger tus articulaciones."
        )
    else:
        feedback["codos"] = "Buen ángulo de codos durante el ejercicio."

    metrics = {
        "angulo_minimo_usuario": user_min_angle,
        "angulo_minimo_experto": expert_min_angle,
        "diferencia_angulo": angle_diff,
        "angulos_promedio_usuario": np.mean(user_angles) if len(user_angles) > 0 else 0,
        "angulos_promedio_experto": (
            np.mean(expert_angles) if len(expert_angles) > 0 else 0
        ),
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_symmetry(user_data, expert_data, exercise_config):
    """Analiza la simetría entre el lado izquierdo y derecho."""
    # Extraer posiciones verticales de muñecas
    user_r_wrist_y = user_data["landmark_right_wrist_y"].values
    user_l_wrist_y = user_data["landmark_left_wrist_y"].values

    # Calcular diferencia promedio entre lados
    height_diff = np.mean(np.abs(user_r_wrist_y - user_l_wrist_y))

    # Normalizar diferencia respecto al rango de movimiento
    user_range = np.max(user_r_wrist_y) - np.min(user_r_wrist_y)
    normalized_diff = height_diff / user_range if user_range > 0 else 0

    # Generar feedback
    feedback = {}
    if normalized_diff > exercise_config["symmetry_threshold"]:
        feedback["simetria"] = (
            "Hay una asimetría notable entre tu lado derecho e izquierdo. Enfócate en levantar ambos brazos por igual."
        )
    else:
        feedback["simetria"] = "Buena simetría bilateral en el movimiento."

    metrics = {
        "diferencia_altura": height_diff,
        "diferencia_normalizada": normalized_diff,
        "rango_movimiento_usuario": user_range,
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_movement_path(user_data, expert_data, exercise_config):
    """Analiza la trayectoria de las muñecas durante el movimiento."""
    # Extraer trayectoria de las muñecas (vista frontal: x,y)
    user_r_wrist_x = user_data["landmark_right_wrist_x"].values
    expert_r_wrist_x = expert_data["landmark_right_wrist_x"].values

    # Calcular desviación lateral promedio
    lateral_deviation = np.mean(np.abs(user_r_wrist_x - expert_r_wrist_x))

    # Normalizar respecto al ancho del cuerpo
    shoulder_width = np.mean(
        np.abs(
            user_data["landmark_right_shoulder_x"]
            - user_data["landmark_left_shoulder_x"]
        )
    )
    normalized_deviation = (
        lateral_deviation / shoulder_width if shoulder_width > 0 else 0
    )

    # Generar feedback
    feedback = {}
    if normalized_deviation > exercise_config["lateral_dev_threshold"]:
        feedback["trayectoria"] = (
            "Tu trayectoria se desvía lateralmente. Intenta mantener un movimiento más vertical."
        )
    else:
        feedback["trayectoria"] = "Buena trayectoria vertical del movimiento."

    metrics = {
        "desviacion_lateral": lateral_deviation,
        "desviacion_normalizada": normalized_deviation,
        "ancho_hombros": shoulder_width,
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_speed(user_data, expert_data, exercise_config):
    """Analiza la velocidad de ejecución en las fases concéntrica y excéntrica."""
    # Extraer posiciones verticales
    user_wrist_y = user_data["landmark_right_wrist_y"].values
    expert_wrist_y = expert_data["landmark_right_wrist_y"].values

    # Calcular velocidades (derivada de la posición)
    user_velocity = np.gradient(user_wrist_y)
    expert_velocity = np.gradient(expert_wrist_y)

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
        user_concentric_avg / expert_concentric_avg if expert_concentric_avg > 0 else 0
    )
    eccentric_ratio = (
        user_eccentric_avg / expert_eccentric_avg if expert_eccentric_avg > 0 else 0
    )

    # Generar feedback
    feedback = {}
    velocity_threshold = exercise_config["velocity_ratio_threshold"]

    if concentric_ratio < (1 - velocity_threshold):
        feedback["velocidad_subida"] = (
            "La fase de subida es demasiado lenta. Intenta ser más explosivo en la fase concéntrica."
        )
    elif concentric_ratio > (1 + velocity_threshold):
        feedback["velocidad_subida"] = (
            "La fase de subida es demasiado rápida. Controla más el movimiento."
        )
    else:
        feedback["velocidad_subida"] = "Buena velocidad en la fase de subida."

    if eccentric_ratio < (1 - velocity_threshold):
        feedback["velocidad_bajada"] = (
            "La fase de bajada es demasiado lenta. Controla el descenso pero no lo ralentices en exceso."
        )
    elif eccentric_ratio > (1 + velocity_threshold):
        feedback["velocidad_bajada"] = (
            "La fase de bajada es demasiado rápida. Intenta controlar más el descenso de las pesas."
        )
    else:
        feedback["velocidad_bajada"] = "Buen control en la fase de bajada."

    metrics = {
        "velocidad_subida_usuario": user_concentric_avg,
        "velocidad_subida_experto": expert_concentric_avg,
        "ratio_subida": concentric_ratio,
        "velocidad_bajada_usuario": user_eccentric_avg,
        "velocidad_bajada_experto": expert_eccentric_avg,
        "ratio_bajada": eccentric_ratio,
    }

    return {"metrics": metrics, "feedback": feedback}


def analyze_shoulder_position(user_data, expert_data, exercise_config):
    """Analiza la posición de los hombros para detectar si están elevados o rotados."""
    # Extraer coordenadas verticales de los hombros
    user_r_shoulder_y = user_data["landmark_right_shoulder_y"].values
    user_l_shoulder_y = user_data["landmark_left_shoulder_y"].values
    expert_r_shoulder_y = expert_data["landmark_right_shoulder_y"].values
    expert_l_shoulder_y = expert_data["landmark_left_shoulder_y"].values

    # Calcular altura media de los hombros
    user_shoulder_height = (user_r_shoulder_y + user_l_shoulder_y) / 2
    expert_shoulder_height = (expert_r_shoulder_y + expert_l_shoulder_y) / 2

    # Calcular diferencia respecto a la altura de la cadera
    user_hip_y = (
        user_data["landmark_right_hip_y"].values
        + user_data["landmark_left_hip_y"].values
    ) / 2
    expert_hip_y = (
        expert_data["landmark_right_hip_y"].values
        + expert_data["landmark_left_hip_y"].values
    ) / 2

    # Calcular ratio altura hombros/caderas (menor valor = hombros más elevados)
    user_shoulder_hip_ratio = np.mean(user_shoulder_height / user_hip_y)
    expert_shoulder_hip_ratio = np.mean(expert_shoulder_height / expert_hip_y)

    # Diferencia en ratios
    ratio_diff = user_shoulder_hip_ratio - expert_shoulder_hip_ratio

    # Generar feedback
    feedback = {}
    if ratio_diff < -0.05:  # Usuario tiene hombros más elevados
        feedback["hombros"] = (
            "Tus hombros están demasiado elevados durante el ejercicio. Intenta relajarlos y mantenerlos bajos."
        )
    elif ratio_diff > 0.05:  # Usuario tiene hombros demasiado bajos
        feedback["hombros"] = (
            "Tus hombros están demasiado bajos. Mantén una postura más erguida durante el ejercicio."
        )
    else:
        feedback["hombros"] = "Buena posición de hombros durante el ejercicio."

    metrics = {
        "ratio_hombros_caderas_usuario": user_shoulder_hip_ratio,
        "ratio_hombros_caderas_experto": expert_shoulder_hip_ratio,
        "diferencia_ratio": ratio_diff,
    }

    return {"metrics": metrics, "feedback": feedback}


def run_exercise_analysis(user_data, expert_data, exercise_name="press_militar"):
    """Ejecuta análisis completo del ejercicio."""
    logger.info(f"Iniciando análisis del ejercicio: {exercise_name}")

    # Obtener configuración específica del ejercicio
    exercise_config = get_exercise_config(exercise_name)

    # Ejecutar todos los análisis
    amplitude_result = analyze_movement_amplitude(
        user_data, expert_data, exercise_config
    )
    elbow_result = analyze_elbow_angles(user_data, expert_data, exercise_config)
    symmetry_result = analyze_symmetry(user_data, expert_data, exercise_config)
    path_result = analyze_movement_path(user_data, expert_data, exercise_config)
    speed_result = analyze_speed(user_data, expert_data, exercise_config)
    shoulder_result = analyze_shoulder_position(user_data, expert_data, exercise_config)

    # Combinar métricas
    all_metrics = {
        "amplitud": amplitude_result["metrics"],
        "angulos_codo": elbow_result["metrics"],
        "simetria": symmetry_result["metrics"],
        "trayectoria": path_result["metrics"],
        "velocidad": speed_result["metrics"],
        "hombros": shoulder_result["metrics"],
    }

    # Combinar feedback
    all_feedback = {
        **amplitude_result["feedback"],
        **elbow_result["feedback"],
        **symmetry_result["feedback"],
        **path_result["feedback"],
        **speed_result["feedback"],
        **shoulder_result["feedback"],
    }

    # Calcular puntuación global y nivel
    overall_score = calculate_overall_score(all_metrics, exercise_config)
    skill_level = determine_skill_level(overall_score)

    logger.info(
        f"Análisis completado - Puntuación: {overall_score:.1f}/100 - Nivel: {skill_level}"
    )

    return {
        "metrics": all_metrics,
        "feedback": all_feedback,
        "score": overall_score,
        "level": skill_level,
        "exercise_config": exercise_config,
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
    }

    # Identificar áreas de mejora y puntos fuertes
    for category, message in analysis_results["feedback"].items():
        if "Buen" in message or "Buena" in message:
            report["puntos_fuertes"].append(message)
        else:
            report["areas_mejora"].append(message)

    # Guardar informe si se especificó una ruta
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        logger.info(f"Informe guardado en: {output_path}")

    return report
