# backend/src/utils/analysis_utis.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Configuraciones de ejercicios
EXERCISE_CONFIGS = {
    "press_militar": {
        "min_elbow_angle": 45,
        "max_elbow_angle": 175,
        "rom_threshold": 0.85,
        "bottom_diff_threshold": 0.2,
        "angle_diff_threshold": 15,
        "symmetry_threshold": 0.15,
        "lateral_dev_threshold": 0.2,
        "velocity_ratio_threshold": 0.3,
    },
    "press_banca": {
        "min_elbow_angle": 45,
        "max_elbow_angle": 175,
        "rom_threshold": 0.80,
        "bottom_diff_threshold": 0.15,
        "angle_diff_threshold": 12,
        "symmetry_threshold": 0.10,
        "lateral_dev_threshold": 0.15,
        "velocity_ratio_threshold": 0.25,
    },
}


def get_exercise_config(exercise_name="press_militar"):
    """Obtiene configuración específica para el ejercicio."""
    return EXERCISE_CONFIGS.get(exercise_name, EXERCISE_CONFIGS["press_militar"])


def calculate_angle(p1, p2, p3):
    """Calcula el ángulo entre tres puntos en el espacio 3D."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return 0

    cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def calculate_individual_scores(all_metrics, exercise_config):
    """Calcula puntuaciones individuales para cada categoría."""
    # Amplitud (0-100)
    rom_ratio = all_metrics["amplitud"]["rom_ratio"]
    rom_score = (
        min(100, 100 * rom_ratio)
        if rom_ratio <= 1
        else max(0, 100 - 50 * (rom_ratio - 1))
    )

    # Ángulos de codo (0-100)
    angle_diff = abs(all_metrics["angulos_codo"]["diferencia_angulo"])
    angle_score = max(0, 100 - 3 * angle_diff)

    # Simetría (0-100)
    sym_score = max(0, 100 - 300 * all_metrics["simetria"]["diferencia_normalizada"])

    # Trayectoria (0-100)
    path_score = max(
        0, 100 - 250 * all_metrics["trayectoria"]["desviacion_normalizada"]
    )

    # Velocidad (0-100)
    speed_concentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_subida"])
    speed_eccentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_bajada"])
    speed_score = (speed_concentric + speed_eccentric) / 2

    # Posición hombros (0-100)
    shoulder_diff = abs(all_metrics["hombros"]["diferencia_ratio"])
    shoulder_score = max(0, 100 - 1000 * shoulder_diff)

    return {
        "rom_score": rom_score,
        "angle_score": angle_score,
        "sym_score": sym_score,
        "path_score": path_score,
        "speed_score": speed_score,
        "shoulder_score": shoulder_score,
    }


def calculate_overall_score(all_metrics, exercise_config):
    """Calcula puntuación global basada en métricas individuales."""
    scores = calculate_individual_scores(all_metrics, exercise_config)
    return np.mean(list(scores.values()))


def determine_skill_level(overall_score):
    """Determina nivel de habilidad basado en puntuación."""
    if overall_score >= 90:
        return "Excelente"
    elif overall_score >= 80:
        return "Muy bueno"
    elif overall_score >= 70:
        return "Bueno"
    elif overall_score >= 60:
        return "Aceptable"
    elif overall_score >= 50:
        return "Necesita mejorar"
    else:
        return "Principiante"


def generate_recommendations(all_feedback, overall_score):
    """Genera recomendaciones específicas basadas en feedback."""
    recommendations = []

    # Mapeo de problemas a recomendaciones
    recommendation_map = {
        "insuficiente": "Practica el movimiento completo con menos peso para mejorar la amplitud.",
        "posicion_baja": "Trabaja en llevar las pesas hasta la altura de los hombros o ligeramente por debajo al bajar.",
        "abiertos": "Realiza ejercicios de conciencia corporal frente al espejo para corregir la apertura de codos.",
        "cerrados": "Intenta mantener los codos apuntando ligeramente hacia fuera durante el ejercicio.",
        "asimetría": "Realiza ejercicios unilaterales (con un brazo a la vez) para equilibrar la fuerza entre ambos lados.",
        "desvía": "Practica frente a un espejo con una barra ligera o sin peso para corregir la trayectoria.",
        "lenta": "Incorpora alguna serie con menor peso pero mayor velocidad controlada en la fase de subida.",
        "rápida": "Cuenta mentalmente durante la bajada para asegurar un descenso controlado (aprox. 2-3 segundos).",
        "elevados": "Antes de cada repetición, realiza una exhalación consciente mientras bajas los hombros.",
    }

    # Buscar problemas en feedback y agregar recomendaciones
    for category, message in all_feedback.items():
        for problem, recommendation in recommendation_map.items():
            if problem in message:
                recommendations.append(recommendation)
                break

    # Recomendaciones generales si la puntuación es baja
    if not recommendations and overall_score < 80:
        recommendations.extend(
            [
                "Grábate realizando el ejercicio regularmente para revisar tu técnica.",
                "Considera realizar el ejercicio con menos peso para enfocarte en la técnica.",
            ]
        )

    if len(recommendations) < 2:
        if overall_score < 70:
            recommendations.append(
                "Considera algunas sesiones con un entrenador personal para perfeccionar tu técnica."
            )
        if overall_score < 60:
            recommendations.append(
                "Comienza con variantes más sencillas del press militar, como el press sentado con respaldo."
            )

    return recommendations
