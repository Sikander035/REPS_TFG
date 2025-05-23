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
        "abduction_angle_threshold": 15,  # NUEVO - para abducción de codos
        "symmetry_threshold": 0.15,
        "lateral_dev_threshold": 0.2,
        "frontal_dev_threshold": 0.15,  # NUEVO - para trayectoria Z
        "velocity_ratio_threshold": 0.3,
        "scapular_stability_threshold": 1.5,  # NUEVO - para estabilidad escapular
    },
    "press_banca": {
        "min_elbow_angle": 45,
        "max_elbow_angle": 175,
        "rom_threshold": 0.80,
        "bottom_diff_threshold": 0.15,
        "abduction_angle_threshold": 12,
        "symmetry_threshold": 0.10,
        "lateral_dev_threshold": 0.15,
        "frontal_dev_threshold": 0.12,
        "velocity_ratio_threshold": 0.25,
        "scapular_stability_threshold": 1.3,
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
    """Calcula puntuaciones individuales para cada categoría ACTUALIZADA."""

    # Amplitud (0-100) - Sin cambios
    rom_ratio = all_metrics["amplitud"]["rom_ratio"]
    rom_score = (
        min(100, 100 * rom_ratio)
        if rom_ratio <= 1
        else max(0, 100 - 50 * (rom_ratio - 1))
    )

    # Abducción de codos (0-100) - ACTUALIZADO
    try:
        abduction_diff = abs(all_metrics["abduccion_codos"]["diferencia_abduccion"])
        abduction_score = max(0, 100 - 3 * abduction_diff)
    except (KeyError, TypeError):
        abduction_score = 50  # Valor por defecto si no está disponible

    # Simetría (0-100) - Sin cambios
    try:
        sym_score = max(
            0, 100 - 300 * all_metrics["simetria"]["diferencia_normalizada"]
        )
    except (KeyError, TypeError):
        sym_score = 50

    # Trayectoria (0-100) - ACTUALIZADO para 3D
    try:
        # Usar la peor de las dos desviaciones (lateral o frontal)
        lateral_ratio = all_metrics["trayectoria"].get("ratio_desviacion_lateral", 1.0)
        frontal_ratio = all_metrics["trayectoria"].get("ratio_desviacion_frontal", 1.0)
        worst_ratio = max(lateral_ratio, frontal_ratio)

        path_score = max(0, 100 - 50 * (worst_ratio - 1)) if worst_ratio >= 1 else 100
    except (KeyError, TypeError):
        path_score = 50

    # Velocidad (0-100) - Sin cambios
    try:
        speed_concentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_subida"])
        speed_eccentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_bajada"])
        speed_score = (speed_concentric + speed_eccentric) / 2
    except (KeyError, TypeError):
        speed_score = 50

    # Estabilidad escapular (0-100) - NUEVO
    try:
        movement_ratio = all_metrics["estabilidad_escapular"]["ratio_movimiento"]
        asymmetry_ratio = all_metrics["estabilidad_escapular"]["ratio_asimetria"]

        # Penalizar ratios > 1.5 (más inestable que experto)
        movement_penalty = (
            max(0, (movement_ratio - 1.5) * 40) if movement_ratio > 1.5 else 0
        )
        asymmetry_penalty = (
            max(0, (asymmetry_ratio - 1.5) * 40) if asymmetry_ratio > 1.5 else 0
        )

        scapular_score = max(0, 100 - movement_penalty - asymmetry_penalty)
    except (KeyError, TypeError):
        scapular_score = 50

    return {
        "rom_score": rom_score,
        "abduction_score": abduction_score,  # CAMBIADO de angle_score
        "sym_score": sym_score,
        "path_score": path_score,
        "speed_score": speed_score,
        "scapular_score": scapular_score,  # NUEVO
    }


def calculate_overall_score(all_metrics, exercise_config):
    """Calcula puntuación global basada en métricas individuales ACTUALIZADA."""
    scores = calculate_individual_scores(all_metrics, exercise_config)

    # Pesos para cada categoría (deben sumar 1.0)
    weights = {
        "rom_score": 0.20,  # 20% - Amplitud
        "abduction_score": 0.20,  # 20% - Abducción de codos
        "sym_score": 0.15,  # 15% - Simetría
        "path_score": 0.20,  # 20% - Trayectoria
        "speed_score": 0.15,  # 15% - Velocidad
        "scapular_score": 0.10,  # 10% - Estabilidad escapular
    }

    # Calcular promedio ponderado
    weighted_score = sum(scores[key] * weights[key] for key in weights.keys())

    return weighted_score


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
    """Genera recomendaciones específicas basadas en feedback ACTUALIZADO."""
    recommendations = []

    # Mapeo de problemas a recomendaciones ACTUALIZADO
    recommendation_map = {
        "insuficiente": "Practica el movimiento completo con menos peso para mejorar la amplitud.",
        "posicion_baja": "Trabaja en llevar los codos hasta la altura de los hombros al bajar.",
        "abiertos": "Realiza ejercicios de conciencia corporal frente al espejo para corregir la abducción de codos.",
        "cerrados": "Intenta mantener los codos en una posición intermedia, ni muy abiertos ni muy cerrados.",
        "asimetría": "Realiza ejercicios unilaterales (con un brazo a la vez) para equilibrar la fuerza entre ambos lados.",
        "desvía": "Practica frente a un espejo con una barra ligera o sin peso para corregir la trayectoria.",
        "lateral": "Concéntrate en mantener un movimiento vertical, evitando desviaciones hacia los lados.",
        "frontal": "Evita empujar las pesas hacia adelante o atrás, mantén un plano vertical.",
        "lenta": "Incorpora alguna serie con menor peso pero mayor velocidad controlada en la fase de subida.",
        "rápida": "Cuenta mentalmente durante la bajada para asegurar un descenso controlado (aprox. 2-3 segundos).",
        "inestabilidad": "Fortalece los músculos de la cintura escapular con ejercicios específicos como retracciones escapulares.",
        "mueven": "Practica mantener los hombros fijos durante todo el movimiento del press.",
    }

    # Buscar problemas en feedback y agregar recomendaciones
    for category, message in all_feedback.items():
        for problem, recommendation in recommendation_map.items():
            if problem in message.lower():
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
