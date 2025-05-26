# backend/src/utils/analysis_utils.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Configuraciones de ejercicios - SIMPLE
EXERCISE_CONFIGS = {
    "press_militar": {
        "min_elbow_angle": 45,
        "max_elbow_angle": 175,
        "rom_threshold": 0.85,
        "bottom_diff_threshold": 0.2,
        "abduction_angle_threshold": 15,
        "symmetry_threshold": 0.15,
        "lateral_dev_threshold": 0.2,
        "frontal_dev_threshold": 0.15,
        "velocity_ratio_threshold": 0.3,
        "scapular_stability_threshold": 1.5,
        # NUEVO: Factores de sensibilidad por análisis (1.0 = normal, >1.0 = más sensible)
        "sensitivity_factors": {
            "amplitud": 3.0,  # MÁS SENSIBLE para amplitud
            "abduccion_codos": 3.0,  # MÁS SENSIBLE para ángulo de codos
            "simetria": 1.0,  # Normal
            "trayectoria": 1.0,  # Normal
            "velocidad": 1.0,  # Normal
            "estabilidad_escapular": 1.0,  # Normal
        },
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
        "sensitivity_factors": {
            "amplitud": 1.8,  # Algo más sensible
            "abduccion_codos": 1.8,  # Algo más sensible
            "simetria": 1.0,
            "trayectoria": 1.0,
            "velocidad": 1.0,
            "estabilidad_escapular": 1.0,
        },
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


def apply_sensitivity_to_score(base_score, sensitivity_factor):
    """
    Aplica factor de sensibilidad a una puntuación.

    Args:
        base_score: Puntuación base (0-100)
        sensitivity_factor: Factor de sensibilidad (1.0 = normal, >1.0 = más sensible)

    Returns:
        Puntuación ajustada
    """
    if sensitivity_factor == 1.0:
        return base_score

    # Con mayor sensibilidad, penalizar más fuertemente puntuaciones mediocres
    if base_score < 100:
        # Amplificar la penalización
        penalty = (100 - base_score) * sensitivity_factor
        adjusted_score = 100 - penalty
        return max(0, adjusted_score)

    return base_score


def apply_sensitivity_to_threshold(threshold, sensitivity_factor):
    """
    Aplica factor de sensibilidad a un umbral.

    Args:
        threshold: Umbral base
        sensitivity_factor: Factor de sensibilidad (>1.0 hace el umbral más estricto)

    Returns:
        Umbral ajustado
    """
    return threshold / sensitivity_factor


def calculate_individual_scores(all_metrics, exercise_config):
    """Calcula puntuaciones individuales para cada categoría CON SENSIBILIDAD APLICABLE."""

    # Obtener factores de sensibilidad
    sensitivity_factors = exercise_config.get("sensitivity_factors", {})

    # Amplitud (0-100) - CON SENSIBILIDAD
    rom_ratio = all_metrics["amplitud"]["rom_ratio"]
    rom_score_base = (
        min(100, 100 * rom_ratio)
        if rom_ratio <= 1
        else max(0, 100 - 50 * (rom_ratio - 1))
    )
    rom_score = apply_sensitivity_to_score(
        rom_score_base, sensitivity_factors.get("amplitud", 1.0)
    )

    # Abducción de codos (0-100) - CON SENSIBILIDAD
    try:
        abduction_diff = abs(all_metrics["abduccion_codos"]["diferencia_abduccion"])
        abduction_score_base = max(0, 100 - 3 * abduction_diff)
        abduction_score = apply_sensitivity_to_score(
            abduction_score_base, sensitivity_factors.get("abduccion_codos", 1.0)
        )
    except (KeyError, TypeError):
        abduction_score = 50

    # Simetría (0-100) - CON SENSIBILIDAD
    try:
        sym_score_base = max(
            0, 100 - 300 * all_metrics["simetria"]["diferencia_normalizada"]
        )
        sym_score = apply_sensitivity_to_score(
            sym_score_base, sensitivity_factors.get("simetria", 1.0)
        )
    except (KeyError, TypeError):
        sym_score = 50

    # Trayectoria (0-100) - CON SENSIBILIDAD
    try:
        lateral_ratio = all_metrics["trayectoria"].get("ratio_desviacion_lateral", 1.0)
        frontal_ratio = all_metrics["trayectoria"].get("ratio_desviacion_frontal", 1.0)
        worst_ratio = max(lateral_ratio, frontal_ratio)

        path_score_base = (
            max(0, 100 - 50 * (worst_ratio - 1)) if worst_ratio >= 1 else 100
        )
        path_score = apply_sensitivity_to_score(
            path_score_base, sensitivity_factors.get("trayectoria", 1.0)
        )
    except (KeyError, TypeError):
        path_score = 50

    # Velocidad (0-100) - CON SENSIBILIDAD
    try:
        speed_concentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_subida"])
        speed_eccentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_bajada"])
        speed_score_base = (speed_concentric + speed_eccentric) / 2
        speed_score = apply_sensitivity_to_score(
            speed_score_base, sensitivity_factors.get("velocidad", 1.0)
        )
    except (KeyError, TypeError):
        speed_score = 50

    # Estabilidad escapular (0-100) - CON SENSIBILIDAD
    try:
        movement_ratio = all_metrics["estabilidad_escapular"]["ratio_movimiento"]
        asymmetry_ratio = all_metrics["estabilidad_escapular"]["ratio_asimetria"]

        movement_penalty = (
            max(0, (movement_ratio - 1.5) * 40) if movement_ratio > 1.5 else 0
        )
        asymmetry_penalty = (
            max(0, (asymmetry_ratio - 1.5) * 40) if asymmetry_ratio > 1.5 else 0
        )

        scapular_score_base = max(0, 100 - movement_penalty - asymmetry_penalty)
        scapular_score = apply_sensitivity_to_score(
            scapular_score_base, sensitivity_factors.get("estabilidad_escapular", 1.0)
        )
    except (KeyError, TypeError):
        scapular_score = 50

    return {
        "rom_score": rom_score,
        "abduction_score": abduction_score,
        "sym_score": sym_score,
        "path_score": path_score,
        "speed_score": speed_score,
        "scapular_score": scapular_score,
    }


def calculate_overall_score(all_metrics, exercise_config):
    """Calcula puntuación global basada en métricas individuales."""
    scores = calculate_individual_scores(all_metrics, exercise_config)

    # Pesos para cada categoría
    weights = {
        "rom_score": 0.20,
        "abduction_score": 0.20,
        "sym_score": 0.15,
        "path_score": 0.20,
        "speed_score": 0.15,
        "scapular_score": 0.10,
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
    """Genera recomendaciones específicas basadas en feedback."""
    recommendations = []

    # Mapeo de problemas a recomendaciones
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
