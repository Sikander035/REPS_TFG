# backend/src/utils/analysis_utils.py - VERSIÓN SIN DEFAULTS
import numpy as np
import logging
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


def apply_unified_sensitivity(base_score, sensitivity_factor, metric_name="unknown"):
    """
    Aplica sensibilidad de manera unificada y bidireccional a todas las métricas.

    Args:
        base_score: Score base calculado sin sensibilidad
        sensitivity_factor: Factor de sensibilidad (0.5=permisivo, 1.0=normal, 2.0=estricto)
        metric_name: Nombre de la métrica para logging

    Returns:
        Score ajustado con sensibilidad aplicada

    Raises:
        ValueError: Si sensitivity_factor es inválido
    """
    if sensitivity_factor <= 0:
        raise ValueError(
            f"Invalid sensitivity factor for {metric_name}: {sensitivity_factor}. Must be positive"
        )

    # Calcular ajuste basado en desviación del factor normal (1.0)
    if sensitivity_factor < 1.0:
        # Factor bajo = Más permisivo = PREMIO (score sube)
        bonus_factor = 1.0 - sensitivity_factor  # 0.5 → 0.5, 0.8 → 0.2
        bonus = min(20, bonus_factor * 40)  # Máximo bonus de 20 puntos
        adjusted_score = min(100, base_score + bonus)
        logger.debug(
            f"{metric_name}: Factor {sensitivity_factor:.2f} → Bonus +{bonus:.1f} → {base_score:.1f}→{adjusted_score:.1f}"
        )

    elif sensitivity_factor > 1.0:
        # Factor alto = Más estricto = CASTIGO (score baja)
        penalty_factor = sensitivity_factor - 1.0  # 2.0 → 1.0, 1.5 → 0.5
        # Limitar penalty máximo para evitar colapsos
        max_penalty = 30 if sensitivity_factor > 2.0 else 25
        penalty = min(max_penalty, penalty_factor * 30)
        adjusted_score = max(10, base_score - penalty)  # Score mínimo de 10
        logger.debug(
            f"{metric_name}: Factor {sensitivity_factor:.2f} → Penalty -{penalty:.1f} → {base_score:.1f}→{adjusted_score:.1f}"
        )

    else:
        # Factor normal = Sin cambios
        adjusted_score = base_score
        logger.debug(
            f"{metric_name}: Factor {sensitivity_factor:.2f} → Sin cambios → {adjusted_score:.1f}"
        )

    return adjusted_score


def calculate_deviation_score(
    actual_value, ideal_value, max_penalty=30, metric_type="linear"
):
    """
    Calcula score base basado en desviación del valor ideal.

    Args:
        actual_value: Valor real medido
        ideal_value: Valor ideal/objetivo
        max_penalty: Penalización máxima por desviación completa
        metric_type: "linear", "ratio", o "logarithmic"

    Returns:
        Score base (antes de aplicar sensibilidad)

    Raises:
        ValueError: Si max_penalty es inválido
    """
    if max_penalty <= 0:
        raise ValueError(f"Invalid max_penalty: {max_penalty}. Must be positive")

    if metric_type == "ratio" and ideal_value != 0:
        # Para ratios (ej: usuario/experto), ideal normalmente es 1.0
        deviation = abs(actual_value - ideal_value) / abs(ideal_value)
    elif metric_type == "logarithmic":
        # Para métricas que necesitan cambios suaves en rangos amplios
        deviation = abs(
            np.log(max(0.01, actual_value)) - np.log(max(0.01, ideal_value))
        )
    else:
        # Linear - para diferencias absolutas
        deviation = abs(actual_value - ideal_value)

    # Convertir desviación a penalty (normalizado)
    # Asumimos que 100% de desviación = penalty máximo
    penalty = min(max_penalty, deviation * max_penalty)
    base_score = max(20, 100 - penalty)  # Score base mínimo de 20

    return base_score


def calculate_elbow_abduction_angle(shoulder_point, elbow_point):
    """
    Calcula el ángulo de abducción lateral del codo.

    Args:
        shoulder_point: Punto del hombro [x, y, z]
        elbow_point: Punto del codo [x, y, z]

    Returns:
        float: Ángulo de abducción en grados
    """
    shoulder = np.array(shoulder_point)
    elbow = np.array(elbow_point)

    vector_shoulder_to_elbow = elbow - shoulder
    horizontal_projection = np.array(
        [
            vector_shoulder_to_elbow[0],
            vector_shoulder_to_elbow[2],
        ]
    )

    x_axis = np.array([1.0, 0.0])
    projection_magnitude = np.linalg.norm(horizontal_projection)

    if projection_magnitude < 1e-6:
        return 90.0

    normalized_projection = horizontal_projection / projection_magnitude
    dot_product = np.dot(normalized_projection, x_axis)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(abs(dot_product))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def apply_sensitivity_to_threshold(threshold, sensitivity_factor):
    """
    Aplica factor de sensibilidad a un umbral.

    Args:
        threshold: Umbral original
        sensitivity_factor: Factor de sensibilidad

    Returns:
        float: Umbral ajustado

    Raises:
        ValueError: Si sensitivity_factor es inválido
    """
    if sensitivity_factor <= 0:
        raise ValueError(
            f"Invalid sensitivity factor: {sensitivity_factor}. Must be positive"
        )

    adjusted_threshold = threshold / sensitivity_factor
    logger.debug(
        f"Threshold adjusted: {threshold} / {sensitivity_factor} = {adjusted_threshold}"
    )
    return adjusted_threshold


def determine_skill_level(overall_score, skill_levels_config=None):
    """
    Determina nivel de habilidad basado en puntuación.

    Args:
        overall_score: Puntuación general
        skill_levels_config: Configuración de niveles (opcional)

    Returns:
        str: Nivel de habilidad

    Raises:
        ValueError: Si overall_score es inválido
    """
    if (
        not isinstance(overall_score, (int, float))
        or overall_score < 0
        or overall_score > 100
    ):
        raise ValueError(
            f"Invalid overall_score: {overall_score}. Must be between 0 and 100"
        )

    # Usar configuración si está disponible
    if skill_levels_config and isinstance(skill_levels_config, dict):
        if overall_score >= skill_levels_config.get("excelente", 90):
            return "Excelente"
        elif overall_score >= skill_levels_config.get("muy_bueno", 80):
            return "Muy bueno"
        elif overall_score >= skill_levels_config.get("bueno", 70):
            return "Bueno"
        elif overall_score >= skill_levels_config.get("aceptable", 60):
            return "Aceptable"
        elif overall_score >= skill_levels_config.get("necesita_mejorar", 50):
            return "Necesita mejorar"
        else:
            return "Principiante"
    else:
        # Valores hardcodeados como fallback (mantenidos para compatibilidad)
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
    """
    Genera recomendaciones específicas basadas en feedback.

    Args:
        all_feedback: Diccionario con feedback por categorías
        overall_score: Puntuación general

    Returns:
        list: Lista de recomendaciones

    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if not isinstance(all_feedback, dict):
        raise ValueError("all_feedback must be a dictionary")

    if (
        not isinstance(overall_score, (int, float))
        or overall_score < 0
        or overall_score > 100
    ):
        raise ValueError(
            f"Invalid overall_score: {overall_score}. Must be between 0 and 100"
        )

    recommendations = []

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

    for category, message in all_feedback.items():
        for problem, recommendation in recommendation_map.items():
            if problem in message.lower():
                recommendations.append(recommendation)
                break

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
