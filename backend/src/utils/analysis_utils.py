# backend/src/utils/analysis_utils.py - VERSIÓN CORREGIDA CON SENSIBILIDAD UNIFICADA
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
    """
    if sensitivity_factor <= 0:
        logger.warning(f"Factor de sensibilidad inválido para {metric_name}: {sensitivity_factor}. Usando 1.0")
        sensitivity_factor = 1.0
    
    # Calcular ajuste basado en desviación del factor normal (1.0)
    if sensitivity_factor < 1.0:
        # Factor bajo = Más permisivo = PREMIO (score sube)
        bonus_factor = (1.0 - sensitivity_factor)  # 0.5 → 0.5, 0.8 → 0.2
        bonus = min(20, bonus_factor * 40)  # Máximo bonus de 20 puntos
        adjusted_score = min(100, base_score + bonus)
        logger.debug(f"{metric_name}: Factor {sensitivity_factor:.2f} → Bonus +{bonus:.1f} → {base_score:.1f}→{adjusted_score:.1f}")
        
    elif sensitivity_factor > 1.0:
        # Factor alto = Más estricto = CASTIGO (score baja)
        penalty_factor = (sensitivity_factor - 1.0)  # 2.0 → 1.0, 1.5 → 0.5
        # Limitar penalty máximo para evitar colapsos
        max_penalty = 30 if sensitivity_factor > 2.0 else 25
        penalty = min(max_penalty, penalty_factor * 30)
        adjusted_score = max(10, base_score - penalty)  # Score mínimo de 10
        logger.debug(f"{metric_name}: Factor {sensitivity_factor:.2f} → Penalty -{penalty:.1f} → {base_score:.1f}→{adjusted_score:.1f}")
        
    else:
        # Factor normal = Sin cambios
        adjusted_score = base_score
        logger.debug(f"{metric_name}: Factor {sensitivity_factor:.2f} → Sin cambios → {adjusted_score:.1f}")
    
    return adjusted_score


def calculate_deviation_score(actual_value, ideal_value, max_penalty=30, metric_type="linear"):
    """
    Calcula score base basado en desviación del valor ideal.
    
    Args:
        actual_value: Valor real medido
        ideal_value: Valor ideal/objetivo
        max_penalty: Penalización máxima por desviación completa
        metric_type: "linear", "ratio", o "logarithmic"
    
    Returns:
        Score base (antes de aplicar sensibilidad)
    """
    if metric_type == "ratio" and ideal_value != 0:
        # Para ratios (ej: usuario/experto), ideal normalmente es 1.0
        deviation = abs(actual_value - ideal_value) / abs(ideal_value)
    elif metric_type == "logarithmic":
        # Para métricas que necesitan cambios suaves en rangos amplios
        deviation = abs(np.log(max(0.01, actual_value)) - np.log(max(0.01, ideal_value)))
    else:
        # Linear - para diferencias absolutas
        deviation = abs(actual_value - ideal_value)
    
    # Convertir desviación a penalty (normalizado)
    # Asumimos que 100% de desviación = penalty máximo
    penalty = min(max_penalty, deviation * max_penalty)
    base_score = max(20, 100 - penalty)  # Score base mínimo de 20
    
    return base_score


def get_exercise_config(exercise_name="press_militar", config_path="config.json"):
    """
    Obtiene configuración específica para el ejercicio desde config.json.
    USA SINGLETON + LEE CONFIGURACIÓN DEL ARCHIVO JSON.
    """
    try:
        # Usar el config_manager singleton
        exercise_config = config_manager.get_exercise_config(exercise_name, config_path)

        # Cargar configuraciones globales si no están cargadas
        if config_path not in config_manager._loaded_files:
            config_manager.load_config_file(config_path)
        config_data = config_manager._loaded_files[config_path]

        # Obtener configuración de análisis del ejercicio específico
        analysis_config = exercise_config.get("analysis_config", {})

        # Obtener configuración global de análisis
        global_analysis = config_data.get("global_analysis_config", {})

        # Combinar: global primero, luego específico del ejercicio (prioridad)
        final_config = {}
        final_config.update(global_analysis)
        final_config.update(analysis_config)

        # Añadir configuraciones adicionales del nivel global
        final_config.update(
            {
                "scoring_weights": config_data.get("scoring_weights", {}),
                "analysis_ratios": config_data.get("analysis_ratios", {}),
                "feedback_multipliers": config_data.get("feedback_multipliers", {}),
                "skill_levels": config_data.get("skill_levels", {}),
                "signal_processing": config_data.get("signal_processing", {}),
            }
        )

        # Si no hay configuración suficiente, completar con valores por defecto
        default_config = {
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
            "sensitivity_factors": {
                "amplitud": 3.0,
                "abduccion_codos": 3.0,
                "simetria": 1.0,
                "trayectoria": 1.0,
                "velocidad": 1.0,
                "estabilidad_escapular": 1.0,
            },
            "scoring_weights": {
                "rom_score": 0.20,
                "abduction_score": 0.20,
                "sym_score": 0.15,
                "path_score": 0.20,
                "speed_score": 0.15,
                "scapular_score": 0.10,
            },
        }

        # Completar valores faltantes con defaults
        for key, value in default_config.items():
            if key not in final_config:
                final_config[key] = value

        logger.info(f"Configuración de análisis cargada para {exercise_name}")
        return final_config

    except Exception as e:
        logger.error(f"Error al cargar configuración: {e}")
        logger.warning("Usando configuración por defecto completa")
        # Valores por defecto completos
        return {
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
            "sensitivity_factors": {
                "amplitud": 3.0,
                "abduccion_codos": 3.0,
                "simetria": 1.0,
                "trayectoria": 1.0,
                "velocidad": 1.0,
                "estabilidad_escapular": 1.0,
            },
            "scoring_weights": {
                "rom_score": 0.20,
                "abduction_score": 0.20,
                "sym_score": 0.15,
                "path_score": 0.20,
                "speed_score": 0.15,
                "scapular_score": 0.10,
            },
        }


def calculate_elbow_abduction_angle(shoulder_point, elbow_point):
    """
    Calcula el ángulo de abducción lateral del codo.
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
    """Aplica factor de sensibilidad a un umbral (MÉTODO LEGACY - mantener compatibilidad)."""
    if sensitivity_factor <= 0:
        logger.warning(
            f"Factor de sensibilidad inválido: {sensitivity_factor}. Usando 1.0"
        )
        sensitivity_factor = 1.0

    adjusted_threshold = threshold / sensitivity_factor
    logger.debug(
        f"Umbral ajustado: {threshold} / {sensitivity_factor} = {adjusted_threshold}"
    )
    return adjusted_threshold


def determine_skill_level(overall_score, exercise_config=None):
    """Determina nivel de habilidad basado en puntuación."""
    # Usar configuración si está disponible, sino valores originales
    if exercise_config and "skill_levels" in exercise_config:
        skill_levels = exercise_config["skill_levels"]
        if overall_score >= skill_levels.get("excelente", 90):
            return "Excelente"
        elif overall_score >= skill_levels.get("muy_bueno", 80):
            return "Muy bueno"
        elif overall_score >= skill_levels.get("bueno", 70):
            return "Bueno"
        elif overall_score >= skill_levels.get("aceptable", 60):
            return "Aceptable"
        elif overall_score >= skill_levels.get("necesita_mejorar", 50):
            return "Necesita mejorar"
        else:
            return "Principiante"
    else:
        # Valores originales hardcodeados para mantener compatibilidad
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