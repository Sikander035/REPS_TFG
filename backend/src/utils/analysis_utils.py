# backend/src/utils/analysis_utils.py
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


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


def apply_sensitivity_to_score(base_score, sensitivity_factor):
    """Aplica factor de sensibilidad a una puntuación."""
    if sensitivity_factor == 1.0:
        return base_score

    if base_score < 100:
        penalty = (100 - base_score) * sensitivity_factor
        adjusted_score = 100 - penalty
        return max(0, adjusted_score)

    return base_score


def apply_sensitivity_to_threshold(threshold, sensitivity_factor):
    """Aplica factor de sensibilidad a un umbral."""
    return threshold / sensitivity_factor


def calculate_individual_scores(all_metrics, exercise_config):
    """Calcula puntuaciones individuales para cada categoría."""
    sensitivity_factors = exercise_config.get("sensitivity_factors", {})

    # Amplitud (0-100)
    rom_ratio = all_metrics["amplitud"]["rom_ratio"]
    rom_score_base = (
        min(100, 100 * rom_ratio)
        if rom_ratio <= 1
        else max(0, 100 - 50 * (rom_ratio - 1))
    )
    rom_score = apply_sensitivity_to_score(
        rom_score_base, sensitivity_factors.get("amplitud", 1.0)
    )

    # Abducción de codos (0-100)
    try:
        abduction_diff = abs(all_metrics["abduccion_codos"]["diferencia_abduccion"])
        abduction_score_base = max(0, 100 - 3 * abduction_diff)
        abduction_score = apply_sensitivity_to_score(
            abduction_score_base, sensitivity_factors.get("abduccion_codos", 1.0)
        )
    except (KeyError, TypeError):
        abduction_score = 50

    # Simetría (0-100)
    try:
        sym_score_base = max(
            0, 100 - 300 * all_metrics["simetria"]["diferencia_normalizada"]
        )
        sym_score = apply_sensitivity_to_score(
            sym_score_base, sensitivity_factors.get("simetria", 1.0)
        )
    except (KeyError, TypeError):
        sym_score = 50

    # Trayectoria (0-100)
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

    # Velocidad (0-100)
    try:
        speed_concentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_subida"])
        speed_eccentric = 100 - 100 * abs(1 - all_metrics["velocidad"]["ratio_bajada"])
        speed_score_base = (speed_concentric + speed_eccentric) / 2
        speed_score = apply_sensitivity_to_score(
            speed_score_base, sensitivity_factors.get("velocidad", 1.0)
        )
    except (KeyError, TypeError):
        speed_score = 50

    # Estabilidad escapular (0-100)
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

    # Pesos - USANDO CONFIGURACIÓN O VALORES ORIGINALES
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

    weighted_score = sum(scores[key] * weights[key] for key in weights.keys())
    return weighted_score


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
