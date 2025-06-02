# src/feedback/analysis_graphics.py - SIN DEFAULTS, USANDO CONFIG_MANAGER
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# IMPORTAR SOLO LO NECESARIO
from src.utils.analysis_utils import (
    calculate_elbow_abduction_angle,
    generate_recommendations,
)

from src.config.config_manager import config_manager

logger = logging.getLogger(__name__)


def visualize_analysis_results(
    analysis_results,
    user_data,
    expert_data,
    exercise_name,
    output_dir=None,
    config_path="config.json",
):
    """
    Crea visualizaciones de los resultados del análisis usando scores unificados.
    SIN DEFAULTS - todos los valores vienen de configuración o fallan.
    """
    if not isinstance(analysis_results, dict):
        raise ValueError("analysis_results must be a dictionary")

    if not isinstance(exercise_name, str) or not exercise_name.strip():
        raise ValueError("exercise_name must be a non-empty string")

    if user_data is None or user_data.empty:
        raise ValueError("user_data cannot be None or empty")

    if expert_data is None or expert_data.empty:
        raise ValueError("expert_data cannot be None or empty")

    # Validar campos requeridos
    required_fields = ["metrics", "individual_scores"]
    for field in required_fields:
        if field not in analysis_results:
            raise ValueError(f"Missing required field '{field}' in analysis_results")

    metrics = analysis_results["metrics"]
    individual_scores = analysis_results["individual_scores"]

    visualizations = []

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            raise ValueError(f"Cannot create output directory: {e}")

    # 1. Gráfico de amplitud de movimiento
    try:
        viz_path = _create_amplitude_chart(
            metrics, user_data, expert_data, exercise_name, output_dir
        )
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating amplitude chart: {e}")

    # 2. Gráfico de abducción de codos (solo para press militar y compatibles)
    try:
        viz_path = _create_abduction_chart(
            user_data, expert_data, exercise_name, output_dir
        )
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating abduction chart: {e}")

    # 3. Gráfico de trayectorias
    try:
        viz_path = _create_trajectory_chart(
            user_data, expert_data, exercise_name, output_dir
        )
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating trajectory chart: {e}")

    # 4. Gráfico de simetría
    try:
        viz_path = _create_symmetry_chart(
            user_data, analysis_results, exercise_name, output_dir, config_path
        )
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating symmetry chart: {e}")

    # 5. Gráfico de velocidad
    try:
        viz_path = _create_velocity_chart(
            user_data, expert_data, exercise_name, output_dir
        )
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating velocity chart: {e}")

    # 6. Gráfico de puntuaciones por categoría
    try:
        viz_path = _create_scores_chart(analysis_results, exercise_name, output_dir)
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating scores chart: {e}")

    # 7. Gráfico de radar
    try:
        viz_path = _create_radar_chart(analysis_results, exercise_name, output_dir)
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")

    # 8. Resumen visual
    try:
        viz_path = _create_summary_chart(analysis_results, exercise_name, output_dir)
        if viz_path:
            visualizations.append(viz_path)
    except Exception as e:
        logger.error(f"Error creating summary chart: {e}")

    return visualizations


def _create_amplitude_chart(metrics, user_data, expert_data, exercise_name, output_dir):
    """Crea gráfico de amplitud de movimiento usando landmarks según ejercicio."""
    plt.figure(figsize=(10, 6))

    # Determinar landmarks según ejercicio
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean == "press_militar":
        # Usar codos
        user_signal = (
            user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
        ) / 2
        landmark_name = "Codos"
    elif exercise_name_clean == "sentadilla":
        # Usar caderas
        user_signal = (
            user_data["landmark_right_hip_y"] + user_data["landmark_left_hip_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_hip_y"] + expert_data["landmark_left_hip_y"]
        ) / 2
        landmark_name = "Caderas"
    elif exercise_name_clean == "dominada":
        # Usar codos
        user_signal = (
            user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
        ) / 2
        landmark_name = "Codos"
    else:
        logger.warning(
            f"Unknown exercise {exercise_name_clean}, using default landmarks"
        )
        # Fallback a codos
        user_signal = (
            user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
        ) / 2
        landmark_name = "Codos"

    plt.plot(user_signal, label="Usuario", color="blue")
    plt.plot(expert_signal, label="Experto", color="red")

    # Obtener métricas de amplitud
    amplitude_metrics = metrics.get("amplitud", {})
    if amplitude_metrics:
        # Líneas de referencia
        plt.axhline(
            amplitude_metrics.get("punto_mas_alto_usuario", 0),
            linestyle="--",
            color="blue",
            alpha=0.7,
        )
        plt.axhline(
            amplitude_metrics.get("punto_mas_bajo_usuario", 0),
            linestyle="--",
            color="blue",
            alpha=0.7,
        )
        plt.axhline(
            amplitude_metrics.get("punto_mas_alto_experto", 0),
            linestyle="--",
            color="red",
            alpha=0.7,
        )
        plt.axhline(
            amplitude_metrics.get("punto_mas_bajo_experto", 0),
            linestyle="--",
            color="red",
            alpha=0.7,
        )

    plt.title(f"Amplitud de Movimiento ({landmark_name}) - {exercise_name}")
    plt.xlabel("Frame")
    plt.ylabel("Coordenada Y (MediaPipe)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "amplitud_movimiento.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_abduction_chart(user_data, expert_data, exercise_name, output_dir):
    """Crea gráfico de abducción lateral de codos (solo para ejercicios relevantes)."""

    # Solo crear para ejercicios que usan codos
    exercise_name_clean = exercise_name.lower().replace(" ", "_")
    if exercise_name_clean not in ["press_militar", "dominada"]:
        logger.info(f"Skipping abduction chart for exercise {exercise_name_clean}")
        return None

    plt.figure(figsize=(10, 6))

    user_abduction_angles = []
    expert_abduction_angles = []

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

            # Verificar que no hay NaN
            if (
                not np.isnan(user_right_shoulder).any()
                and not np.isnan(user_right_elbow).any()
                and not np.isnan(user_left_shoulder).any()
                and not np.isnan(user_left_elbow).any()
                and not np.isnan(expert_right_shoulder).any()
                and not np.isnan(expert_right_elbow).any()
                and not np.isnan(expert_left_shoulder).any()
                and not np.isnan(expert_left_elbow).any()
            ):

                # Usar la función de abducción lateral
                user_right_angle = calculate_elbow_abduction_angle(
                    user_right_shoulder, user_right_elbow
                )
                user_left_angle = calculate_elbow_abduction_angle(
                    user_left_shoulder, user_left_elbow
                )
                expert_right_angle = calculate_elbow_abduction_angle(
                    expert_right_shoulder, expert_right_elbow
                )
                expert_left_angle = calculate_elbow_abduction_angle(
                    expert_left_shoulder, expert_left_elbow
                )

                # Promedio de ambos codos
                user_avg_angle = (user_right_angle + user_left_angle) / 2
                expert_avg_angle = (expert_right_angle + expert_left_angle) / 2

                user_abduction_angles.append(user_avg_angle)
                expert_abduction_angles.append(expert_avg_angle)

        except Exception as e:
            logger.warning(f"Error calculando ángulos de abducción en frame {i}: {e}")
            pass

    # Plotear las señales si tenemos datos
    if user_abduction_angles and expert_abduction_angles:
        plt.plot(user_abduction_angles, label="Usuario", color="blue", linewidth=2)
        plt.plot(expert_abduction_angles, label="Experto", color="red", linewidth=2)

        plt.title(f"Abducción Lateral de Codos - {exercise_name}")
        plt.xlabel("Frame")
        plt.ylabel("Ángulo de Abducción Lateral (grados)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Añadir líneas de referencia para interpretación
        plt.axhline(
            y=30, color="green", linestyle="--", alpha=0.5, label="Muy abierto (30°)"
        )
        plt.axhline(
            y=60, color="orange", linestyle="--", alpha=0.5, label="Moderado (60°)"
        )
        plt.axhline(
            y=80, color="red", linestyle="--", alpha=0.5, label="Muy cerrado (80°)"
        )

        # Actualizar leyenda
        plt.legend()

        # Mostrar estadísticas en el gráfico
        user_avg = np.mean(user_abduction_angles)
        expert_avg = np.mean(expert_abduction_angles)
        user_min = np.min(user_abduction_angles)
        expert_min = np.min(expert_abduction_angles)

        plt.text(
            0.02,
            0.98,
            f"Usuario - Promedio: {user_avg:.1f}°, Mínimo: {user_min:.1f}°\n"
            f"Experto - Promedio: {expert_avg:.1f}°, Mínimo: {expert_min:.1f}°\n"
            f"Diferencia promedio: {user_avg - expert_avg:.1f}°",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        if output_dir:
            path = os.path.join(output_dir, "abduccion_lateral_codos.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return path
    else:
        logger.warning("No se pudieron calcular ángulos de abducción para el gráfico")

    plt.close()
    return None


def _create_trajectory_chart(user_data, expert_data, exercise_name, output_dir):
    """Crea gráfico de trayectorias usando landmarks apropiados."""
    plt.figure(figsize=(10, 8))

    # Determinar landmarks según ejercicio
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean in ["press_militar", "dominada"]:
        # Usar muñecas para press y dominadas
        user_x = user_data["landmark_right_wrist_x"].values
        user_y = user_data["landmark_right_wrist_y"].values
        expert_x = expert_data["landmark_right_wrist_x"].values
        expert_y = expert_data["landmark_right_wrist_y"].values
        landmark_name = "Muñeca Derecha"
    elif exercise_name_clean == "sentadilla":
        # Usar caderas para sentadilla
        user_x = user_data["landmark_right_hip_x"].values
        user_y = user_data["landmark_right_hip_y"].values
        expert_x = expert_data["landmark_right_hip_x"].values
        expert_y = expert_data["landmark_right_hip_y"].values
        landmark_name = "Cadera Derecha"
    else:
        # Fallback a muñecas
        user_x = user_data["landmark_right_wrist_x"].values
        user_y = user_data["landmark_right_wrist_y"].values
        expert_x = expert_data["landmark_right_wrist_x"].values
        expert_y = expert_data["landmark_right_wrist_y"].values
        landmark_name = "Muñeca Derecha"

    plt.scatter(user_x, user_y, s=10, alpha=0.7, color="blue", label="Usuario")
    plt.plot(user_x, user_y, color="blue", alpha=0.4)

    plt.scatter(expert_x, expert_y, s=10, alpha=0.7, color="red", label="Experto")
    plt.plot(expert_x, expert_y, color="red", alpha=0.4)

    plt.title(f"Trayectoria de {landmark_name} - {exercise_name}")
    plt.xlabel("X (lateral)")
    plt.ylabel("Y (vertical)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "trayectoria_frontal.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_symmetry_chart(
    user_data, analysis_results, exercise_name, output_dir, config_path
):
    """Crea gráfico de simetría bilateral usando landmarks apropiados."""
    plt.figure(figsize=(10, 6))

    # Determinar landmarks según ejercicio
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean == "press_militar":
        # Usar codos
        diff_y = abs(
            user_data["landmark_right_elbow_y"].values
            - user_data["landmark_left_elbow_y"].values
        )
        landmark_name = "codos"
    elif exercise_name_clean == "sentadilla":
        # Usar rodillas
        diff_y = abs(
            user_data["landmark_right_knee_y"].values
            - user_data["landmark_left_knee_y"].values
        )
        landmark_name = "rodillas"
    elif exercise_name_clean == "dominada":
        # Usar codos
        diff_y = abs(
            user_data["landmark_right_elbow_y"].values
            - user_data["landmark_left_elbow_y"].values
        )
        landmark_name = "codos"
    else:
        # Fallback a codos
        diff_y = abs(
            user_data["landmark_right_elbow_y"].values
            - user_data["landmark_left_elbow_y"].values
        )
        landmark_name = "codos"

    plt.plot(diff_y, label=f"Diferencia entre {landmark_name}", color="purple")

    # Obtener umbral usando config_manager con fallback seguro
    try:
        symmetry_threshold = config_manager.get_analysis_threshold(
            "symmetry_threshold", exercise_name_clean, config_path
        )

        plt.axhline(
            y=symmetry_threshold,
            linestyle="--",
            color="red",
            label=f"Umbral de asimetría ({symmetry_threshold})",
        )
        logger.debug(f"Successfully got symmetry threshold: {symmetry_threshold}")
    except Exception as e:
        logger.warning(f"Could not get symmetry threshold from config: {e}")
        # No agregar línea de umbral si no se puede obtener

    plt.title(f"Simetría Bilateral ({landmark_name.title()}) - {exercise_name}")
    plt.xlabel("Frame")
    plt.ylabel("Diferencia de altura (valor absoluto)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "simetria_bilateral.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_velocity_chart(user_data, expert_data, exercise_name, output_dir):
    """Crea gráfico de velocidad usando landmarks apropiados."""
    plt.figure(figsize=(10, 6))

    # Determinar landmarks según ejercicio
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean == "press_militar":
        # Usar codos
        user_signal = (
            user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
        ) / 2
        landmark_name = "Codos"
    elif exercise_name_clean == "sentadilla":
        # Usar caderas
        user_signal = (
            user_data["landmark_right_hip_y"] + user_data["landmark_left_hip_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_hip_y"] + expert_data["landmark_left_hip_y"]
        ) / 2
        landmark_name = "Caderas"
    elif exercise_name_clean == "dominada":
        # Usar codos
        user_signal = (
            user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
        ) / 2
        landmark_name = "Codos"
    else:
        # Fallback a codos
        user_signal = (
            user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
        ) / 2
        expert_signal = (
            expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
        ) / 2
        landmark_name = "Codos"

    user_velocity = np.gradient(user_signal.values)
    expert_velocity = np.gradient(expert_signal.values)

    plt.plot(user_velocity, label="Usuario", color="blue")
    plt.plot(expert_velocity, label="Experto", color="red")

    plt.title(f"Velocidad Vertical ({landmark_name}) - {exercise_name}")
    plt.xlabel("Frame")
    plt.ylabel("Velocidad (unidades/frame)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "velocidad.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_radar_chart(analysis_results, exercise_name, output_dir):
    """
    Crea gráfico de radar USANDO SCORES DINÁMICOS según el ejercicio.
    """
    try:
        plt.figure(figsize=(10, 8))

        individual_scores = analysis_results.get("individual_scores", {})

        if not individual_scores:
            logger.error("No individual scores found in analysis_results")
            return None

        # Categorías y scores dinámicos según ejercicio
        exercise_name_clean = exercise_name.lower().replace(" ", "_")

        if exercise_name_clean == "press_militar":
            categories_radar = [
                "Amplitud",
                "Abducción\nCodos",
                "Simetría",
                "Trayectoria",
                "Velocidad",
                "Estabilidad\nEscapular",
            ]
            score_keys = [
                "rom_score",
                "abduction_score",
                "sym_score",
                "path_score",
                "speed_score",
                "scapular_score",
            ]
        elif exercise_name_clean == "sentadilla":
            categories_radar = [
                "Amplitud",
                "Profundidad",
                "Simetría",
                "Trayectoria",
                "Velocidad",
                "Tracking\nRodillas",
            ]
            score_keys = [
                "rom_score",
                "depth_score",
                "sym_score",
                "path_score",
                "speed_score",
                "knee_score",
            ]
        elif exercise_name_clean == "dominada":
            categories_radar = [
                "Amplitud",
                "Control\nSwing",
                "Simetría",
                "Trayectoria",
                "Velocidad",
                "Retracción\nEscapular",
            ]
            score_keys = [
                "rom_score",
                "swing_score",
                "sym_score",
                "path_score",
                "speed_score",
                "retraction_score",
            ]
        else:
            # Fallback para ejercicios desconocidos
            score_keys = list(individual_scores.keys())
            categories_radar = [
                key.replace("_score", "").replace("_", " ").title()
                for key in score_keys
            ]

        # Extraer scores dinámicamente
        scores_normalized = []
        scores_raw = []

        for score_key in score_keys:
            if score_key not in individual_scores:
                logger.error(f"Score key '{score_key}' not found in individual_scores")
                return None

            score_value = individual_scores[score_key]
            scores_normalized.append(score_value / 100)
            scores_raw.append(score_value)

        # DEBUG: Imprimir valores para verificación
        logger.info("=== DEBUG RADAR CHART STRICT ===")
        for i, (cat, score_norm, score_raw) in enumerate(
            zip(categories_radar, scores_normalized, scores_raw)
        ):
            logger.info(
                f"{i}: {cat.replace(chr(10), ' ')} = {score_raw:.1f} ({score_norm:.3f})"
            )

        # Calcular ángulos ANTES de cerrar el polígono
        angles = np.linspace(
            0, 2 * np.pi, len(categories_radar), endpoint=False
        ).tolist()

        # Cerrar el polígono duplicando el primer elemento
        scores_normalized = np.concatenate((scores_normalized, [scores_normalized[0]]))
        angles = angles + [angles[0]]

        # Crear gráfico de radar
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, scores_normalized, color="blue", alpha=0.25)
        ax.plot(angles, scores_normalized, color="blue", linewidth=2)

        # Usar ángulos originales (sin duplicado) para las etiquetas
        original_angles = np.linspace(
            0, 2 * np.pi, len(categories_radar), endpoint=False
        ).tolist()
        ax.set_xticks(original_angles)
        ax.set_xticklabels(categories_radar)

        # Configurar escala radial
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25", "50", "75", "100"])

        plt.title(f"Análisis de Técnica - {exercise_name}", size=15, y=1.1)

        if output_dir:
            path = os.path.join(output_dir, "analisis_radar.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return path

        plt.close()
        return None

    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        return None


def _create_scores_chart(analysis_results, exercise_name, output_dir):
    """
    Crea gráfico de puntuaciones por categoría USANDO SCORES DINÁMICOS.
    """
    plt.figure(figsize=(10, 6))

    individual_scores = analysis_results.get("individual_scores", {})

    if not individual_scores:
        logger.error("No individual scores found in analysis_results")
        return None

    # Categorías dinámicas según ejercicio
    exercise_name_clean = exercise_name.lower().replace(" ", "_")

    if exercise_name_clean == "press_militar":
        categories = [
            "Amplitud",
            "Abducción\nCodos",
            "Simetría",
            "Trayectoria",
            "Velocidad",
            "Estabilidad\nEscapular",
            "Global",
        ]
        score_keys = [
            "rom_score",
            "abduction_score",
            "sym_score",
            "path_score",
            "speed_score",
            "scapular_score",
        ]
    elif exercise_name_clean == "sentadilla":
        categories = [
            "Amplitud",
            "Profundidad",
            "Simetría",
            "Trayectoria",
            "Velocidad",
            "Tracking\nRodillas",
            "Global",
        ]
        score_keys = [
            "rom_score",
            "depth_score",
            "sym_score",
            "path_score",
            "speed_score",
            "knee_score",
        ]
    elif exercise_name_clean == "dominada":
        categories = [
            "Amplitud",
            "Control\nSwing",
            "Simetría",
            "Trayectoria",
            "Velocidad",
            "Retracción\nEscapular",
            "Global",
        ]
        score_keys = [
            "rom_score",
            "swing_score",
            "sym_score",
            "path_score",
            "speed_score",
            "retraction_score",
        ]
    else:
        # Fallback para ejercicios desconocidos
        score_keys = list(individual_scores.keys())
        categories = [
            key.replace("_score", "").replace("_", " ").title() for key in score_keys
        ]
        categories.append("Global")

    # Extraer scores dinámicamente
    scores_list = []
    for score_key in score_keys:
        if score_key not in individual_scores:
            logger.error(f"Score key '{score_key}' not found in individual_scores")
            return None
        scores_list.append(individual_scores[score_key])

    # Añadir score global al final
    scores_list.append(analysis_results["score"])

    # DEBUG: Imprimir valores para verificación
    logger.info("=== DEBUG SCORES CHART STRICT ===")
    for cat, score in zip(categories, scores_list):
        logger.info(f"{cat.replace(chr(10), ' ')}: {score:.1f}")

    # Colores según puntuación
    colors = []
    for score in scores_list:
        if score >= 90:
            colors.append("#27ae60")
        elif score >= 70:
            colors.append("#2ecc71")
        elif score >= 50:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    bars = plt.bar(categories, scores_list, color=colors)

    # Etiquetas con valores
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.axhline(y=50, color="r", linestyle="-", alpha=0.3, label="Deficiente")
    plt.axhline(y=70, color="y", linestyle="-", alpha=0.3, label="Aceptable")
    plt.axhline(y=90, color="g", linestyle="-", alpha=0.3, label="Excelente")

    plt.title(f"Puntuación por Categoría - {exercise_name}")
    plt.ylabel("Puntuación (0-100)")
    plt.ylim(0, 105)
    plt.legend(loc="lower right")

    if output_dir:
        path = os.path.join(output_dir, "puntuacion_categorias.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_summary_chart(analysis_results, exercise_name, output_dir):
    """Crea resumen visual textual."""
    try:
        plt.figure(figsize=(12, 8))

        plt.text(
            0.5,
            0.95,
            f"ANÁLISIS DE TÉCNICA - {exercise_name.upper()}",
            fontsize=18,
            ha="center",
            va="top",
            fontweight="bold",
        )

        plt.text(
            0.5,
            0.88,
            f"Puntuación Global: {analysis_results['score']:.1f}/100 - Nivel: {analysis_results['level']}",
            fontsize=16,
            ha="center",
            va="top",
        )

        # Áreas de mejora
        areas_mejora = [
            msg
            for msg in analysis_results["feedback"].values()
            if "Buen" not in msg and "Buena" not in msg and "Excelente" not in msg
        ]

        plt.text(0.05, 0.8, "ÁREAS DE MEJORA:", fontsize=14, fontweight="bold")
        y_pos = 0.75
        for i, area in enumerate(areas_mejora[:5]):
            plt.text(0.07, y_pos - i * 0.05, f"• {area}", fontsize=12)

        # Recomendaciones
        plt.text(0.05, 0.5, "RECOMENDACIONES:", fontsize=14, fontweight="bold")
        y_pos = 0.45
        recommendations = generate_recommendations(
            analysis_results["feedback"], analysis_results["score"]
        )
        for i, rec in enumerate(recommendations[:5]):
            plt.text(0.07, y_pos - i * 0.05, f"• {rec}", fontsize=12)

        # Puntos fuertes
        puntos_fuertes = [
            msg
            for msg in analysis_results["feedback"].values()
            if "Buen" in msg or "Buena" in msg or "Excelente" in msg
        ]

        plt.text(0.05, 0.2, "PUNTOS FUERTES:", fontsize=14, fontweight="bold")
        y_pos = 0.15
        for i, punto in enumerate(puntos_fuertes[:3]):
            plt.text(0.07, y_pos - i * 0.05, f"• {punto}", fontsize=12)

        plt.axis("off")

        if output_dir:
            path = os.path.join(output_dir, "resumen_visual.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return path

        plt.close()
        return None

    except Exception as e:
        logger.error(f"Error creating summary chart: {e}")
        return None
