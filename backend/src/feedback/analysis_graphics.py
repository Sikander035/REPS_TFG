# src/feedback/analysis_graphics.py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.analysis_utils import (
    calculate_angle,
    calculate_individual_scores,
    generate_recommendations,
)

logger = logging.getLogger(__name__)


def visualize_analysis_results(
    analysis_results, user_data, expert_data, exercise_name, output_dir=None
):
    """
    Crea visualizaciones de los resultados del análisis.
    """
    metrics = analysis_results["metrics"]
    visualizations = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Gráfico de amplitud de movimiento
    visualizations.append(
        _create_amplitude_chart(
            metrics, user_data, expert_data, exercise_name, output_dir
        )
    )

    # 2. Gráfico de ángulos de codo
    visualizations.append(
        _create_angles_chart(user_data, expert_data, exercise_name, output_dir)
    )

    # 3. Gráfico de trayectorias
    visualizations.append(
        _create_trajectory_chart(user_data, expert_data, exercise_name, output_dir)
    )

    # 4. Gráfico de simetría
    visualizations.append(
        _create_symmetry_chart(user_data, analysis_results, exercise_name, output_dir)
    )

    # 5. Gráfico de velocidad
    visualizations.append(
        _create_velocity_chart(user_data, expert_data, exercise_name, output_dir)
    )

    # 6. Gráfico de puntuaciones por categoría
    visualizations.append(
        _create_scores_chart(analysis_results, exercise_name, output_dir)
    )

    # 7. Gráfico de radar (IMPORTANTE)
    visualizations.append(
        _create_radar_chart(analysis_results, exercise_name, output_dir)
    )

    # 8. Resumen visual
    visualizations.append(
        _create_summary_chart(analysis_results, exercise_name, output_dir)
    )

    # Filtrar None values
    return [viz for viz in visualizations if viz is not None]


def _create_amplitude_chart(metrics, user_data, expert_data, exercise_name, output_dir):
    """Crea gráfico de amplitud de movimiento."""
    plt.figure(figsize=(10, 6))

    user_wrist_y = user_data["landmark_right_wrist_y"].values
    expert_wrist_y = expert_data["landmark_right_wrist_y"].values

    plt.plot(user_wrist_y, label="Usuario", color="blue")
    plt.plot(expert_wrist_y, label="Experto", color="red")

    # Líneas de referencia
    plt.axhline(
        metrics["amplitud"]["punto_mas_alto_usuario"],
        linestyle="--",
        color="blue",
        alpha=0.7,
    )
    plt.axhline(
        metrics["amplitud"]["punto_mas_bajo_usuario"],
        linestyle="--",
        color="blue",
        alpha=0.7,
    )
    plt.axhline(
        metrics["amplitud"]["punto_mas_alto_experto"],
        linestyle="--",
        color="red",
        alpha=0.7,
    )
    plt.axhline(
        metrics["amplitud"]["punto_mas_bajo_experto"],
        linestyle="--",
        color="red",
        alpha=0.7,
    )

    plt.title(f"Amplitud de Movimiento - {exercise_name}")
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


def _create_angles_chart(user_data, expert_data, exercise_name, output_dir):
    """Crea gráfico de ángulos de codo."""
    plt.figure(figsize=(10, 6))

    user_angles = []
    expert_angles = []

    for i in range(len(user_data)):
        try:
            # Calcular ángulos para cada frame
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
        except:
            pass

    plt.plot(user_angles, label="Usuario", color="blue")
    plt.plot(expert_angles, label="Experto", color="red")

    plt.title(f"Ángulos de Codo - {exercise_name}")
    plt.xlabel("Frame")
    plt.ylabel("Ángulo (grados)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        path = os.path.join(output_dir, "angulos_codo.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    plt.close()
    return None


def _create_trajectory_chart(user_data, expert_data, exercise_name, output_dir):
    """Crea gráfico de trayectorias."""
    plt.figure(figsize=(10, 8))

    user_r_wrist_x = user_data["landmark_right_wrist_x"].values
    user_r_wrist_y = user_data["landmark_right_wrist_y"].values
    expert_r_wrist_x = expert_data["landmark_right_wrist_x"].values
    expert_r_wrist_y = expert_data["landmark_right_wrist_y"].values

    plt.scatter(
        user_r_wrist_x, user_r_wrist_y, s=10, alpha=0.7, color="blue", label="Usuario"
    )
    plt.plot(user_r_wrist_x, user_r_wrist_y, color="blue", alpha=0.4)

    plt.scatter(
        expert_r_wrist_x,
        expert_r_wrist_y,
        s=10,
        alpha=0.7,
        color="red",
        label="Experto",
    )
    plt.plot(expert_r_wrist_x, expert_r_wrist_y, color="red", alpha=0.4)

    plt.title(f"Trayectoria de la Muñeca Derecha - {exercise_name}")
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


def _create_symmetry_chart(user_data, analysis_results, exercise_name, output_dir):
    """Crea gráfico de simetría bilateral."""
    plt.figure(figsize=(10, 6))

    diff_y = abs(
        user_data["landmark_right_wrist_y"].values
        - user_data["landmark_left_wrist_y"].values
    )

    plt.plot(diff_y, label="Diferencia entre muñecas", color="purple")
    plt.axhline(
        y=analysis_results["exercise_config"]["symmetry_threshold"],
        linestyle="--",
        color="red",
        label=f'Umbral de asimetría ({analysis_results["exercise_config"]["symmetry_threshold"]})',
    )

    plt.title(f"Simetría Bilateral - {exercise_name}")
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
    """Crea gráfico de velocidad."""
    plt.figure(figsize=(10, 6))

    user_velocity = np.gradient(user_data["landmark_right_wrist_y"].values)
    expert_velocity = np.gradient(expert_data["landmark_right_wrist_y"].values)

    plt.plot(user_velocity, label="Usuario", color="blue")
    plt.plot(expert_velocity, label="Experto", color="red")

    plt.title(f"Velocidad Vertical - {exercise_name}")
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


def _create_scores_chart(analysis_results, exercise_name, output_dir):
    """Crea gráfico de puntuaciones por categoría."""
    plt.figure(figsize=(10, 6))

    categories = [
        "Amplitud",
        "Ángulos\nCodos",
        "Simetría",
        "Trayectoria",
        "Velocidad",
        "Posición\nHombros",
        "Global",
    ]

    scores = calculate_individual_scores(
        analysis_results["metrics"], analysis_results["exercise_config"]
    )
    scores_list = [
        scores["rom_score"],
        scores["angle_score"],
        scores["sym_score"],
        scores["path_score"],
        scores["speed_score"],
        scores["shoulder_score"],
        analysis_results["score"],
    ]

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


def _create_radar_chart(analysis_results, exercise_name, output_dir):
    """Crea gráfico de radar - EL MÁS IMPORTANTE."""
    try:
        plt.figure(figsize=(10, 8))

        categories_radar = [
            "Amplitud",
            "Ángulos\nCodos",
            "Simetría",
            "Trayectoria",
            "Velocidad",
            "Posición\nHombros",
        ]

        scores = calculate_individual_scores(
            analysis_results["metrics"], analysis_results["exercise_config"]
        )
        scores_normalized = [
            scores["rom_score"] / 100,
            scores["angle_score"] / 100,
            scores["sym_score"] / 100,
            scores["path_score"] / 100,
            scores["speed_score"] / 100,
            scores["shoulder_score"] / 100,
        ]

        # Cerrar el polígono
        scores_normalized = np.concatenate((scores_normalized, [scores_normalized[0]]))
        categories_radar = np.concatenate((categories_radar, [categories_radar[0]]))

        # Ángulos para cada categoría
        angles = np.linspace(
            0, 2 * np.pi, len(categories_radar), endpoint=False
        ).tolist()

        # Crear gráfico de radar
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, scores_normalized, color="blue", alpha=0.25)
        ax.plot(angles, scores_normalized, color="blue", linewidth=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_radar[:-1])
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
        logger.warning(f"Error al crear gráfico de radar: {e}")
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
            if "Buen" not in msg and "Buena" not in msg
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
            if "Buen" in msg or "Buena" in msg
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
        logger.warning(f"Error al crear resumen visual: {e}")
        return None
