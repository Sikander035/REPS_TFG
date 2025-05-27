# src/feedback/analysis_graphics.py - CORRECCIÓN SIMPLE: Solo cambiar import
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ÚNICO CAMBIO: Importar de analysis_utils en lugar de usar función duplicada
from src.utils.analysis_utils import (
    calculate_elbow_abduction_angle,
    calculate_individual_scores,
    generate_recommendations,
)

logger = logging.getLogger(__name__)


def visualize_analysis_results(
    analysis_results, user_data, expert_data, exercise_name, output_dir=None
):
    """
    Crea visualizaciones de los resultados del análisis.
    LÓGICA ORIGINAL INTACTA.
    """
    metrics = analysis_results["metrics"]
    visualizations = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Gráfico de amplitud de movimiento (ACTUALIZADO para codos)
    visualizations.append(
        _create_amplitude_chart(
            metrics, user_data, expert_data, exercise_name, output_dir
        )
    )

    # 2. Gráfico de abducción de codos (ACTUALIZADO)
    visualizations.append(
        _create_abduction_chart(user_data, expert_data, exercise_name, output_dir)
    )

    # 3. Gráfico de trayectorias
    visualizations.append(
        _create_trajectory_chart(user_data, expert_data, exercise_name, output_dir)
    )

    # 4. Gráfico de simetría (ACTUALIZADO para codos)
    visualizations.append(
        _create_symmetry_chart(user_data, analysis_results, exercise_name, output_dir)
    )

    # 5. Gráfico de velocidad (ACTUALIZADO para codos)
    visualizations.append(
        _create_velocity_chart(user_data, expert_data, exercise_name, output_dir)
    )

    # 6. Gráfico de puntuaciones por categoría (ACTUALIZADO)
    visualizations.append(
        _create_scores_chart(analysis_results, exercise_name, output_dir)
    )

    # 7. Gráfico de radar (ACTUALIZADO)
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
    """Crea gráfico de amplitud de movimiento usando CODOS."""
    plt.figure(figsize=(10, 6))

    # ACTUALIZADO: Usar codos en lugar de muñecas
    user_elbow_y = (
        user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
    ) / 2
    expert_elbow_y = (
        expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
    ) / 2

    plt.plot(user_elbow_y, label="Usuario", color="blue")
    plt.plot(expert_elbow_y, label="Experto", color="red")

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

    plt.title(f"Amplitud de Movimiento (Codos) - {exercise_name}")
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
    """Crea gráfico de abducción lateral de codos CORREGIDO."""
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

                # CORREGIDO: Usar la nueva función de abducción lateral
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

        # Añadir líneas de referencia para interpretación - VALORES ORIGINALES
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
    """Crea gráfico de simetría bilateral usando CODOS."""
    plt.figure(figsize=(10, 6))

    # ACTUALIZADO: Usar codos en lugar de muñecas
    diff_y = abs(
        user_data["landmark_right_elbow_y"].values
        - user_data["landmark_left_elbow_y"].values
    )

    plt.plot(diff_y, label="Diferencia entre codos", color="purple")
    plt.axhline(
        y=analysis_results["exercise_config"]["symmetry_threshold"],
        linestyle="--",
        color="red",
        label=f'Umbral de asimetría ({analysis_results["exercise_config"]["symmetry_threshold"]})',
    )

    plt.title(f"Simetría Bilateral (Codos) - {exercise_name}")
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
    """Crea gráfico de velocidad usando CODOS."""
    plt.figure(figsize=(10, 6))

    # ACTUALIZADO: Usar codos en lugar de muñecas
    user_elbow_y = (
        user_data["landmark_right_elbow_y"] + user_data["landmark_left_elbow_y"]
    ) / 2
    expert_elbow_y = (
        expert_data["landmark_right_elbow_y"] + expert_data["landmark_left_elbow_y"]
    ) / 2

    user_velocity = np.gradient(user_elbow_y.values)
    expert_velocity = np.gradient(expert_elbow_y.values)

    plt.plot(user_velocity, label="Usuario", color="blue")
    plt.plot(expert_velocity, label="Experto", color="red")

    plt.title(f"Velocidad Vertical (Codos) - {exercise_name}")
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
    """Crea gráfico de radar CORREGIDO."""
    try:
        plt.figure(figsize=(10, 8))

        # CORREGIDO: Definir 6 categorías exactas
        categories_radar = [
            "Amplitud",
            "Abducción\nCodos",
            "Simetría",
            "Trayectoria",
            "Velocidad",
            "Estabilidad\nEscapular",
        ]

        scores = calculate_individual_scores(
            analysis_results["metrics"], analysis_results["exercise_config"]
        )

        # CORREGIDO: Extraer exactamente 6 scores normalizados
        scores_normalized = [
            scores["rom_score"] / 100,
            scores["abduction_score"] / 100,
            scores["sym_score"] / 100,
            scores["path_score"] / 100,
            scores["speed_score"] / 100,
            scores["scapular_score"] / 100,
        ]

        # DEBUG: Imprimir valores para verificación
        logger.info("=== DEBUG RADAR CHART ===")
        for i, (cat, score_norm, score_raw) in enumerate(
            zip(
                categories_radar,
                scores_normalized,
                [
                    scores["rom_score"],
                    scores["abduction_score"],
                    scores["sym_score"],
                    scores["path_score"],
                    scores["speed_score"],
                    scores["scapular_score"],
                ],
            )
        ):
            logger.info(
                f"{i}: {cat.replace(chr(10), ' ')} = {score_raw:.1f} ({score_norm:.3f})"
            )

        # CORREGIDO: Calcular ángulos ANTES de cerrar el polígono
        # Crear exactamente 6 ángulos para 6 categorías
        angles = np.linspace(
            0, 2 * np.pi, len(categories_radar), endpoint=False
        ).tolist()

        # DEBUG: Verificar ángulos
        logger.info(f"Ángulos (grados): {[f'{np.degrees(a):.1f}°' for a in angles]}")

        # CORREGIDO: Ahora sí, cerrar el polígono duplicando el primer elemento
        scores_normalized = np.concatenate((scores_normalized, [scores_normalized[0]]))
        angles = angles + [angles[0]]  # Duplicar solo el primer ángulo

        # Crear gráfico de radar
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, scores_normalized, color="blue", alpha=0.25)
        ax.plot(angles, scores_normalized, color="blue", linewidth=2)

        # CORREGIDO: Usar ángulos originales (sin duplicado) para las etiquetas
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
        logger.warning(f"Error al crear gráfico de radar: {e}")
        return None


def _create_scores_chart(analysis_results, exercise_name, output_dir):
    """Crea gráfico de puntuaciones por categoría CORREGIDO."""
    plt.figure(figsize=(10, 6))

    # CORREGIDO: Usar exactamente las mismas 6 categorías que el radar
    categories = [
        "Amplitud",
        "Abducción\nCodos",
        "Simetría",
        "Trayectoria",
        "Velocidad",
        "Estabilidad\nEscapular",
        "Global",
    ]

    scores = calculate_individual_scores(
        analysis_results["metrics"], analysis_results["exercise_config"]
    )

    # CORREGIDO: Usar exactamente los mismos scores que el radar + global
    scores_list = [
        scores["rom_score"],
        scores["abduction_score"],
        scores["sym_score"],
        scores["path_score"],
        scores["speed_score"],
        scores["scapular_score"],
        analysis_results["score"],
    ]

    # DEBUG: Imprimir valores para verificación
    logger.info("=== DEBUG SCORES CHART ===")
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
        logger.warning(f"Error al crear resumen visual: {e}")
        return None
