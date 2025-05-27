# backend/src/core/min_square_normalization.py
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import config_manager
from src.utils.landmark_utils import (
    extract_landmarks_as_matrix,
    solve_affine_transform_3d,
    transform_points,
    compute_transform_error,
    visualize_landmarks_3d,
)

# Configurar logging para depuración
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_skeleton_with_affine_transform(
    expert_frame,
    user_frame,
    landmarks_to_use=None,
    regularization=0.0,
    compute_error=False,
    visualize=False,
    output_dir=None,
):
    """
    Normaliza el esqueleto del experto al del usuario usando una transformación
    afín global calculada con mínimos cuadrados.
    """
    # Extraer landmarks como matrices
    expert_points, expert_landmark_names = extract_landmarks_as_matrix(
        expert_frame, landmarks_to_use
    )
    user_points, user_landmark_names = extract_landmarks_as_matrix(
        user_frame, landmarks_to_use
    )

    # Verificar que tengamos suficientes puntos correspondientes
    common_landmarks = set(expert_landmark_names).intersection(set(user_landmark_names))
    if len(common_landmarks) < 4:
        logger.warning(
            f"Solo se encontraron {len(common_landmarks)} landmarks comunes. Se necesitan al menos 4."
        )
        return (expert_frame.copy(), None) if compute_error else expert_frame.copy()

    # Extraer solo los puntos correspondientes
    expert_indices = [expert_landmark_names.index(name) for name in common_landmarks]
    user_indices = [user_landmark_names.index(name) for name in common_landmarks]

    expert_matched = expert_points[expert_indices]
    user_matched = user_points[user_indices]

    try:
        # Calcular la transformación afín
        transform = solve_affine_transform_3d(
            expert_matched, user_matched, regularization
        )

        # Aplicar transformación a todos los puntos del experto
        normalized_points = transform_points(expert_points, transform)

        # Reconstruir el frame con los puntos normalizados
        result = expert_frame.copy()
        for i, joint in enumerate(expert_landmark_names):
            result[f"{joint}_x"] = normalized_points[i, 0]
            result[f"{joint}_y"] = normalized_points[i, 1]
            result[f"{joint}_z"] = normalized_points[i, 2]

        # Calcular error si se solicita
        error_metrics = None
        if compute_error:
            error_metrics = compute_transform_error(
                expert_matched, user_matched, normalized_points[expert_indices]
            )
            logger.debug(
                f"Error inicial: {error_metrics['initial_error']:.6f}, "
                f"Error final: {error_metrics['final_error']:.6f}, "
                f"Mejora: {error_metrics['improvement']:.2f}%"
            )

        # Visualizar si se solicita
        if visualize and output_dir:
            _visualize_transformation(
                user_matched,
                expert_matched,
                normalized_points[expert_indices],
                output_dir,
            )

        return (result, error_metrics) if compute_error else result

    except Exception as e:
        logger.error(f"Error al calcular la transformación afín: {e}")
        return (expert_frame.copy(), None) if compute_error else expert_frame.copy()


def _visualize_transformation(
    user_points, expert_points, normalized_points, output_dir
):
    """
    Visualiza la transformación antes y después.
    Esta función auxiliar simplifica normalize_skeleton_with_affine_transform.
    """
    os.makedirs(output_dir, exist_ok=True)
    viz_path = os.path.join(output_dir, "normalization_visualization.png")

    plt.figure(figsize=(15, 5))

    # Antes de normalización
    ax1 = plt.subplot(131, projection="3d")
    ax1.scatter(
        user_points[:, 0],
        user_points[:, 1],
        user_points[:, 2],
        c="blue",
        label="Usuario",
    )
    ax1.scatter(
        expert_points[:, 0],
        expert_points[:, 1],
        expert_points[:, 2],
        c="red",
        label="Experto",
    )
    ax1.set_title("Antes de normalización")
    ax1.legend()

    # Después de normalización
    ax2 = plt.subplot(132, projection="3d")
    ax2.scatter(
        user_points[:, 0],
        user_points[:, 1],
        user_points[:, 2],
        c="blue",
        label="Usuario",
    )
    ax2.scatter(
        normalized_points[:, 0],
        normalized_points[:, 1],
        normalized_points[:, 2],
        c="green",
        label="Experto normalizado",
    )
    ax2.set_title("Después de normalización")
    ax2.legend()

    # Errores
    errors = np.sqrt(np.sum((normalized_points - user_points) ** 2, axis=1))
    ax3 = plt.subplot(133)
    ax3.bar(range(len(errors)), errors)
    ax3.set_title("Errores de alineación")

    plt.tight_layout()
    plt.savefig(viz_path)
    plt.close()


def normalize_skeletons_with_affine_method(
    user_data,
    expert_data,
    config=None,
    exercise_name=None,
    config_path="config.json",
    compute_error=False,
    visualize=False,
    visualize_frames=None,
    output_dir=None,
):
    """
    Normaliza todos los frames del DataFrame del experto al usuario
    usando una transformación afín global con mínimos cuadrados.
    """
    if user_data.empty or expert_data.empty:
        raise ValueError("DataFrames de usuario o experto vacíos")
    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los DataFrames deben tener la misma longitud. Usuario: {len(user_data)}, Experto: {len(expert_data)}"
        )

    # Cargar configuración usando singleton
    if exercise_name and not config:
        try:
            exercise_config = config_manager.get_exercise_config(
                exercise_name, config_path
            )
            config = exercise_config.get("sync_config", {})
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    if config is None:
        config = {}

    # Parámetros de normalización
    landmarks_to_use = config.get("normalization_landmarks", None)
    regularization = config.get("normalization_regularization", 0.0)

    # Preparar DataFrames
    frames = (
        expert_data["frame"].copy()
        if "frame" in expert_data.columns
        else pd.Series(range(len(expert_data)))
    )
    exp_coords = expert_data.filter(regex=r"landmark_.*_[xyz]$")
    usr_coords = user_data.filter(regex=r"landmark_.*_[xyz]$")
    normalized = pd.DataFrame(index=exp_coords.index, columns=exp_coords.columns)

    # Métricas
    error_metrics_df = (
        pd.DataFrame(
            index=range(len(usr_coords)),
            columns=["initial_error", "final_error", "improvement"],
        )
        if compute_error
        else None
    )

    # Determinar frames a visualizar
    if visualize and visualize_frames is None:
        total_frames = len(usr_coords)
        visualize_frames = (
            [
                0,
                total_frames // 4,
                total_frames // 2,
                3 * total_frames // 4,
                total_frames - 1,
            ]
            if total_frames > 5
            else list(range(total_frames))
        )

    # Crear directorio
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Procesamiento secuencial - Eliminamos el procesamiento paralelo para simplificar
    logger.info(f"Normalizando {len(exp_coords)} frames...")
    for i in range(len(usr_coords)):
        should_visualize = visualize and i in visualize_frames
        frame_dir = (
            os.path.join(output_dir, f"frame_{i}")
            if should_visualize and output_dir
            else None
        )

        try:
            if compute_error:
                result, metrics = normalize_skeleton_with_affine_transform(
                    exp_coords.iloc[i],
                    usr_coords.iloc[i],
                    landmarks_to_use,
                    regularization,
                    compute_error,
                    should_visualize,
                    frame_dir,
                )

                normalized.iloc[i] = result
                if metrics:
                    error_metrics_df.loc[i, "initial_error"] = metrics["initial_error"]
                    error_metrics_df.loc[i, "final_error"] = metrics["final_error"]
                    error_metrics_df.loc[i, "improvement"] = metrics["improvement"]
            else:
                normalized.iloc[i] = normalize_skeleton_with_affine_transform(
                    exp_coords.iloc[i],
                    usr_coords.iloc[i],
                    landmarks_to_use,
                    regularization,
                    compute_error,
                    should_visualize,
                    frame_dir,
                )
        except Exception as e:
            logger.error(f"Error procesando frame {i}: {e}")
            normalized.iloc[i] = exp_coords.iloc[i]

    # Añadir columna de frame
    normalized.insert(0, "frame", frames.values)

    # Visualizar resumen
    if compute_error and visualize and output_dir:
        _visualize_error_summary(error_metrics_df, output_dir)

    logger.info(f"Normalización completada para {len(normalized)} frames")
    return (normalized, error_metrics_df) if compute_error else normalized


def _visualize_error_summary(error_metrics_df, output_dir):
    """
    Visualiza un resumen de errores de normalización.
    Función auxiliar separada para simplificar el código principal.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(error_metrics_df["initial_error"], label="Error inicial")
        plt.plot(error_metrics_df["final_error"], label="Error final")
        plt.title("Errores de normalización por frame")
        plt.xlabel("Frame")
        plt.ylabel("Error (distancia)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "normalization_errors.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"Error al crear resumen visual: {e}")
