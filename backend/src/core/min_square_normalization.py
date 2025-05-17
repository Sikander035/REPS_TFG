import numpy as np
import sys
import pandas as pd
import logging
from scipy.linalg import lstsq
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional, Union, Any, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import load_exercise_config


# Configurar logging para depuración
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_landmarks_as_matrix(frame_data, landmarks_to_use=None):
    """
    Extrae landmarks del frame como una matriz Nx3 donde N es el número de landmarks.
    Retorna también la lista de nombres de landmarks para reconstrucción.

    Args:
        frame_data: Serie de pandas con datos de un frame
        landmarks_to_use: Lista opcional de nombres de landmarks a extraer

    Returns:
        (landmark_points, landmark_names): Matriz de puntos y lista de nombres
    """
    landmark_points = []
    landmark_names = []

    # Filtrar columnas si se especifican landmarks
    if landmarks_to_use:
        landmarks_filter = []
        for landmark in landmarks_to_use:
            for suffix in ["_x", "_y", "_z"]:
                landmarks_filter.append(f"{landmark}{suffix}")
    else:
        landmarks_filter = None

    for col in frame_data.index:
        if col.endswith("_x") and not col.startswith("frame"):
            joint = col.rsplit("_", 1)[0]

            # Si hay filtro, verificar que el landmark esté incluido
            if landmarks_filter and f"{joint}_x" not in landmarks_filter:
                continue

            if f"{joint}_y" in frame_data.index and f"{joint}_z" in frame_data.index:
                # Extraer las coordenadas x, y, z
                point = [
                    frame_data[f"{joint}_x"],
                    frame_data[f"{joint}_y"],
                    frame_data[f"{joint}_z"],
                ]

                # Verificar que no haya NaN
                if not np.isnan(point).any():
                    landmark_points.append(point)
                    landmark_names.append(joint)

    return np.array(landmark_points), landmark_names


def solve_affine_transform_3d(points_source, points_target, regularization=0.0):
    """
    Encuentra la transformación afín óptima de source a target
    usando cuadrados mínimos.

    Args:
        points_source: Matriz de puntos de origen
        points_target: Matriz de puntos de destino
        regularization: Factor de regularización para estabilidad

    Returns:
        Matriz de transformación afín 4x4
    """
    if (
        len(points_source) < 4
    ):  # Necesitamos al menos 4 puntos para una transformación 3D
        raise ValueError(
            "Se necesitan al menos 4 puntos para calcular una transformación afín 3D"
        )

    # Añadir columna de unos para coordenadas homogéneas
    ones = np.ones((points_source.shape[0], 1))
    X = np.hstack((points_source, ones))

    # Matriz de transformación a calcular
    A = np.zeros((4, 4))

    # Resolver el sistema para cada coordenada por separado
    if regularization > 0:
        # Con regularización (para estabilidad numérica)
        X_reg = np.eye(X.shape[1]) * regularization
        X_extended = np.vstack([X, X_reg])

        for i in range(3):  # x, y, z
            Y_extended = np.concatenate([points_target[:, i], np.zeros(X_reg.shape[0])])
            A[:, i] = lstsq(X_extended, Y_extended)[0]
    else:
        # Sin regularización (método original)
        for i in range(3):  # x, y, z
            A[:, i] = lstsq(X, points_target[:, i])[0]

    # Última fila de la matriz de transformación
    A[3, :] = [0, 0, 0, 1]

    return A


def transform_points(points, transform_matrix):
    """
    Aplica la matriz de transformación a los puntos.

    Args:
        points: Matriz de puntos a transformar
        transform_matrix: Matriz de transformación afín

    Returns:
        Matriz de puntos transformados
    """
    # Convertir a coordenadas homogéneas
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))

    # Aplicar transformación
    transformed_points = np.dot(points_homogeneous, transform_matrix)

    # Volver a coordenadas cartesianas
    return transformed_points[:, :3]


def compute_transform_error(original_points, target_points, transformed_points):
    """
    Calcula el error residual después de la transformación

    Args:
        original_points: Puntos originales
        target_points: Puntos objetivo
        transformed_points: Puntos después de transformación

    Returns:
        Diccionario con métricas de error
    """
    # Error antes de la transformación
    initial_error = np.mean(
        np.sqrt(np.sum((original_points - target_points) ** 2, axis=1))
    )

    # Error después de la transformación
    final_error = np.mean(
        np.sqrt(np.sum((transformed_points - target_points) ** 2, axis=1))
    )

    # Mejora porcentual
    improvement = (1 - final_error / initial_error) * 100 if initial_error > 0 else 0

    return {
        "initial_error": initial_error,
        "final_error": final_error,
        "improvement": improvement,
        "per_point_error": np.sqrt(
            np.sum((transformed_points - target_points) ** 2, axis=1)
        ),
    }


def visualize_normalization(
    user_points,
    expert_points,
    normalized_points,
    title="Normalización de esqueletos",
    output_path=None,
):
    """
    Visualiza los puntos antes y después de la normalización

    Args:
        user_points: Matriz de puntos del usuario
        expert_points: Matriz de puntos del experto (original)
        normalized_points: Matriz de puntos del experto normalizados
        title: Título para la visualización
        output_path: Ruta para guardar la imagen (opcional)
    """
    try:
        fig = plt.figure(figsize=(15, 5))

        # Antes de normalización
        ax1 = fig.add_subplot(131, projection="3d")
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
        ax2 = fig.add_subplot(132, projection="3d")
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

        # Errores de alineación
        errors = np.sqrt(np.sum((normalized_points - user_points) ** 2, axis=1))
        ax3 = fig.add_subplot(133)
        ax3.bar(range(len(errors)), errors)
        ax3.set_title("Errores de alineación por punto")
        ax3.set_xlabel("Índice de punto")
        ax3.set_ylabel("Error (distancia euclídea)")

        plt.suptitle(title)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            logger.info(f"Visualización guardada en: {output_path}")

        plt.show()

    except Exception as e:
        logger.warning(f"No se pudo generar la visualización: {e}")


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

    Args:
        expert_frame: Serie de datos del frame del experto
        user_frame: Serie de datos del frame del usuario
        landmarks_to_use: Lista opcional de landmarks a usar
        regularization: Factor de regularización
        compute_error: Si es True, calcula y devuelve métricas de error
        visualize: Si es True, genera visualización
        output_dir: Directorio para guardar visualizaciones

    Returns:
        Serie con el frame normalizado, y opcionalmente métricas de error
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
        # Retornar esqueleto original si no hay suficientes puntos
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
        if visualize:
            try:
                # Crear directorio si se especifica y no existe
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    viz_path = os.path.join(
                        output_dir, "normalization_visualization.png"
                    )
                else:
                    viz_path = None

                visualize_normalization(
                    user_matched,
                    expert_matched,
                    normalized_points[expert_indices],
                    title="Normalización de esqueleto por mínimos cuadrados",
                    output_path=viz_path,
                )
            except Exception as viz_error:
                logger.warning(f"Error en visualización: {viz_error}")

        return (result, error_metrics) if compute_error else result

    except Exception as e:
        logger.error(f"Error al calcular la transformación afín: {e}")
        return (expert_frame.copy(), None) if compute_error else expert_frame.copy()


def normalize_skeletons_with_affine_method(
    user_data,
    expert_data,
    config=None,
    exercise_name=None,
    config_path="config_expanded.json",
    use_parallel=False,
    max_workers=4,
    compute_error=False,
    visualize=False,
    visualize_frames=None,
    output_dir=None,
):
    """
    Normaliza todos los frames del DataFrame del experto al usuario
    usando una transformación afín global con mínimos cuadrados.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        config: Configuración personalizada (opcional)
        exercise_name: Nombre del ejercicio para cargar configuración (opcional)
        config_path: Ruta al archivo de configuración
        use_parallel: Si es True, procesa frames en paralelo
        max_workers: Número máximo de workers para procesamiento paralelo
        compute_error: Si es True, calcula y devuelve métricas de error
        visualize: Si es True, genera visualizaciones
        visualize_frames: Lista de índices de frames a visualizar
        output_dir: Directorio para guardar visualizaciones

    Returns:
        DataFrame con los datos del experto normalizados, y opcionalmente un DataFrame de métricas
    """
    if user_data.empty or expert_data.empty:
        raise ValueError("DataFrames de usuario o experto vacíos")
    if len(user_data) != len(expert_data):
        raise ValueError("Distinto número de frames entre usuario y experto")

    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and not config:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Configuración por defecto si es None
    if config is None:
        config = {}

    # Landmarks a utilizar para la normalización
    landmarks_to_use = config.get("normalization_landmarks", None)
    regularization = config.get("normalization_regularization", 0.0)

    # Conservar columna 'frame'
    if "frame" in expert_data.columns:
        frames = expert_data["frame"].copy()
    else:
        frames = pd.Series(range(len(expert_data)), name="frame")

    # Filtrar solo columnas de coordenadas
    exp_coords = expert_data.filter(regex=r"landmark_.*_[xyz]$")
    usr_coords = user_data.filter(regex=r"landmark_.*_[xyz]$")

    # Crear DataFrame para resultado
    normalized = pd.DataFrame(index=exp_coords.index, columns=exp_coords.columns)

    # DataFrame para métricas de error si se solicita
    error_metrics_df = None
    if compute_error:
        error_metrics_df = pd.DataFrame(
            index=range(len(usr_coords)),
            columns=["initial_error", "final_error", "improvement"],
        )

    # Determinar frames a visualizar
    if visualize and visualize_frames is None:
        # Por defecto, visualizar algunos frames representativos
        total_frames = len(usr_coords)
        if total_frames <= 5:
            visualize_frames = list(range(total_frames))
        else:
            visualize_frames = [
                0,
                total_frames // 4,
                total_frames // 2,
                3 * total_frames // 4,
                total_frames - 1,
            ]

    # Crear directorio de salida si es necesario
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Procesar usando paralelismo si se solicita y hay suficientes frames
    if use_parallel and len(usr_coords) > 10:
        logger.info(
            f"Normalizando {len(usr_coords)} frames en paralelo con {max_workers} workers"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Enviar trabajos al pool
            for i in range(len(usr_coords)):
                # Determinar si visualizar este frame
                should_visualize = visualize and (
                    visualize_frames is not None and i in visualize_frames
                )

                # Crear directorio específico para este frame
                frame_output_dir = (
                    os.path.join(output_dir, f"frame_{i}")
                    if should_visualize and output_dir
                    else None
                )

                # Enviar trabajo
                futures.append(
                    executor.submit(
                        normalize_skeleton_with_affine_transform,
                        exp_coords.iloc[i],
                        usr_coords.iloc[i],
                        landmarks_to_use,
                        regularization,
                        compute_error,
                        should_visualize,
                        frame_output_dir,
                    )
                )

            # Procesar resultados a medida que se completan
            for i, future in enumerate(as_completed(futures)):
                try:
                    if compute_error:
                        normalized_frame, error_metrics = future.result()
                        normalized.iloc[i] = normalized_frame

                        if error_metrics:
                            error_metrics_df.loc[i, "initial_error"] = error_metrics[
                                "initial_error"
                            ]
                            error_metrics_df.loc[i, "final_error"] = error_metrics[
                                "final_error"
                            ]
                            error_metrics_df.loc[i, "improvement"] = error_metrics[
                                "improvement"
                            ]
                    else:
                        normalized.iloc[i] = future.result()

                except Exception as e:
                    logger.error(f"Error procesando frame {i}: {e}")
                    normalized.iloc[i] = exp_coords.iloc[
                        i
                    ]  # Usar original en caso de error

    else:
        logger.info(f"Normalizando {len(usr_coords)} frames secuencialmente")

        # Normalizar cada frame secuencialmente
        for i in range(len(usr_coords)):
            # Determinar si visualizar este frame
            should_visualize = visualize and (
                visualize_frames is not None and i in visualize_frames
            )

            # Crear directorio específico para este frame
            frame_output_dir = (
                os.path.join(output_dir, f"frame_{i}")
                if should_visualize and output_dir
                else None
            )

            try:
                if compute_error:
                    normalized_frame, error_metrics = (
                        normalize_skeleton_with_affine_transform(
                            exp_coords.iloc[i],
                            usr_coords.iloc[i],
                            landmarks_to_use,
                            regularization,
                            compute_error,
                            should_visualize,
                            frame_output_dir,
                        )
                    )
                    normalized.iloc[i] = normalized_frame

                    if error_metrics:
                        error_metrics_df.loc[i, "initial_error"] = error_metrics[
                            "initial_error"
                        ]
                        error_metrics_df.loc[i, "final_error"] = error_metrics[
                            "final_error"
                        ]
                        error_metrics_df.loc[i, "improvement"] = error_metrics[
                            "improvement"
                        ]
                else:
                    normalized.iloc[i] = normalize_skeleton_with_affine_transform(
                        exp_coords.iloc[i],
                        usr_coords.iloc[i],
                        landmarks_to_use,
                        regularization,
                        compute_error,
                        should_visualize,
                        frame_output_dir,
                    )
            except Exception as e:
                logger.error(f"Error procesando frame {i}: {e}")
                normalized.iloc[i] = exp_coords.iloc[
                    i
                ]  # Usar original en caso de error

    # Añadir columna de frame
    normalized.insert(0, "frame", frames.values)

    # Si se solicitaron métricas, mostrar resumen
    if compute_error and error_metrics_df is not None:
        avg_improvement = error_metrics_df["improvement"].mean()
        logger.info(f"Mejora promedio en alineación: {avg_improvement:.2f}%")

        if visualize:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(error_metrics_df["initial_error"], label="Error inicial")
                plt.plot(error_metrics_df["final_error"], label="Error final")
                plt.title("Errores de normalización por frame")
                plt.xlabel("Frame")
                plt.ylabel("Error (distancia media)")
                plt.legend()
                plt.grid(True, alpha=0.3)

                if output_dir:
                    error_path = os.path.join(output_dir, "normalization_errors.png")
                    plt.savefig(error_path, dpi=100)
                    logger.info(f"Gráfica de errores guardada en: {error_path}")

                plt.show()
            except Exception as e:
                logger.warning(f"Error al visualizar métricas: {e}")

    logger.info(f"Normalización completada para {len(normalized)} frames")
    return (normalized, error_metrics_df) if compute_error else normalized
