# backend/src/core/landmark_utils.py
import numpy as np
import pandas as pd
import logging
from scipy.linalg import lstsq
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Set

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#############################################################################
# FUNCIONES DE EXTRACCIÓN Y MANIPULACIÓN DE LANDMARKS
#############################################################################


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


def get_landmark_coordinates(frame_data, landmark_name):
    """
    Extrae las coordenadas 3D de un landmark específico.

    Args:
        frame_data: Serie o DataFrame con datos de landmarks
        landmark_name: Nombre del landmark (sin sufijo _x, _y, _z)

    Returns:
        np.array: Array [x, y, z] con las coordenadas, o None si no existe
    """
    if isinstance(frame_data, pd.DataFrame) and len(frame_data) > 0:
        frame_data = frame_data.iloc[0]  # Usar el primer frame si es DataFrame

    try:
        x = frame_data[f"{landmark_name}_x"]
        y = frame_data[f"{landmark_name}_y"]
        z = frame_data[f"{landmark_name}_z"]

        coords = np.array([x, y, z])
        if np.isnan(coords).any():
            return None
        return coords
    except (KeyError, TypeError):
        return None


def calculate_distance(point1, point2):
    """
    Calcula la distancia euclidiana entre dos puntos en 3D.

    Args:
        point1, point2: Arrays con coordenadas [x, y, z]

    Returns:
        float: Distancia euclidiana
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def calculate_angle(p1, p2, p3):
    """
    Calcula el ángulo entre tres puntos en el espacio 3D.
    El ángulo está centrado en p2.

    Args:
        p1, p2, p3: Puntos 3D (arrays con coordenadas [x, y, z])

    Returns:
        float: Ángulo en grados
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    # Evitar errores de cálculo con vectores nulos
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return 0

    cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


#############################################################################
# FUNCIONES PARA TRANSFORMACIONES AFINES
#############################################################################


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


#############################################################################
# FUNCIONES PARA ALINEACIÓN DE ESQUELETOS
#############################################################################


def align_skeleton_frame(
    expert_landmarks,
    user_landmarks,
    alignment_method="centroid",
    reference_landmarks=None,
    compute_error=False,
):
    """
    Alinea un frame del esqueleto del experto con el del usuario.

    Args:
        expert_landmarks: Serie/diccionario con landmarks del experto
        user_landmarks: Serie/diccionario con landmarks del usuario
        alignment_method: Método de alineación ('centroid', 'shoulders', 'hips', 'custom')
        reference_landmarks: Lista de landmarks a usar para alineación con método 'custom'
        compute_error: Si es True, calcula y devuelve el error de alineación

    Returns:
        Serie/diccionario con landmarks del experto alineados, y opcionalmente error
    """
    # Crear una copia para no modificar los originales
    aligned_expert = (
        expert_landmarks.copy()
        if isinstance(expert_landmarks, pd.Series)
        else expert_landmarks.copy()
    )

    # Determinar landmarks de referencia según el método
    if alignment_method == "shoulders":
        reference_points = ["landmark_left_shoulder", "landmark_right_shoulder"]
    elif alignment_method == "hips":
        reference_points = ["landmark_left_hip", "landmark_right_hip"]
    elif alignment_method == "torso":
        reference_points = [
            "landmark_left_shoulder",
            "landmark_right_shoulder",
            "landmark_left_hip",
            "landmark_right_hip",
        ]
    elif alignment_method == "custom" and reference_landmarks:
        reference_points = reference_landmarks
    else:  # 'centroid' (default) o fallback
        # Usar todos los landmarks disponibles
        reference_points = []
        for col in (
            expert_landmarks.index
            if isinstance(expert_landmarks, pd.Series)
            else expert_landmarks.keys()
        ):
            if col.endswith("_x"):
                landmark = col[:-2]
                if f"{landmark}_y" in (
                    expert_landmarks.index
                    if isinstance(expert_landmarks, pd.Series)
                    else expert_landmarks.keys()
                ) and f"{landmark}_z" in (
                    expert_landmarks.index
                    if isinstance(expert_landmarks, pd.Series)
                    else expert_landmarks.keys()
                ):
                    reference_points.append(landmark)

    # Calcular centroides para usuario y experto
    user_centroid = np.array([0.0, 0.0, 0.0])
    expert_centroid = np.array([0.0, 0.0, 0.0])
    valid_points = 0

    for point in reference_points:
        # Verificar que el landmark existe en ambos datasets
        if all(
            f"{point}_{axis}"
            in (
                expert_landmarks.index
                if isinstance(expert_landmarks, pd.Series)
                else expert_landmarks.keys()
            )
            and f"{point}_{axis}"
            in (
                user_landmarks.index
                if isinstance(user_landmarks, pd.Series)
                else user_landmarks.keys()
            )
            for axis in ["x", "y", "z"]
        ):

            # Extraer coordenadas
            if isinstance(expert_landmarks, pd.Series):
                expert_point = np.array(
                    [
                        expert_landmarks[f"{point}_x"],
                        expert_landmarks[f"{point}_y"],
                        expert_landmarks[f"{point}_z"],
                    ]
                )
                user_point = np.array(
                    [
                        user_landmarks[f"{point}_x"],
                        user_landmarks[f"{point}_y"],
                        user_landmarks[f"{point}_z"],
                    ]
                )
            else:
                expert_point = np.array(
                    [
                        expert_landmarks[f"{point}_x"],
                        expert_landmarks[f"{point}_y"],
                        expert_landmarks[f"{point}_z"],
                    ]
                )
                user_point = np.array(
                    [
                        user_landmarks[f"{point}_x"],
                        user_landmarks[f"{point}_y"],
                        user_landmarks[f"{point}_z"],
                    ]
                )

            # Verificar que no hay NaN
            if not np.isnan(expert_point).any() and not np.isnan(user_point).any():
                expert_centroid += expert_point
                user_centroid += user_point
                valid_points += 1

    # Si no hay puntos válidos, devolver sin cambios
    if valid_points == 0:
        logger.warning("No se encontraron landmarks válidos para alineación")
        return (
            (aligned_expert, {"error": np.nan, "valid_points": 0})
            if compute_error
            else aligned_expert
        )

    # Calcular centroides promedio
    expert_centroid /= valid_points
    user_centroid /= valid_points

    # Calcular vector de desplazamiento
    displacement = user_centroid - expert_centroid

    # Aplicar desplazamiento a todos los landmarks del experto
    error_before = 0
    error_after = 0

    for key in (
        expert_landmarks.index
        if isinstance(expert_landmarks, pd.Series)
        else expert_landmarks.keys()
    ):
        if key.endswith("_x"):
            landmark = key[:-2]
            if f"{landmark}_y" in (
                expert_landmarks.index
                if isinstance(expert_landmarks, pd.Series)
                else expert_landmarks.keys()
            ) and f"{landmark}_z" in (
                expert_landmarks.index
                if isinstance(expert_landmarks, pd.Series)
                else expert_landmarks.keys()
            ):

                # Calcular error antes de alineación (solo para landmarks de referencia)
                if compute_error and landmark in reference_points:
                    if isinstance(expert_landmarks, pd.Series):
                        expert_point = np.array(
                            [
                                expert_landmarks[f"{landmark}_x"],
                                expert_landmarks[f"{landmark}_y"],
                                expert_landmarks[f"{landmark}_z"],
                            ]
                        )
                        user_point = np.array(
                            [
                                user_landmarks[f"{landmark}_x"],
                                user_landmarks[f"{landmark}_y"],
                                user_landmarks[f"{landmark}_z"],
                            ]
                        )
                    else:
                        expert_point = np.array(
                            [
                                expert_landmarks[f"{landmark}_x"],
                                expert_landmarks[f"{landmark}_y"],
                                expert_landmarks[f"{landmark}_z"],
                            ]
                        )
                        user_point = np.array(
                            [
                                user_landmarks[f"{landmark}_x"],
                                user_landmarks[f"{landmark}_y"],
                                user_landmarks[f"{landmark}_z"],
                            ]
                        )

                    error_before += np.sum((expert_point - user_point) ** 2)

                # Aplicar desplazamiento
                if isinstance(aligned_expert, pd.Series):
                    aligned_expert[f"{landmark}_x"] += displacement[0]
                    aligned_expert[f"{landmark}_y"] += displacement[1]
                    aligned_expert[f"{landmark}_z"] += displacement[2]
                else:
                    aligned_expert[f"{landmark}_x"] += displacement[0]
                    aligned_expert[f"{landmark}_y"] += displacement[1]
                    aligned_expert[f"{landmark}_z"] += displacement[2]

                # Calcular error después de alineación (solo para landmarks de referencia)
                if compute_error and landmark in reference_points:
                    if isinstance(aligned_expert, pd.Series):
                        aligned_point = np.array(
                            [
                                aligned_expert[f"{landmark}_x"],
                                aligned_expert[f"{landmark}_y"],
                                aligned_expert[f"{landmark}_z"],
                            ]
                        )
                    else:
                        aligned_point = np.array(
                            [
                                aligned_expert[f"{landmark}_x"],
                                aligned_expert[f"{landmark}_y"],
                                aligned_expert[f"{landmark}_z"],
                            ]
                        )

                    error_after += np.sum((aligned_point - user_point) ** 2)

    # Calcular error promedio (raíz del error cuadrático medio)
    if compute_error and valid_points > 0:
        error_before = np.sqrt(error_before / valid_points)
        error_after = np.sqrt(error_after / valid_points)
        improvement = (
            ((error_before - error_after) / error_before * 100)
            if error_before > 0
            else 0
        )

        error_metrics = {
            "error_before": error_before,
            "error_after": error_after,
            "improvement": improvement,
            "valid_points": valid_points,
            "displacement": displacement,
        }

        return aligned_expert, error_metrics

    return aligned_expert


#############################################################################
# FUNCIONES DE VISUALIZACIÓN
#############################################################################


def visualize_landmarks_3d(landmarks_data, title="Landmarks 3D", output_path=None):
    """
    Visualiza landmarks en 3D.

    Args:
        landmarks_data: DataFrame o Serie con datos de landmarks
        title: Título para la visualización
        output_path: Ruta para guardar la visualización (opcional)

    Returns:
        Figure de matplotlib
    """
    # Extraer landmarks como matriz
    landmark_points, landmark_names = extract_landmarks_as_matrix(landmarks_data)

    if len(landmark_points) == 0:
        logger.warning("No se encontraron landmarks válidos para visualizar")
        return None

    # Crear figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Graficar puntos
    x = landmark_points[:, 0]
    y = landmark_points[:, 1]
    z = landmark_points[:, 2]

    ax.scatter(x, y, z, c="blue", marker="o", s=50)

    # Añadir etiquetas a los puntos
    for i, txt in enumerate(landmark_names):
        ax.text(x[i], y[i], z[i], txt.replace("landmark_", ""), size=8)

    # Configurar ejes y etiquetas
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Guardar si se proporciona una ruta
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        logger.info(f"Visualización guardada en: {output_path}")

    plt.tight_layout()
    return fig
