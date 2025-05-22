"""
Utilidades para visualización de esqueletos y datos de movimiento.
Proporciona funciones comunes usadas en varios tipos de visualizaciones.
"""

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# Importar el config_manager singleton
from src.config.config_manager import config_manager

# Configurar logging
logger = logging.getLogger("visualization")
logger.setLevel(logging.INFO)


def draw_skeleton(
    image: np.ndarray,
    landmarks_frame: Union[pd.Series, pd.DataFrame],
    color: Tuple[int, int, int],
    thickness: int,
    alpha: float,
    connections: List[Tuple[str, str]],
    highlight_landmarks: Optional[List[str]] = None,
    show_labels: bool = False,
) -> np.ndarray:
    """
    Dibuja un esqueleto en la imagen utilizando landmarks de un frame.

    Args:
        image: Imagen de OpenCV donde dibujar
        landmarks_frame: DataFrame/Serie con landmarks de un frame
        color: Color del esqueleto (BGR) - OBLIGATORIO
        thickness: Grosor de las líneas - OBLIGATORIO
        alpha: Transparencia (1.0 = opaco) - OBLIGATORIO
        connections: Lista de conexiones entre landmarks - OBLIGATORIO
        highlight_landmarks: Lista de landmarks para resaltar (opcional)
        show_labels: Mostrar etiquetas de landmarks - OBLIGATORIO

    Returns:
        Imagen con el esqueleto dibujado

    Raises:
        ValueError: Si faltan parámetros obligatorios
    """
    if image is None:
        raise ValueError("image es obligatorio")
    if landmarks_frame is None or landmarks_frame.empty:
        raise ValueError("landmarks_frame es obligatorio y no puede estar vacío")
    if not connections:
        raise ValueError("connections es obligatorio y no puede estar vacío")
    if not isinstance(color, (tuple, list)) or len(color) != 3:
        raise ValueError("color debe ser una tupla/lista de 3 elementos (BGR)")
    if not isinstance(thickness, int) or thickness <= 0:
        raise ValueError("thickness debe ser un entero positivo")
    if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
        raise ValueError("alpha debe ser un número entre 0 y 1")

    h, w = image.shape[:2]
    overlay = image.copy()
    landmark_dict = {}

    # Extraer coordenadas de landmarks
    is_series = isinstance(landmarks_frame, pd.Series)

    for col in landmarks_frame.index if is_series else landmarks_frame.columns:
        if not col.endswith("_x"):
            continue

        joint = col.rsplit("_", 1)[0]
        y_col = f"{joint}_y"
        z_col = f"{joint}_z"

        # Verificar que existan coordenadas y/z
        if y_col not in (
            landmarks_frame.index if is_series else landmarks_frame.columns
        ):
            continue

        # Extraer valores de coordenadas
        try:
            if is_series:
                x = float(landmarks_frame[col])
                y = float(landmarks_frame[y_col])
                z = (
                    float(landmarks_frame[z_col])
                    if z_col in landmarks_frame.index
                    else 0.0
                )
            else:
                x = float(landmarks_frame[0, landmarks_frame.columns.get_loc(col)])
                y = float(landmarks_frame[0, landmarks_frame.columns.get_loc(y_col)])
                z = (
                    float(landmarks_frame[0, landmarks_frame.columns.get_loc(z_col)])
                    if z_col in landmarks_frame.columns
                    else 0.0
                )

            # Convertir a píxeles
            x_px, y_px = int(x * w), int(y * h)
            landmark_dict[joint] = (x_px, y_px, z)
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"Error procesando landmark {joint}: {e}")
            continue

    if not landmark_dict:
        raise ValueError("No se pudieron extraer landmarks válidos del frame")

    # Dibujar conexiones
    highlight_color = (0, 255, 255)  # Amarillo para resaltar

    for start_joint, end_joint in connections:
        if start_joint not in landmark_dict or end_joint not in landmark_dict:
            continue

        start_point = landmark_dict[start_joint][:2]
        end_point = landmark_dict[end_joint][:2]

        # Verificar que los puntos estén dentro de la imagen
        if not (
            0 <= start_point[0] < w
            and 0 <= start_point[1] < h
            and 0 <= end_point[0] < w
            and 0 <= end_point[1] < h
        ):
            continue

        # Determinar color y grosor
        use_highlight = highlight_landmarks and (
            start_joint in highlight_landmarks or end_joint in highlight_landmarks
        )

        if use_highlight:
            cv2.line(overlay, start_point, end_point, highlight_color, thickness + 2)
        else:
            # Calcular z promedio para ajustar color
            z_avg = (landmark_dict[start_joint][2] + landmark_dict[end_joint][2]) / 2
            z_factor = min(1.5, max(0.5, 1.0 - z_avg))
            line_color = tuple(min(255, int(c * z_factor)) for c in color)
            cv2.line(overlay, start_point, end_point, line_color, thickness)

    # Dibujar puntos (articulaciones)
    for joint, (x, y, z) in landmark_dict.items():
        if not (0 <= x < w and 0 <= y < h):
            continue

        # Determinar radio del círculo
        radius = max(3, min(7, int(5 - z * 2))) if z else 5

        # Calcular color basado en profundidad
        z_factor = min(1.5, max(0.5, 1.0 - z))
        point_color = tuple(min(255, int(c * z_factor)) for c in color)

        # Verificar si es un landmark resaltado
        if highlight_landmarks and joint in highlight_landmarks:
            cv2.circle(overlay, (x, y), radius + 2, highlight_color, -1)
        else:
            cv2.circle(overlay, (x, y), radius, point_color, -1)

        # Mostrar etiquetas si está habilitado
        if show_labels:
            label = joint.replace("landmark_", "")
            cv2.putText(
                overlay,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    # Combinar overlay con la imagen original según transparencia
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image


def extract_frame_ranges(
    user_data_orig: pd.DataFrame, user_processed_data: pd.DataFrame
) -> List[int]:
    """
    Mapea frames procesados a frames del video original.

    Args:
        user_data_orig: DataFrame con datos originales del usuario
        user_processed_data: DataFrame con datos procesados/sincronizados

    Returns:
        Lista de números de frame originales

    Raises:
        ValueError: Si faltan datos obligatorios
    """
    if user_data_orig is None or user_data_orig.empty:
        raise ValueError("user_data_orig es obligatorio y no puede estar vacío")
    if user_processed_data is None or user_processed_data.empty:
        raise ValueError("user_processed_data es obligatorio y no puede estar vacío")

    # Verificar que hay columna 'frame' en datos originales
    if "frame" not in user_data_orig.columns:
        raise ValueError("Columna 'frame' no encontrada en user_data_orig")

    # Obtener frame mínimo y máximo
    min_frame = user_data_orig["frame"].min()
    max_frame = user_data_orig["frame"].max()

    if pd.isna(min_frame) or pd.isna(max_frame):
        raise ValueError("Valores NaN encontrados en la columna 'frame'")

    # Crear mapeo
    frame_count = len(user_processed_data)
    original_frames = np.linspace(min_frame, max_frame, frame_count, dtype=int)

    logger.info(f"Mapeo de frames: {min_frame} → {max_frame}, {frame_count} frames")
    return original_frames.tolist()


def save_visualization(fig, output_path: str, dpi: int = 100) -> None:
    """
    Guarda una figura y asegura que el directorio exista.

    Args:
        fig: Figura matplotlib - OBLIGATORIO
        output_path: Ruta donde guardar la imagen - OBLIGATORIO
        dpi: Resolución

    Raises:
        ValueError: Si faltan parámetros obligatorios
    """
    if fig is None:
        raise ValueError("fig es obligatorio")
    if not output_path:
        raise ValueError("output_path es obligatorio")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi debe ser un entero positivo")

    try:
        # Asegurar que el directorio existe
        directory = Path(output_path).parent
        directory.mkdir(parents=True, exist_ok=True)

        # Guardar la figura
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Visualización guardada: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error guardando visualización: {e}")


def select_frames_for_visualization(data: pd.DataFrame, num_frames: int) -> List[int]:
    """
    Selecciona frames representativos para visualización.

    Args:
        data: DataFrame con datos - OBLIGATORIO
        num_frames: Número de frames a seleccionar - OBLIGATORIO

    Returns:
        Lista de índices de frames seleccionados

    Raises:
        ValueError: Si faltan parámetros obligatorios
    """
    if data is None or data.empty:
        raise ValueError("data es obligatorio y no puede estar vacío")
    if not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("num_frames debe ser un entero positivo")
    if num_frames > len(data):
        raise ValueError(
            f"num_frames ({num_frames}) no puede ser mayor que el número de frames disponibles ({len(data)})"
        )

    # Selección uniforme
    step = len(data) / num_frames
    return [int(i * step) for i in range(num_frames)]
