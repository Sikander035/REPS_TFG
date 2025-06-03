"""
Visualizaciones duales de movimiento corporal - THREAD-SAFE VERSION.
Proporciona funciones para generar videos y capturas con dos esqueletos.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import logging
import time

# CRÍTICO: Configurar matplotlib ANTES que cualquier import de plotting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.matplotlib_config import (
    ThreadSafeMatplotlib,
    ensure_matplotlib_thread_safety,
)

# Ahora importar matplotlib
import matplotlib.pyplot as plt

# Importar funciones de utilidad
from src.utils.visualization_utils import (
    draw_skeleton,
    extract_frame_ranges,
    save_visualization,
)

# Importar el config_manager singleton
from src.config.config_manager import config_manager

# Configurar logging
logger = logging.getLogger("visualization.dual")
logger.setLevel(logging.INFO)


def generate_dual_skeleton_video(
    original_video_path: str,
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    output_video_path: str,
    config_path: str,
    original_user_data: pd.DataFrame = None,
    exercise_frame_range: tuple = None,
    frame_range: tuple = None,
    fps_factor: float = 1.0,
) -> str:
    """
    Genera video con esqueletos del usuario y experto superpuestos.
    THREAD-SAFE VERSION con cleanup automático.
    """
    # Verificaciones de entrada obligatorias
    if not original_video_path:
        raise ValueError("original_video_path es obligatorio")
    if user_data is None or user_data.empty:
        raise ValueError("user_data es obligatorio y no puede estar vacío")
    if expert_data is None or expert_data.empty:
        raise ValueError("expert_data es obligatorio y no puede estar vacío")
    if not output_video_path:
        raise ValueError("output_video_path es obligatorio")
    if not config_path:
        raise ValueError("config_path es obligatorio")

    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los datos tienen longitudes diferentes: usuario={len(user_data)}, experto={len(expert_data)}"
        )

    if not isinstance(fps_factor, (int, float)) or fps_factor <= 0:
        raise ValueError("fps_factor debe ser un número positivo")

    # Cargar configuración usando el config_manager singleton
    try:
        viz_config = config_manager.get_global_visualization_config(config_path)
        connections = config_manager.get_global_connections(config_path)
    except Exception as e:
        raise RuntimeError(f"Error cargando configuración: {e}")

    # Variables de OpenCV para cleanup
    cap = None
    out = None

    try:
        # Obtener parámetros de configuración
        user_color = tuple(viz_config["user_color"])
        expert_color = tuple(viz_config["expert_color"])
        user_alpha = viz_config["user_alpha"]
        expert_alpha = viz_config["expert_alpha"]
        user_thickness = viz_config["user_thickness"]
        expert_thickness = viz_config["expert_thickness"]
        show_progress = viz_config["show_progress"]
        text_info = viz_config["text_info"]
        show_labels = viz_config["show_labels"]
        resize_factor = viz_config["resize_factor"]
        highlight_landmarks = viz_config.get("highlight_landmarks")

        # Abrir video original
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {original_video_path}")

        # Obtener propiedades del video
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            raise RuntimeError("No se pudo obtener FPS del video")

        fps = original_fps * fps_factor
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)

        if width <= 0 or height <= 0:
            raise RuntimeError("Dimensiones de video inválidas")

        # Crear directorio de salida
        output_dir = Path(output_video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configurar escritor de video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)

        if not out.isOpened():
            raise RuntimeError("No se pudo inicializar el escritor de video")

        # Mapear frames procesados a frames originales
        try:
            original_frames = extract_frame_ranges(
                original_user_data if original_user_data is not None else user_data,
                user_data,
                exercise_frame_range,
            )
        except Exception as e:
            raise RuntimeError(f"Error mapeando frames: {e}")

        # Aplicar rango si se especifica
        if frame_range is not None:
            if not isinstance(frame_range, (tuple, list)) or len(frame_range) != 2:
                raise ValueError("frame_range debe ser una tupla de 2 elementos")

            start_idx, end_idx = frame_range
            if not isinstance(start_idx, int) or not isinstance(end_idx, int):
                raise ValueError("frame_range debe contener enteros")

            start_idx = max(0, start_idx)
            end_idx = min(len(original_frames), end_idx)

            if start_idx >= end_idx:
                raise ValueError(
                    "frame_range inválido: start_idx debe ser menor que end_idx"
                )

            original_frames = original_frames[start_idx:end_idx]

        # Número total de frames a procesar
        total_frames = len(original_frames)
        if total_frames == 0:
            raise ValueError("No hay frames para procesar")

        logger.info(f"Generando video con {total_frames} frames ({fps:.1f} FPS)")

        # Iniciar tiempo para cálculos de ETA
        start_time = time.time()
        last_update = start_time
        update_interval = 2.0

        # Procesar frames
        frame_iterator = (
            tqdm(enumerate(original_frames), total=total_frames, desc="Generando video")
            if show_progress
            else enumerate(original_frames)
        )

        for i, orig_frame_idx in frame_iterator:
            # Actualizar progreso ocasionalmente
            current_time = time.time()
            if current_time - last_update >= update_interval and not show_progress:
                elapsed = current_time - start_time
                if i > 10:
                    frames_remaining = total_frames - i
                    time_per_frame = elapsed / i
                    eta = frames_remaining * time_per_frame
                    logger.info(
                        f"Progreso: {i/total_frames*100:.1f}% - ETA: {int(eta//60)}:{int(eta%60):02d}"
                    )
                last_update = current_time

            # Leer frame original
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"No se pudo leer frame {orig_frame_idx}")
                continue

            # Redimensionar si es necesario
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (width, height))

            # Obtener landmarks para este frame
            if i < len(user_data) and i < len(expert_data):
                user_landmarks = user_data.iloc[i]
                expert_landmarks = expert_data.iloc[i]

                # Dibujar esqueletos (experto primero para que esté detrás)
                frame = draw_skeleton(
                    frame,
                    expert_landmarks,
                    color=expert_color,
                    thickness=expert_thickness,
                    alpha=expert_alpha,
                    connections=connections,
                    highlight_landmarks=highlight_landmarks,
                    show_labels=show_labels,
                )

                frame = draw_skeleton(
                    frame,
                    user_landmarks,
                    color=user_color,
                    thickness=user_thickness,
                    alpha=user_alpha,
                    connections=connections,
                    highlight_landmarks=highlight_landmarks,
                    show_labels=show_labels,
                )

                # Añadir información textual
                if text_info:
                    cv2.putText(
                        frame,
                        f"Frame: {orig_frame_idx}/{i}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "Usuario",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        user_color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        "Experto",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        expert_color,
                        2,
                    )

            # Escribir frame al video
            out.write(frame)

        # Calcular tiempo total
        total_time = time.time() - start_time
        logger.info(
            f"Video generado exitosamente: {output_video_path} "
            f"({total_frames} frames en {total_time:.1f}s)"
        )

        return output_video_path

    except Exception as e:
        logger.error(f"Error generando video: {e}")
        raise RuntimeError(f"Error generando video: {e}")

    finally:
        # CLEANUP CRÍTICO: Liberar recursos OpenCV
        if cap is not None:
            cap.release()
            logger.debug("🧹 VideoCapture liberado")

        if out is not None:
            out.release()
            logger.debug("🧹 VideoWriter liberado")

        # Forzar liberación de recursos de OpenCV
        cv2.destroyAllWindows()


def visualize_frame_dual_skeletons(
    original_image: Union[np.ndarray, str],
    user_frame_data: pd.Series,
    expert_frame_data: pd.Series,
    config_path: str,
    save_path: str = None,
    show_image: bool = False,
    title: str = "Comparación de Esqueletos",
) -> np.ndarray:
    """
    Visualiza un frame con esqueletos superpuestos.
    THREAD-SAFE VERSION con cleanup automático.
    """
    # Verificaciones de entrada obligatorias
    if original_image is None:
        raise ValueError("original_image es obligatorio")
    if user_frame_data is None or user_frame_data.empty:
        raise ValueError("user_frame_data es obligatorio y no puede estar vacío")
    if expert_frame_data is None or expert_frame_data.empty:
        raise ValueError("expert_frame_data es obligatorio y no puede estar vacío")
    if not config_path:
        raise ValueError("config_path es obligatorio")

    # Cargar configuración usando el config_manager singleton
    try:
        viz_config = config_manager.get_global_visualization_config(config_path)
        connections = config_manager.get_global_connections(config_path)
    except Exception as e:
        raise RuntimeError(f"Error cargando configuración: {e}")

    # Obtener parámetros de configuración
    user_color = tuple(viz_config["user_color"])
    expert_color = tuple(viz_config["expert_color"])
    user_alpha = viz_config["user_alpha"]
    expert_alpha = viz_config["expert_alpha"]
    user_thickness = viz_config["user_thickness"]
    expert_thickness = viz_config["expert_thickness"]
    show_labels = viz_config["show_labels"]
    highlight_landmarks = viz_config.get("highlight_landmarks")

    # Cargar la imagen si es una ruta
    if isinstance(original_image, str):
        image = cv2.imread(original_image)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {original_image}")
    else:
        image = original_image.copy()

    try:
        # Dibujar esqueletos (experto primero para que esté detrás)
        image = draw_skeleton(
            image,
            expert_frame_data,
            color=expert_color,
            thickness=expert_thickness,
            alpha=expert_alpha,
            connections=connections,
            highlight_landmarks=highlight_landmarks,
            show_labels=show_labels,
        )

        image = draw_skeleton(
            image,
            user_frame_data,
            color=user_color,
            thickness=user_thickness,
            alpha=user_alpha,
            connections=connections,
            highlight_landmarks=highlight_landmarks,
            show_labels=show_labels,
        )

        # Añadir leyenda
        cv2.putText(
            image, "Usuario", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, user_color, 2
        )
        cv2.putText(
            image, "Experto", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, expert_color, 2
        )

        # Mostrar la imagen (thread-safe)
        if show_image:
            with ThreadSafeMatplotlib():
                plt.figure(figsize=(12, 10))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title(title)
                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=100, bbox_inches="tight")

                plt.show()

        # Guardar la imagen si se especifica ruta (sin mostrar)
        elif save_path:
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, image)
                logger.info(f"Imagen guardada: {save_path}")
            except Exception as e:
                raise RuntimeError(f"Error guardando imagen: {e}")

        return image

    except Exception as e:
        raise RuntimeError(f"Error dibujando esqueletos: {e}")
