import cv2
import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import load_exercise_config


# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def draw_skeleton(
    image,
    landmarks_frame,
    color=(0, 255, 0),
    thickness=2,
    alpha=1.0,
    joint_radius=None,
    connections=None,
    highlight_landmarks=None,
    show_labels=False,
    z_coloring=True,
):
    """
    Dibuja un esqueleto en la imagen utilizando los landmarks de un frame específico.
    Con soporte para transparencia y personalización avanzada.

    Args:
        image: Imagen de OpenCV donde dibujar
        landmarks_frame: DataFrame/Serie con los landmarks de un frame
        color: Color del esqueleto (BGR)
        thickness: Grosor de las líneas
        alpha: Transparencia (1.0 = opaco, 0.0 = totalmente transparente)
        joint_radius: Radio de las articulaciones (None = auto basado en Z)
        connections: Lista de tuplas de conexiones entre landmarks (None = por defecto)
        highlight_landmarks: Lista de landmarks para resaltar
        show_labels: Mostrar etiquetas de landmarks
        z_coloring: Colorear según profundidad Z

    Returns:
        Imagen con el esqueleto dibujado
    """
    # Definir las conexiones de los landmarks para formar el esqueleto
    if connections is None:
        connections = [
            # Torso
            ("landmark_left_shoulder", "landmark_right_shoulder"),
            ("landmark_left_shoulder", "landmark_left_hip"),
            ("landmark_right_shoulder", "landmark_right_hip"),
            ("landmark_left_hip", "landmark_right_hip"),
            # Brazos
            ("landmark_left_shoulder", "landmark_left_elbow"),
            ("landmark_left_elbow", "landmark_left_wrist"),
            ("landmark_right_shoulder", "landmark_right_elbow"),
            ("landmark_right_elbow", "landmark_right_wrist"),
            # Piernas (si están presentes)
            ("landmark_left_hip", "landmark_left_knee"),
            ("landmark_right_hip", "landmark_right_knee"),
            ("landmark_left_knee", "landmark_left_ankle"),
            ("landmark_right_knee", "landmark_right_ankle"),
        ]

    # Obtener el ancho y alto de la imagen
    h, w = image.shape[:2]

    # Crear una capa transparente para dibujar el esqueleto
    overlay = image.copy()

    # Extraer landmarks usando columnas existentes en el DataFrame
    landmark_dict = {}
    for col in (
        landmarks_frame.index
        if isinstance(landmarks_frame, pd.Series)
        else landmarks_frame.columns
    ):
        if col.endswith("_x"):
            joint = col.rsplit("_", 1)[0]
            if f"{joint}_y" in (
                landmarks_frame.index
                if isinstance(landmarks_frame, pd.Series)
                else landmarks_frame.columns
            ):
                x = float(
                    landmarks_frame[f"{joint}_x"]
                    if isinstance(landmarks_frame, pd.Series)
                    else landmarks_frame[
                        0, landmarks_frame.columns.get_loc(f"{joint}_x")
                    ]
                )
                y = float(
                    landmarks_frame[f"{joint}_y"]
                    if isinstance(landmarks_frame, pd.Series)
                    else landmarks_frame[
                        0, landmarks_frame.columns.get_loc(f"{joint}_y")
                    ]
                )
                z = 0.0

                if f"{joint}_z" in (
                    landmarks_frame.index
                    if isinstance(landmarks_frame, pd.Series)
                    else landmarks_frame.columns
                ):
                    z = float(
                        landmarks_frame[f"{joint}_z"]
                        if isinstance(landmarks_frame, pd.Series)
                        else landmarks_frame[
                            0, landmarks_frame.columns.get_loc(f"{joint}_z")
                        ]
                    )

                # Convertir coordenadas normalizadas (0-1) a píxeles de la imagen
                x_px = int(x * w)
                y_px = int(y * h)

                landmark_dict[joint] = (x_px, y_px, z)

    # Dibujar las conexiones
    for start_joint, end_joint in connections:
        if start_joint in landmark_dict and end_joint in landmark_dict:
            start_point = landmark_dict[start_joint][:2]  # Solo x,y
            end_point = landmark_dict[end_joint][:2]  # Solo x,y

            # Verificar que los puntos estén dentro de la imagen
            if (
                0 <= start_point[0] < w
                and 0 <= start_point[1] < h
                and 0 <= end_point[0] < w
                and 0 <= end_point[1] < h
            ):
                # Calcular color basado en Z si está habilitado
                if z_coloring:
                    z_avg = (
                        landmark_dict[start_joint][2] + landmark_dict[end_joint][2]
                    ) / 2
                    # Ajustar color según profundidad: más cercano (z negativo) = más brillante
                    z_factor = min(1.5, max(0.5, 1.0 - z_avg))
                    line_color = tuple(min(255, int(c * z_factor)) for c in color)
                else:
                    line_color = color

                # Verificar si es una conexión resaltada
                if highlight_landmarks and (
                    start_joint in highlight_landmarks
                    or end_joint in highlight_landmarks
                ):
                    # Usar un grosor mayor y un color más saturado para las conexiones resaltadas
                    highlight_thickness = thickness + 2
                    highlight_color = (0, 255, 255)  # Amarillo para resaltar
                    cv2.line(
                        overlay,
                        start_point,
                        end_point,
                        highlight_color,
                        highlight_thickness,
                    )
                else:
                    cv2.line(overlay, start_point, end_point, line_color, thickness)

    # Dibujar los puntos del esqueleto
    for joint, (x, y, z) in landmark_dict.items():
        if 0 <= x < w and 0 <= y < h:
            # Determinar el radio del círculo
            if joint_radius is not None:
                radius = joint_radius
            else:
                # Radio proporcional a la profundidad (z)
                radius = max(3, min(7, int(5 - z * 2))) if z else 5

            # Ajustar color según Z si está habilitado
            if z_coloring:
                # Ajustar color según profundidad: más cercano = más brillante
                z_factor = min(1.5, max(0.5, 1.0 - z))
                point_color = tuple(min(255, int(c * z_factor)) for c in color)
            else:
                point_color = color

            # Verificar si es un landmark resaltado
            if highlight_landmarks and joint in highlight_landmarks:
                highlight_radius = radius + 2
                highlight_color = (0, 255, 255)  # Amarillo para resaltar
                cv2.circle(overlay, (x, y), highlight_radius, highlight_color, -1)
            else:
                cv2.circle(overlay, (x, y), radius, point_color, -1)

            # Mostrar etiquetas si está habilitado
            if show_labels:
                # Extraer solo el nombre del landmark (sin el prefijo)
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

    # Combinar la capa transparente con la imagen original
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image


def extract_frame_ranges(
    user_data_orig, user_processed_data, detect_repetitions_fn=None
):
    """
    Extrae el rango de frames originales que corresponden a los datos procesados.

    Args:
        user_data_orig: DataFrame con los datos originales del usuario
        user_processed_data: DataFrame con los datos procesados/sincronizados
        detect_repetitions_fn: Función para detectar repeticiones (opcional)

    Returns:
        original_frames: Lista de números de frame originales que corresponden
                        a los frames procesados
    """
    # Comprobar si hay columna frame en los datos originales
    if "frame" not in user_data_orig.columns:
        logger.warning("No hay columna 'frame' en los datos originales")
        return list(range(len(user_processed_data)))

    # Obtener repeticiones detectadas en los datos originales
    try:
        if detect_repetitions_fn is None:
            try:
                from src.core.data_segmentation.detect_repetitions import (
                    detect_repetitions,
                )

                detect_repetitions_fn = detect_repetitions
            except ImportError:
                logger.warning("No se pudo importar la función detect_repetitions")
                return list(range(len(user_processed_data)))

        repetitions_orig = detect_repetitions_fn(user_data_orig, plot_graph=False)

        if not repetitions_orig:
            logger.warning("No se detectaron repeticiones en los datos originales")
            return list(range(len(user_processed_data)))

        # Extraer el rango de frames que cubren las repeticiones
        min_frame = min(
            rep["start_frame"]
            for rep in repetitions_orig
            if not np.isnan(rep["start_frame"])
        )
        max_frame = max(
            rep["end_frame"]
            for rep in repetitions_orig
            if not np.isnan(rep["end_frame"])
        )

        logger.info(f"Rango de frames originales: {min_frame} a {max_frame}")

        # Crear un mapeo entre frames procesados y originales
        # Asumiendo una correspondencia lineal dentro del rango detectado
        frame_count = len(user_processed_data)
        original_frames = np.linspace(min_frame, max_frame, frame_count, dtype=int)

        return original_frames

    except Exception as e:
        logger.error(f"Error al extraer rangos de frame: {e}")
        return list(range(len(user_processed_data)))


def generate_dual_skeleton_video(
    original_video_path,
    user_data,
    expert_data,
    output_video_path,
    original_user_data=None,  # Datos originales del usuario
    user_color=(0, 255, 0),  # Verde para el usuario
    expert_color=(0, 0, 255),  # Rojo para el experto
    user_alpha=0.7,  # Transparencia del usuario
    expert_alpha=0.9,  # Transparencia del experto
    user_thickness=2,
    expert_thickness=3,  # Mayor grosor para el experto
    resize_factor=1.0,
    show_progress=True,
    text_info=True,
    exercise_name=None,
    config=None,
    config_path="config_expanded.json",
    highlight_landmarks=None,
    show_labels=False,
    frame_range=None,
    fps_factor=1.0,
    output_quality=95,
):
    """
    Genera un video con los esqueletos del usuario y del experto superpuestos.
    Versión mejorada con personalización avanzada.

    Args:
        original_video_path: Ruta al video original del usuario
        user_data: DataFrame con los landmarks del usuario procesados
        expert_data: DataFrame con los landmarks del experto procesados y alineados
        output_video_path: Ruta donde guardar el video resultante
        original_user_data: DataFrame con los datos originales del usuario (para mapeo de frames)
        user_color: Color para el esqueleto del usuario (BGR)
        expert_color: Color para el esqueleto del experto (BGR)
        user_alpha: Transparencia del esqueleto del usuario
        expert_alpha: Transparencia del esqueleto del experto
        user_thickness: Grosor de las líneas del usuario
        expert_thickness: Grosor de las líneas del experto
        resize_factor: Factor de redimensionamiento del video
        show_progress: Mostrar barra de progreso
        text_info: Mostrar información textual en el video
        exercise_name: Nombre del ejercicio para cargar configuración
        config: Configuración personalizada (opcional)
        config_path: Ruta al archivo de configuración
        highlight_landmarks: Lista de landmarks a resaltar
        show_labels: Mostrar etiquetas de landmarks
        frame_range: Rango de frames a procesar (None = todos)
        fps_factor: Factor para ajustar FPS (1.0 = original)
        output_quality: Calidad del video de salida (0-100)
    """
    # Verificar que los DataFrames tengan la misma longitud
    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los datos de usuario y experto deben tener la misma longitud. "
            f"Usuario: {len(user_data)}, Experto: {len(expert_data)}"
        )

    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and config is None:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Si config sigue siendo None, inicializar vacío
    if config is None:
        config = {}

    # Personalizar conexiones desde configuración si existe
    connections = config.get("visualization_connections", None)

    # Abrir el video original
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video en {original_video_path}")

    # Obtener propiedades del video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = original_fps * fps_factor
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Video original: {width}x{height}, {original_fps} FPS, {total_frames} frames"
    )
    logger.info(f"Video salida: {width}x{height}, {fps} FPS")

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Códec para MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)

    # Determinar el mapeo de frames procesados a frames originales
    if original_user_data is not None:
        try:
            original_frames = extract_frame_ranges(original_user_data, user_data)
        except Exception as e:
            logger.error(f"Error al extraer rango de frames: {e}")
            original_frames = list(range(len(user_data)))
    else:
        # Si no tenemos datos originales, asumimos que son frames secuenciales
        original_frames = list(range(len(user_data)))

    # Aplicar rango de frames si se especifica
    if frame_range is not None:
        start_idx, end_idx = frame_range
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(original_frames):
            end_idx = len(original_frames)

        original_frames = original_frames[start_idx:end_idx]
        logger.info(f"Procesando rango de frames reducido: {start_idx} a {end_idx}")

    logger.info(f"Generando video con {len(original_frames)} frames")

    # Tiempo estimado (para barra de progreso)
    start_time = time.time()

    # Usar tqdm para mostrar una barra de progreso
    frame_iterator = (
        tqdm(
            enumerate(original_frames),
            total=len(original_frames),
            desc="Generando video",
        )
        if show_progress
        else enumerate(original_frames)
    )

    try:
        for i, orig_frame_idx in frame_iterator:
            # Posicionar el video en el frame original correspondiente
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"No se pudo leer el frame {orig_frame_idx} del video")
                continue

            # Redimensionar el frame si es necesario
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (width, height))

            # Obtener los datos de landmarks para este frame procesado
            if i < len(user_data) and i < len(expert_data):
                user_landmarks = user_data.iloc[i]
                expert_landmarks = expert_data.iloc[i]

                # Dibujar primero el esqueleto del experto (para que esté detrás)
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

                # Luego dibujar el esqueleto del usuario
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

                # Añadir información de texto
                if text_info:
                    # Frame info
                    cv2.putText(
                        frame,
                        f"Frame orig: {orig_frame_idx} / Frame sync: {i}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # User/Expert legend
                    user_color_rgb = tuple(reversed(user_color))  # BGR a RGB
                    cv2.putText(
                        frame,
                        f"Usuario: {exercise_name or ''}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        user_color,
                        2,
                    )

                    expert_color_rgb = tuple(reversed(expert_color))  # BGR a RGB
                    cv2.putText(
                        frame,
                        f"Experto",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        expert_color,
                        2,
                    )

                    # Tiempo transcurrido y estimado
                    elapsed = time.time() - start_time
                    if i > 10:  # Esperar unos frames para estimación más precisa
                        frames_remaining = len(original_frames) - i
                        time_per_frame = elapsed / i
                        eta = frames_remaining * time_per_frame

                        cv2.putText(
                            frame,
                            f"Tiempo: {int(elapsed//60)}:{int(elapsed%60):02d} | ETA: {int(eta//60)}:{int(eta%60):02d}",
                            (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 200, 200),
                            1,
                        )

            # Escribir el frame procesado al video de salida
            out.write(frame)

        # Liberar recursos
        cap.release()
        out.release()

        logger.info(f"Video generado exitosamente: {output_video_path}")

        return output_video_path

    except Exception as e:
        logger.error(f"Error al generar video: {e}")
        # Asegurar que los recursos se liberan
        cap.release()
        out.release()
        raise


def visualize_frame_dual_skeletons(
    original_image,
    user_frame_data,
    expert_frame_data,
    user_color=(0, 255, 0),
    expert_color=(0, 0, 255),
    user_alpha=0.7,
    expert_alpha=0.9,
    user_thickness=2,
    expert_thickness=3,
    show_image=True,
    save_path=None,
    title="Comparación de Esqueletos",
    highlight_landmarks=None,
    show_labels=False,
    connections=None,
    config=None,
    exercise_name=None,
    config_path="config_expanded.json",
    figure_size=(12, 10),
    dpi=100,
):
    """
    Visualiza un solo frame con los esqueletos del usuario y experto superpuestos.
    Versión mejorada con más opciones de personalización.

    Args:
        original_image: Ruta a la imagen o array de imagen
        user_frame_data: DataFrame o Serie con datos del usuario
        expert_frame_data: DataFrame o Serie con datos del experto
        user_color: Color del usuario (BGR)
        expert_color: Color del experto (BGR)
        user_alpha: Transparencia del usuario
        expert_alpha: Transparencia del experto
        user_thickness: Grosor de líneas del usuario
        expert_thickness: Grosor de líneas del experto
        show_image: Mostrar imagen (True/False)
        save_path: Ruta para guardar imagen
        title: Título de la visualización
        highlight_landmarks: Lista de landmarks a resaltar
        show_labels: Mostrar etiquetas de landmarks
        connections: Lista de conexiones entre landmarks
        config: Configuración personalizada
        exercise_name: Nombre del ejercicio para cargar configuración
        config_path: Ruta al archivo de configuración
        figure_size: Tamaño de la figura (ancho, alto) en pulgadas
        dpi: Resolución de la imagen

    Returns:
        Imagen procesada
    """
    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and config is None:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Si config sigue siendo None, inicializar vacío
    if config is None:
        config = {}

    # Obtener conexiones específicas desde configuración si existe
    if connections is None:
        connections = config.get("visualization_connections", None)

    # Cargar la imagen si se proporciona una ruta
    if isinstance(original_image, str):
        image = cv2.imread(original_image)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen desde {original_image}")
    else:
        image = original_image.copy()

    # Dibujar primero el esqueleto del experto (para que esté detrás)
    draw_skeleton(
        image,
        expert_frame_data,
        color=expert_color,
        thickness=expert_thickness,
        alpha=expert_alpha,
        connections=connections,
        highlight_landmarks=highlight_landmarks,
        show_labels=show_labels,
    )

    # Luego dibujar el esqueleto del usuario
    draw_skeleton(
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
        image,
        "Usuario",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        user_color,
        2,
    )
    cv2.putText(
        image,
        "Experto",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        expert_color,
        2,
    )

    if exercise_name:
        cv2.putText(
            image,
            f"Ejercicio: {exercise_name}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # Mostrar la imagen
    if show_image:
        plt.figure(figsize=figure_size, dpi=dpi)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # Guardar la imagen
    if save_path:
        directory = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(save_path, image)
        logger.info(f"Imagen guardada en {save_path}")

    return image


def generate_comparison_grid(
    original_video_path,
    user_data,
    expert_data,
    output_path,
    num_frames=6,
    frame_selection="uniform",
    frame_indices=None,
    user_color=(0, 255, 0),
    expert_color=(0, 0, 255),
    grid_size=None,
    title="Comparación de Movimiento",
    highlight_landmarks=None,
    connections=None,
    config=None,
    exercise_name=None,
    config_path="config_expanded.json",
    original_user_data=None,
    figure_size=(15, 10),
    dpi=150,
    show_labels=False,
):
    """
    Genera una cuadrícula de comparación con múltiples frames del movimiento.

    Args:
        original_video_path: Ruta al video original
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        output_path: Ruta para guardar la imagen
        num_frames: Número de frames a incluir en la cuadrícula
        frame_selection: 'uniform', 'key_points' o 'manual'
        frame_indices: Lista de índices específicos (para 'manual')
        user_color: Color para el usuario (BGR)
        expert_color: Color para el experto (BGR)
        grid_size: Tamaño de la cuadrícula (filas, columnas) o None para auto
        title: Título de la cuadrícula
        highlight_landmarks: Lista de landmarks a resaltar
        connections: Lista de conexiones personalizadas
        config: Configuración personalizada
        exercise_name: Nombre del ejercicio
        config_path: Ruta al archivo de configuración
        original_user_data: DataFrame con datos originales del usuario
        figure_size: Tamaño de figura (ancho, alto) en pulgadas
        dpi: Resolución de la imagen
        show_labels: Mostrar etiquetas en los landmarks

    Returns:
        Ruta a la imagen generada
    """
    logger.info(f"Generando cuadrícula de comparación con {num_frames} frames")

    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and config is None:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Si config sigue siendo None, inicializar vacío
    if config is None:
        config = {}

    # Obtener conexiones específicas desde configuración si existe
    if connections is None:
        connections = config.get("visualization_connections", None)

    # Verificar video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {original_video_path}")

    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Mapear frames sincrónicos a frames originales
    if original_user_data is not None:
        try:
            original_frames = extract_frame_ranges(original_user_data, user_data)
        except Exception as e:
            logger.error(f"Error al extraer rango de frames: {e}")
            original_frames = list(range(len(user_data)))
    else:
        original_frames = list(range(min(len(user_data), total_frames)))

    # Seleccionar frames según el método especificado
    selected_indices = []

    if frame_selection == "manual" and frame_indices is not None:
        # Usar índices específicos proporcionados
        selected_indices = [i for i in frame_indices if 0 <= i < len(original_frames)]
        if len(selected_indices) > num_frames:
            selected_indices = selected_indices[:num_frames]
        elif len(selected_indices) < num_frames:
            logger.warning(
                f"Solo se proporcionaron {len(selected_indices)} índices válidos"
            )

    elif frame_selection == "key_points":
        # Intentar identificar frames clave (inicio, medio, final de repeticiones)
        try:
            from src.core.data_segmentation.detect_repetitions import detect_repetitions

            repetitions = detect_repetitions(user_data, plot_graph=False)
            if repetitions:
                key_frames = []
                for rep in repetitions:
                    # Añadir inicio, medio y fin de cada repetición
                    key_frames.extend(
                        [
                            rep["start_frame"],
                            (
                                rep["mid_frame"]
                                if not np.isnan(rep["mid_frame"])
                                else (rep["start_frame"] + rep["end_frame"]) // 2
                            ),
                            rep["end_frame"],
                        ]
                    )

                # Seleccionar frames distribuidos del conjunto de key_frames
                if len(key_frames) <= num_frames:
                    selected_indices = key_frames
                else:
                    # Tomar una muestra uniforme de los key_frames
                    step = len(key_frames) / num_frames
                    selected_indices = [
                        key_frames[min(len(key_frames) - 1, int(i * step))]
                        for i in range(num_frames)
                    ]
            else:
                logger.warning(
                    "No se detectaron repeticiones, usando selección uniforme"
                )
                # Fallback a uniforme
                step = len(original_frames) / num_frames
                selected_indices = [int(i * step) for i in range(num_frames)]
        except Exception as e:
            logger.error(f"Error al detectar key_points: {e}")
            # Fallback a uniforme
            step = len(original_frames) / num_frames
            selected_indices = [int(i * step) for i in range(num_frames)]

    else:  # 'uniform' (default)
        # Seleccionar frames uniformemente distribuidos
        step = len(original_frames) / num_frames
        selected_indices = [int(i * step) for i in range(num_frames)]

    # Asegurar que los índices son únicos y ordenados
    selected_indices = sorted(list(set(selected_indices)))
    logger.info(f"Frames seleccionados: {selected_indices}")

    # Determinar tamaño de cuadrícula
    if grid_size is None:
        # Calcular automáticamente: preferir más columnas que filas
        cols = int(np.ceil(np.sqrt(len(selected_indices))))
        rows = int(np.ceil(len(selected_indices) / cols))
        grid_size = (rows, cols)
    else:
        rows, cols = grid_size

    # Crear figura
    fig, axes = plt.subplots(rows, cols, figsize=figure_size, dpi=dpi)
    fig.suptitle(f"{title} - {exercise_name or 'Ejercicio'}", fontsize=16)

    # Asegurar que axes sea un array 2D aunque solo haya una fila o columna
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Procesar cada frame seleccionado
    for i, (ax_idx, frame_idx) in enumerate(
        zip(np.ndindex((rows, cols)), selected_indices)
    ):
        if i >= len(selected_indices):
            # Ocultar axes no utilizados
            axes[ax_idx].axis("off")
            continue

        # Obtener el frame original correspondiente
        orig_frame_idx = original_frames[frame_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"No se pudo leer el frame {orig_frame_idx}")
            continue

        # Dibujar esqueletos
        try:
            # Experto primero (para que esté detrás)
            frame = draw_skeleton(
                frame,
                expert_data.iloc[frame_idx],
                color=expert_color,
                thickness=3,
                alpha=0.9,
                connections=connections,
                highlight_landmarks=highlight_landmarks,
                show_labels=show_labels,
            )

            # Usuario encima
            frame = draw_skeleton(
                frame,
                user_data.iloc[frame_idx],
                color=user_color,
                thickness=2,
                alpha=0.7,
                connections=connections,
                highlight_landmarks=highlight_landmarks,
                show_labels=show_labels,
            )

            # Añadir número de frame como subtítulo
            cv2.putText(
                frame,
                f"Frame {orig_frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        except Exception as e:
            logger.error(f"Error al dibujar frame {frame_idx}: {e}")

        # Mostrar en la cuadrícula
        axes[ax_idx].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title(f"Frame {frame_idx} (orig: {orig_frame_idx})")

    # Añadir leyenda
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=tuple(c / 255 for c in reversed(user_color)),
            markersize=10,
            label="Usuario",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=tuple(c / 255 for c in reversed(expert_color)),
            markersize=10,
            label="Experto",
        ),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08)  # Ajustar para título y leyenda

    # Guardar imagen
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Cuadrícula guardada en: {output_path}")

    # Liberar recursos
    cap.release()
    plt.close(fig)

    return output_path


def generate_side_by_side_comparison(
    user_video_path,
    expert_video_path,
    user_data,
    expert_data,
    output_video_path,
    user_color=(0, 255, 0),
    expert_color=(0, 0, 255),
    resize_factor=1.0,
    show_progress=True,
    text_info=True,
    fps_factor=1.0,
    frame_range=None,
    highlight_landmarks=None,
    connections=None,
    config=None,
    exercise_name=None,
    config_path="config_expanded.json",
):
    """
    Genera un video de comparación lado a lado del usuario y experto.

    Args:
        user_video_path: Ruta al video del usuario
        expert_video_path: Ruta al video del experto
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        output_video_path: Ruta para guardar el video
        user_color: Color para el esqueleto del usuario
        expert_color: Color para el esqueleto del experto
        resize_factor: Factor de redimensionamiento
        show_progress: Mostrar barra de progreso
        text_info: Mostrar información textual
        fps_factor: Factor para ajustar FPS
        frame_range: Rango de frames a procesar
        highlight_landmarks: Lista de landmarks a resaltar
        connections: Lista de conexiones personalizadas
        config: Configuración personalizada
        exercise_name: Nombre del ejercicio
        config_path: Ruta al archivo de configuración

    Returns:
        Ruta al video generado
    """
    logger.info("Generando comparación lado a lado")

    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and config is None:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Si config sigue siendo None, inicializar vacío
    if config is None:
        config = {}

    # Obtener conexiones específicas desde configuración si existe
    if connections is None:
        connections = config.get("visualization_connections", None)

    # Abrir los videos
    user_cap = cv2.VideoCapture(user_video_path)
    expert_cap = cv2.VideoCapture(expert_video_path)

    if not user_cap.isOpened():
        raise ValueError(f"No se pudo abrir el video del usuario: {user_video_path}")
    if not expert_cap.isOpened():
        raise ValueError(f"No se pudo abrir el video del experto: {expert_video_path}")

    # Obtener propiedades de los videos
    user_fps = user_cap.get(cv2.CAP_PROP_FPS)
    expert_fps = expert_cap.get(cv2.CAP_PROP_FPS)
    user_width = int(user_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    user_height = int(user_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    expert_width = int(expert_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    expert_height = int(expert_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)

    # Calcular el tamaño del video combinado
    output_width = user_width + expert_width
    output_height = max(user_height, expert_height)
    output_fps = max(user_fps, expert_fps) * fps_factor

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path, fourcc, output_fps, (output_width, output_height), True
    )

    # Calcular número de frames a procesar
    num_frames = min(len(user_data), len(expert_data))

    # Aplicar rango de frames si se especifica
    if frame_range is not None:
        start_idx, end_idx = frame_range
        if start_idx < 0:
            start_idx = 0
        if end_idx > num_frames:
            end_idx = num_frames

        num_frames = end_idx - start_idx
        logger.info(
            f"Procesando rango de frames: {start_idx} a {end_idx}, total {num_frames}"
        )
    else:
        start_idx = 0
        end_idx = num_frames

    # Tiempo estimado (para barra de progreso)
    start_time = time.time()

    # Usar tqdm para mostrar una barra de progreso
    frame_iterator = (
        tqdm(range(start_idx, end_idx), total=num_frames, desc="Generando video")
        if show_progress
        else range(start_idx, end_idx)
    )

    try:
        for i in frame_iterator:
            # Leer frame del usuario y experto
            user_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                user_data.loc[i, "frame"] if "frame" in user_data.columns else i,
            )
            expert_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                expert_data.loc[i, "frame"] if "frame" in expert_data.columns else i,
            )

            ret_user, user_frame = user_cap.read()
            ret_expert, expert_frame = expert_cap.read()

            if not ret_user or not ret_expert:
                logger.warning(f"No se pudo leer frame {i}")
                continue

            # Redimensionar si es necesario
            if resize_factor != 1.0:
                user_frame = cv2.resize(user_frame, (user_width, user_height))
                expert_frame = cv2.resize(expert_frame, (expert_width, expert_height))

            # Dibujar esqueletos
            user_frame = draw_skeleton(
                user_frame,
                user_data.iloc[i],
                color=user_color,
                thickness=2,
                alpha=0.7,
                connections=connections,
                highlight_landmarks=highlight_landmarks,
            )

            expert_frame = draw_skeleton(
                expert_frame,
                expert_data.iloc[i],
                color=expert_color,
                thickness=3,
                alpha=0.9,
                connections=connections,
                highlight_landmarks=highlight_landmarks,
            )

            # Crear frame combinado
            combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

            # Añadir video del usuario
            combined_frame[:user_height, :user_width] = user_frame

            # Añadir video del experto
            combined_frame[:expert_height, user_width:] = expert_frame

            # Añadir línea divisoria
            cv2.line(
                combined_frame,
                (user_width, 0),
                (user_width, output_height),
                (255, 255, 255),
                2,
            )

            # Añadir información de texto
            if text_info:
                # Etiquetas
                cv2.putText(
                    combined_frame,
                    "Usuario",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    user_color,
                    2,
                )

                cv2.putText(
                    combined_frame,
                    "Experto",
                    (user_width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    expert_color,
                    2,
                )

                # Número de frame
                cv2.putText(
                    combined_frame,
                    f"Frame: {i}",
                    (10, output_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                # Tiempo
                elapsed = time.time() - start_time
                if (
                    i > start_idx + 10
                ):  # Esperar unos frames para estimación más precisa
                    frames_remaining = end_idx - i
                    time_per_frame = elapsed / (i - start_idx)
                    eta = frames_remaining * time_per_frame

                    cv2.putText(
                        combined_frame,
                        f"ETA: {int(eta//60)}:{int(eta%60):02d}",
                        (output_width - 150, output_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                    )

            # Escribir frame
            out.write(combined_frame)

        # Liberar recursos
        user_cap.release()
        expert_cap.release()
        out.release()

        logger.info(f"Video de comparación lado a lado generado: {output_video_path}")
        return output_video_path

    except Exception as e:
        logger.error(f"Error al generar video: {e}")
        # Liberar recursos
        user_cap.release()
        expert_cap.release()
        out.release()
        raise


def generate_error_heatmap(
    user_data,
    expert_data,
    output_path,
    error_metric="distance",
    selected_landmarks=None,
    title="Mapa de calor de error",
    exercise_name=None,
    config=None,
    config_path="config_expanded.json",
    colormap="hot",
    figure_size=(14, 8),
    dpi=100,
    show_plot=True,
):
    """
    Genera un mapa de calor que muestra el error entre el usuario y el experto.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        output_path: Ruta para guardar la imagen
        error_metric: Métrica de error ('distance', 'angle', 'velocity')
        selected_landmarks: Lista de landmarks a incluir (None = todos)
        title: Título del gráfico
        exercise_name: Nombre del ejercicio
        config: Configuración personalizada
        config_path: Ruta al archivo de configuración
        colormap: Mapa de colores para el heatmap
        figure_size: Tamaño de la figura en pulgadas
        dpi: Resolución de la imagen
        show_plot: Mostrar el gráfico después de guardarlo

    Returns:
        Ruta a la imagen generada
    """
    logger.info(f"Generando mapa de calor de error con métrica: {error_metric}")

    # Verificar que los DataFrames tengan la misma longitud
    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los DataFrames tienen diferentes longitudes: usuario={len(user_data)}, experto={len(expert_data)}"
        )

    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and config is None:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Si config sigue siendo None, inicializar vacío
    if config is None:
        config = {}

    # Seleccionar landmarks a analizar
    if selected_landmarks is None:
        # Usar landmarks de la configuración o encontrar todos los disponibles
        selected_landmarks = config.get("landmarks", None)

        if selected_landmarks is None:
            # Detectar automáticamente los landmarks disponibles
            landmark_cols = [
                col
                for col in user_data.columns
                if col.endswith("_x") and col.startswith("landmark_")
            ]
            selected_landmarks = [col[:-2] for col in landmark_cols]  # Quitar el _x

    # Preparar matriz de error
    num_frames = len(user_data)
    num_landmarks = len(selected_landmarks)
    error_matrix = np.zeros((num_frames, num_landmarks))

    # Calcular error para cada landmark y frame
    for i, landmark in enumerate(selected_landmarks):
        for axis in ["x", "y", "z"]:
            col = f"{landmark}_{axis}"
            if col in user_data.columns and col in expert_data.columns:
                if error_metric == "distance":
                    # Error de posición
                    error_matrix[:, i] += np.square(
                        user_data[col].values - expert_data[col].values
                    )
                elif error_metric == "velocity":
                    # Error de velocidad (derivada)
                    user_vel = np.gradient(user_data[col].values)
                    expert_vel = np.gradient(expert_data[col].values)
                    error_matrix[:, i] += np.square(user_vel - expert_vel)
                elif error_metric == "acceleration":
                    # Error de aceleración (segunda derivada)
                    user_vel = np.gradient(user_data[col].values)
                    expert_vel = np.gradient(expert_data[col].values)
                    user_accel = np.gradient(user_vel)
                    expert_accel = np.gradient(expert_vel)
                    error_matrix[:, i] += np.square(user_accel - expert_accel)

        # Raíz cuadrada para distancia euclidiana
        if error_metric == "distance":
            error_matrix[:, i] = np.sqrt(error_matrix[:, i])

    # Crear etiquetas más legibles para los landmarks
    landmark_labels = [
        landmark.replace("landmark_", "") for landmark in selected_landmarks
    ]

    # Generar mapa de calor
    plt.figure(figsize=figure_size, dpi=dpi)

    # Usar imshow para el mapa de calor
    im = plt.imshow(error_matrix.T, aspect="auto", cmap=colormap)

    # Configurar ejes
    plt.xlabel("Frame")
    plt.ylabel("Landmark")
    plt.colorbar(label=f"Error ({error_metric})")

    # Ajustar etiquetas del eje Y (landmarks)
    plt.yticks(np.arange(len(landmark_labels)), landmark_labels)

    # Añadir título
    plt.title(f"{title} - {exercise_name or 'Ejercicio'}")

    # Añadir líneas para repeticiones si están disponibles
    try:
        from detect_repetitions import detect_repetitions

        repetitions = detect_repetitions(user_data, plot_graph=False)

        if repetitions:
            for rep in repetitions:
                start_frame = rep["start_frame"]
                mid_frame = rep["mid_frame"] if not np.isnan(rep["mid_frame"]) else None
                end_frame = rep["end_frame"]

                plt.axvline(x=start_frame, color="green", linestyle="--", alpha=0.7)
                if mid_frame is not None:
                    plt.axvline(x=mid_frame, color="red", linestyle="--", alpha=0.7)
                plt.axvline(x=end_frame, color="blue", linestyle="--", alpha=0.7)
    except Exception as e:
        logger.warning(f"No se pudieron detectar repeticiones: {e}")

    plt.tight_layout()

    # Guardar imagen
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Mapa de calor guardado en: {output_path}")

    # Mostrar gráfico si se solicita
    if show_plot:
        plt.show()
    else:
        plt.close()

    return output_path


# Función de entrada principal para procesamiento de visualizaciones
def process_visualizations(
    user_data,
    expert_data,
    original_video_path,
    output_dir="visualizations",
    types=None,
    exercise_name=None,
    config=None,
    config_path="config_expanded.json",
    original_user_data=None,
):
    """
    Función principal para generar múltiples visualizaciones desde un solo lugar.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        original_video_path: Ruta al video original
        output_dir: Directorio para guardar visualizaciones
        types: Lista de tipos de visualización a generar
        exercise_name: Nombre del ejercicio
        config: Configuración personalizada
        config_path: Ruta al archivo de configuración
        original_user_data: DataFrame con datos originales del usuario

    Returns:
        Diccionario con rutas a las visualizaciones generadas
    """
    # Tipos de visualización disponibles
    all_types = ["video", "frame", "grid", "heatmap", "timeline"]

    # Si no se especifican tipos, usar todos
    if types is None:
        types = all_types

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Resultados
    results = {}

    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and config is None:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Generar visualizaciones según los tipos solicitados
    try:
        # Video con esqueletos superpuestos
        if "video" in types:
            video_path = os.path.join(
                output_dir, f"{exercise_name or 'exercise'}_video.mp4"
            )
            results["video"] = generate_dual_skeleton_video(
                original_video_path=original_video_path,
                user_data=user_data,
                expert_data=expert_data,
                output_video_path=video_path,
                original_user_data=original_user_data,
                exercise_name=exercise_name,
                config=config,
            )

        # Frame individual
        if "frame" in types:
            # Seleccionar un frame representativo (mitad de los datos)
            mid_frame = len(user_data) // 2

            # Obtener frame del video
            cap = cv2.VideoCapture(original_video_path)
            if original_user_data is not None:
                # Mapear a frame original
                original_frames = extract_frame_ranges(original_user_data, user_data)
                orig_frame_idx = original_frames[mid_frame]
            else:
                orig_frame_idx = mid_frame

            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_path = os.path.join(
                    output_dir, f"{exercise_name or 'exercise'}_frame.png"
                )
                results["frame"] = visualize_frame_dual_skeletons(
                    original_image=frame,
                    user_frame_data=user_data.iloc[mid_frame],
                    expert_frame_data=expert_data.iloc[mid_frame],
                    save_path=frame_path,
                    exercise_name=exercise_name,
                    config=config,
                )

        # Cuadrícula de comparación
        if "grid" in types:
            grid_path = os.path.join(
                output_dir, f"{exercise_name or 'exercise'}_grid.png"
            )
            results["grid"] = generate_comparison_grid(
                original_video_path=original_video_path,
                user_data=user_data,
                expert_data=expert_data,
                output_path=grid_path,
                num_frames=6,
                frame_selection="key_points",
                exercise_name=exercise_name,
                config=config,
                original_user_data=original_user_data,
            )

        # Mapa de calor de error
        if "heatmap" in types:
            heatmap_path = os.path.join(
                output_dir, f"{exercise_name or 'exercise'}_heatmap.png"
            )
            results["heatmap"] = generate_error_heatmap(
                user_data=user_data,
                expert_data=expert_data,
                output_path=heatmap_path,
                exercise_name=exercise_name,
                config=config,
                show_plot=False,
            )

        # Línea de tiempo de error
        if "timeline" in types:
            # Calcular error total por frame
            error = np.zeros(len(user_data))

            for col in user_data.columns:
                if col.endswith(("_x", "_y", "_z")) and col in expert_data.columns:
                    error += np.square(user_data[col].values - expert_data[col].values)

            error = np.sqrt(error)

            # Crear gráfico
            plt.figure(figsize=(12, 6), dpi=100)
            plt.plot(error)
            plt.title(f"Error por Frame - {exercise_name or 'Ejercicio'}")
            plt.xlabel("Frame")
            plt.ylabel("Error (distancia Euclidiana)")
            plt.grid(True, alpha=0.3)

            # Añadir líneas para repeticiones si están disponibles
            try:
                from src.core.data_segmentation.detect_repetitions import (
                    detect_repetitions,
                )

                repetitions = detect_repetitions(user_data, plot_graph=False)

                if repetitions:
                    for rep in repetitions:
                        start_frame = rep["start_frame"]
                        mid_frame = (
                            rep["mid_frame"] if not np.isnan(rep["mid_frame"]) else None
                        )
                        end_frame = rep["end_frame"]

                        plt.axvline(
                            x=start_frame,
                            color="green",
                            linestyle="--",
                            alpha=0.7,
                            label="Inicio",
                        )
                        if mid_frame is not None:
                            plt.axvline(
                                x=mid_frame,
                                color="red",
                                linestyle="--",
                                alpha=0.7,
                                label="Medio",
                            )
                        plt.axvline(
                            x=end_frame,
                            color="blue",
                            linestyle="--",
                            alpha=0.7,
                            label="Fin",
                        )

                        # Solo añadir etiquetas para el primer conjunto
                        if rep == repetitions[0]:
                            plt.legend()
            except Exception as e:
                logger.warning(f"No se pudieron detectar repeticiones: {e}")

            timeline_path = os.path.join(
                output_dir, f"{exercise_name or 'exercise'}_timeline.png"
            )
            plt.savefig(timeline_path, dpi=100, bbox_inches="tight")
            plt.close()

            results["timeline"] = timeline_path

    except Exception as e:
        logger.error(f"Error al generar visualizaciones: {e}")

    return results
