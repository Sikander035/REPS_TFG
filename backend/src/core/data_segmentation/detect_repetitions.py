import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import logging
import os
from typing import List, Dict, Tuple, Optional, Union, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.repetition_utils import (
    detect_repetitions_from_landmarks,
    visualize_repetitions,
    calculate_repetition_metrics,
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_repetitions(
    data: pd.DataFrame,
    prominence: float = 0.2,
    smoothing_window: int = 11,
    polyorder: int = 2,
    positive_distance: int = 20,
    negative_distance: int = 50,
    peak_height_threshold: float = -0.8,
    plot_graph: bool = False,
    config: Optional[Dict[str, Any]] = None,
    exercise_name: Optional[str] = None,
    config_path: str = "config.json",
    output_dir: Optional[str] = None,
) -> List[Dict[str, int]]:
    """
    Detecta las repeticiones basadas en la posición de landmarks configurables
    y agrega el frame de altura mínima o máxima como punto medio.

    Args:
        data (pd.DataFrame): DataFrame con las columnas de landmarks.
        prominence (float): Prominencia mínima para considerar un pico o valle.
        smoothing_window (int): Tamaño de la ventana para suavizar la señal.
        polyorder (int): Orden del polinomio para el filtro de Savitzky-Golay.
        positive_distance (int): Distancia mínima entre valles.
        negative_distance (int): Distancia mínima entre picos.
        peak_height_threshold (float): Umbral de altura para considerar un pico.
        plot_graph (bool): Si es True, genera una gráfica de las repeticiones detectadas.
        config (dict, optional): Configuración personalizada. Si es None y exercise_name no es None,
                               se cargará del archivo de configuración.
        exercise_name (str, optional): Nombre del ejercicio para cargar configuración si config es None.
        config_path (str): Ruta al archivo de configuración expandida.
        output_dir (str, optional): Directorio para guardar visualizaciones. Si None, usa directorio actual.

    Returns:
        list[dict]: Lista de diccionarios con información sobre las repeticiones.
    """
    # Cargar configuración SOLO si no se proporciona config
    if config is None and exercise_name is not None:
        try:
            logger.debug(f"Cargando configuración para ejercicio: {exercise_name}")
            from src.config.config_manager import load_exercise_config

            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
        except Exception as e:
            logger.warning(f"Error al cargar config para {exercise_name}: {e}")
            config = {}

    # Si sigue siendo None, inicializar como diccionario vacío
    if config is None:
        config = {}

    # Extraer parámetros de config o usar los valores por defecto proporcionados
    prominence = config.get("rep_prominence", prominence)
    smoothing_window = config.get("rep_smoothing_window", smoothing_window)
    polyorder = config.get("rep_polyorder", polyorder)
    positive_distance = config.get("rep_positive_distance", positive_distance)
    negative_distance = config.get("rep_negative_distance", negative_distance)
    peak_height_threshold = config.get(
        "rep_peak_height_threshold", peak_height_threshold
    )

    # Asegurar que smoothing_window sea impar
    if smoothing_window % 2 == 0:
        smoothing_window += 1

    # Obtener landmarks y eje de la configuración
    division_landmarks = config.get(
        "division_landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
    )
    axis = config.get("division_axis", "y")
    exercise_type = config.get("exercise_type", "press")

    # Usar la función del módulo repetition_utils para detectar repeticiones
    repetitions, smoothed_signal = detect_repetitions_from_landmarks(
        data,
        landmark_columns=division_landmarks,
        axis=axis,
        prominence=prominence,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        positive_distance=positive_distance,
        negative_distance=negative_distance,
        peak_height_threshold=peak_height_threshold,
        exercise_type=exercise_type,
    )

    # Visualizar los resultados si se solicita
    if plot_graph:
        landmark_name = division_landmarks[0] if division_landmarks else "unknown"

        # Determinar la ruta de salida
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            exercise_str = exercise_name or "exercise"
            filename = f"repetitions_{exercise_str}_{len(repetitions)}reps.png"
            output_path = os.path.join(output_dir, filename)

        visualize_repetitions(
            signal=None,  # No es necesario pasar la señal original
            repetitions=repetitions,
            exercise_name=exercise_name,
            exercise_type=exercise_type,
            output_path=output_path,
            landmark_name=landmark_name,
            smoothed_signal=smoothed_signal,
        )

    return repetitions
