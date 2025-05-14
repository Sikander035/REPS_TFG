import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def detect_repetitions(
    data,
    prominence=0.3,
    smoothing_window=11,
    polyorder=2,
    positive_distance=20,
    negative_distance=50,
    height_threshold=-0.8,
):
    """
    Detecta las repeticiones basadas en la posición Y de las muñecas y agrega el frame de altura mínima.

    Args:
        data (pd.DataFrame): DataFrame con las columnas de landmarks.
        prominence (float): Prominencia mínima para considerar un pico o valle.
        distance (int): Distancia mínima medida en frames entre picos o valles.
        smoothing_window (int): Tamaño de la ventana para suavizar la señal.
        polyorder (int): Orden del polinomio para el filtro de Savitzky-Golay.

    Returns:
        list[dict]: Lista de diccionarios con información sobre las repeticiones.
    """
    # Extraer las posiciones Y de las muñecas e invertir la señal
    wrist_y = -(
        data[["landmark_right_wrist_y", "landmark_left_wrist_y"]].min(axis=1).values
    )

    # Suavizar la señal
    smoothed_wrist_y = savgol_filter(wrist_y, smoothing_window, polyorder)

    # Añadir valores artificiales al inicio y al final
    padding_value = smoothed_wrist_y.mean()
    padded_signal = np.concatenate(
        ([padding_value - 10], smoothed_wrist_y, [padding_value - 10])
    )

    # Encontrar picos usando la señal extendida
    peaks, _ = find_peaks(
        padded_signal,
        prominence=prominence,
        distance=negative_distance,
        height=height_threshold,
    )

    # Encontrar valles usando la señal extendida (invertida)
    valleys, _ = find_peaks(
        -padded_signal,
        prominence=prominence,
        distance=positive_distance,
        height=height_threshold,
    )

    # Ajustar los índices de los picos y valles al rango original
    peaks = peaks - 1
    valleys = valleys - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(smoothed_wrist_y))]
    valleys = valleys[(valleys >= 0) & (valleys < len(smoothed_wrist_y))]

    # Construir las repeticiones
    repetitions = []
    for i in range(len(peaks) - 1):
        start_frame = peaks[i]
        end_frame = peaks[i + 1]

        # Buscar el primer valle entre dos picos consecutivos
        valley_candidates = valleys[(valleys > start_frame) & (valleys < end_frame)]
        if len(valley_candidates) > 0:
            mid_frame = valley_candidates[0]
        else:
            mid_frame = np.nan

        repetitions.append(
            {
                "start_frame": start_frame,
                "mid_frame": mid_frame,
                "end_frame": end_frame,
            }
        )

    return repetitions
