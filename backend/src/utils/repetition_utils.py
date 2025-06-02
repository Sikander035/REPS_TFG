# backend/src/utils/repetition_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import logging
import os
from typing import List, Dict, Tuple, Optional, Union, Any

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#############################################################################
# FUNCIONES DE PROCESAMIENTO DE SEÑALES
#############################################################################


def smoothen_signal(signal, window_size=11, polyorder=2):
    """
    Suaviza una señal utilizando el filtro Savitzky-Golay.

    Args:
        signal: Array con valores de la señal
        window_size: Tamaño de la ventana de suavizado (debe ser impar)
        polyorder: Orden del polinomio para ajuste

    Returns:
        Array con la señal suavizada
    """
    # Asegurar que window_size sea impar y apropiado
    if window_size % 2 == 0:
        window_size += 1

    if len(signal) < window_size:
        # Reducir ventana si la señal es más corta
        actual_window = min(
            (len(signal) - 2 if len(signal) % 2 == 0 else len(signal) - 1),
            5,
        )
        if actual_window < 3:
            # No suavizar si la señal es muy corta
            logger.warning("Señal demasiado corta para suavizar")
            return signal
        else:
            return savgol_filter(
                signal, actual_window, min(polyorder, actual_window - 1)
            )
    else:
        return savgol_filter(signal, window_size, polyorder)


def find_signal_peaks(
    signal,
    prominence=0.2,
    distance=20,
    height=-0.8,
):
    """
    Encuentra picos en una señal.

    Args:
        signal: Array con valores de la señal
        prominence: Prominencia mínima para considerar un pico
        distance: Distancia mínima entre picos
        height: Altura mínima para considerar un pico

    Returns:
        Array con índices de los picos
    """
    peaks, _ = find_peaks(
        signal,
        prominence=prominence,
        distance=distance,
        height=height,
    )

    return peaks


def find_signal_valleys(
    signal,
    prominence=0.2,
    distance=20,
    height=-0.8,
):
    """
    Encuentra valles en una señal.

    Args:
        signal: Array con valores de la señal
        prominence: Prominencia mínima para considerar un valle
        distance: Distancia mínima entre valles
        height: Altura máxima para considerar un valle

    Returns:
        Array con índices de los valles
    """
    # Un valle es un pico en la señal invertida
    valleys, _ = find_peaks(
        -signal,
        prominence=prominence,
        distance=distance,
        height=height,
    )

    return valleys


def prepare_signal_for_peak_detection(data_frame, landmark_columns, axis="y"):
    """
    Prepara una señal para detección de picos a partir de landmarks.

    Args:
        data_frame: DataFrame con datos de landmarks
        landmark_columns: Lista de columnas de landmarks a usar (sin sufijo)
        axis: Eje a analizar ('x', 'y', 'z')

    Returns:
        Array con la señal procesada
    """
    # Verificar que las columnas existen
    valid_columns = []
    for landmark in landmark_columns:
        col = f"{landmark}_{axis}"
        if col in data_frame.columns:
            valid_columns.append(col)

    if not valid_columns:
        raise ValueError(f"No se encontraron landmarks válidos para el eje {axis}")

    # Extraer valores (invertidos para tener valores altos = posición alta)
    # En MediaPipe, las coordenadas Y crecen hacia abajo, por eso se invierte
    if axis == "y":
        return -(data_frame[valid_columns].min(axis=1).values)
    else:
        return data_frame[valid_columns].min(axis=1).values


#############################################################################
# FUNCIONES PARA DETECCIÓN DE REPETICIONES
#############################################################################


def detect_repetitions_from_signal(
    signal,
    prominence=0.2,
    smoothing_window=11,
    polyorder=2,
    positive_distance=20,
    negative_distance=50,
    peak_height_threshold=-0.8,
    exercise_type="press",
):
    """
    Detecta repeticiones a partir de una señal.

    Args:
        signal: Array con valores de la señal
        prominence: Prominencia mínima para detectar picos/valles
        smoothing_window: Tamaño de la ventana de suavizado
        polyorder: Orden del polinomio para el filtro
        positive_distance: Distancia mínima entre valles
        negative_distance: Distancia mínima entre picos
        peak_height_threshold: Umbral de altura para picos/valles
        exercise_type: Tipo de ejercicio ('press' o 'pull')

    Returns:
        Lista de diccionarios con info de repeticiones (start_frame, mid_frame, end_frame)
    """
    # Suavizar la señal
    smoothed_signal = smoothen_signal(signal, smoothing_window, polyorder)

    # Añadir valores artificiales al inicio y final para facilitar detección
    padding_value = smoothed_signal.mean()
    padded_signal = np.concatenate(
        ([padding_value - 10], smoothed_signal, [padding_value - 10])
    )

    # Encontrar picos (puntos de inicio/fin de repetición)
    peaks = find_signal_peaks(
        padded_signal,
        prominence=prominence,
        distance=negative_distance,
        height=peak_height_threshold,
    )

    # Encontrar valles (puntos medios de repetición)
    valleys = find_signal_valleys(
        padded_signal,
        prominence=prominence,
        distance=positive_distance,
        height=peak_height_threshold,
    )

    # Ajustar los índices al rango original
    peaks = peaks - 1
    valleys = valleys - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(smoothed_signal))]
    valleys = valleys[(valleys >= 0) & (valleys < len(smoothed_signal))]

    logger.debug(f"Picos encontrados: {len(peaks)}, Valles encontrados: {len(valleys)}")

    # Construir las repeticiones
    repetitions = []
    for i in range(len(peaks) - 1):
        start_frame = peaks[i]
        end_frame = peaks[i + 1]

        # Buscar el valle entre dos picos consecutivos
        if exercise_type == "press":
            # Para ejercicios de empuje, el valle es el punto medio
            valley_candidates = valleys[(valleys > start_frame) & (valleys < end_frame)]
        else:
            # Para ejercicios de tracción, usamos los valles calculados pero interpretados diferente
            valley_candidates = valleys[(valleys > start_frame) & (valleys < end_frame)]

        if len(valley_candidates) > 0:
            mid_frame = valley_candidates[0]
        else:
            # Si no hay valle detectado, usar el punto con valor mínimo/máximo entre los picos
            segment = smoothed_signal[start_frame:end_frame]
            if exercise_type == "press":
                # Para press, el punto más bajo es el medio
                relative_mid = np.argmin(segment)
            else:
                # Para tracción, el punto más alto es el medio
                relative_mid = np.argmax(segment)
            mid_frame = start_frame + relative_mid
            logger.debug(
                f"No se encontró valle entre picos {start_frame}-{end_frame}. "
                f"Usando {mid_frame} como punto medio basado en {exercise_type}."
            )

        repetitions.append(
            {
                "start_frame": int(start_frame),
                "mid_frame": int(mid_frame),
                "end_frame": int(end_frame),
            }
        )

    return repetitions, smoothed_signal


def detect_repetitions_from_landmarks(
    data,
    landmark_columns=None,
    axis="y",
    prominence=0.2,
    smoothing_window=11,
    polyorder=2,
    positive_distance=20,
    negative_distance=50,
    peak_height_threshold=-0.8,
    exercise_type="press",
):
    """
    Detecta repeticiones a partir de landmarks.

    Args:
        data: DataFrame con datos de landmarks
        landmark_columns: Lista de columnas de landmarks a usar
        axis: Eje a analizar ('x', 'y', 'z')
        prominence: Prominencia mínima para detectar picos/valles
        smoothing_window: Tamaño de la ventana de suavizado
        polyorder: Orden del polinomio para el filtro
        positive_distance: Distancia mínima entre valles
        negative_distance: Distancia mínima entre picos
        peak_height_threshold: Umbral de altura para picos/valles
        exercise_type: Tipo de ejercicio ('press' o 'pull')

    Returns:
        Lista de diccionarios con info de repeticiones (start_frame, mid_frame, end_frame)
    """
    # Si no se especifican landmarks, usar muñecas por defecto
    if landmark_columns is None:
        landmark_columns = ["landmark_right_wrist", "landmark_left_wrist"]

    # Construir las columnas a utilizar
    valid_landmark_columns = []
    for landmark in landmark_columns:
        col = f"{landmark}_{axis}"
        if col in data.columns:
            valid_landmark_columns.append(landmark)

    # Si no hay columnas encontradas, usar las muñecas por defecto
    if not valid_landmark_columns:
        default_cols = ["landmark_right_wrist", "landmark_left_wrist"]
        available_cols = [
            col.rsplit("_", 1)[0]
            for col in data.columns
            if col.endswith(f"_{axis}") and col.rsplit("_", 1)[0] in default_cols
        ]

        if not available_cols:
            # Si no hay landmarks de muñecas, buscar cualquier landmark disponible
            all_landmark_cols = [
                col.rsplit("_", 1)[0]
                for col in data.columns
                if col.startswith("landmark_") and col.endswith(f"_{axis}")
            ]
            if all_landmark_cols:
                valid_landmark_columns = [
                    all_landmark_cols[0]
                ]  # Tomar el primero disponible
                logger.warning(f"Usando {valid_landmark_columns[0]} como alternativa")
            else:
                raise ValueError(
                    f"No se encontraron columnas de landmarks para el eje {axis}"
                )
        else:
            valid_landmark_columns = available_cols
            logger.debug(f"Usando columnas por defecto: {valid_landmark_columns}")

    # Validar calidad de datos para los landmarks seleccionados
    nan_ratio = 0
    for landmark in valid_landmark_columns:
        col = f"{landmark}_{axis}"
        if col in data.columns:
            nan_ratio += data[col].isna().mean()

    if len(valid_landmark_columns) > 0:
        nan_ratio /= len(valid_landmark_columns)

    if nan_ratio > 0.2:  # Más de 20% NaNs
        logger.warning(
            f"Alto porcentaje de valores NaN ({nan_ratio:.1%}) en landmarks. "
            "La detección de repeticiones puede ser imprecisa."
        )

    # Extraer la señal a partir de los landmarks
    signal = prepare_signal_for_peak_detection(data, valid_landmark_columns, axis)

    # Detectar repeticiones a partir de la señal
    repetitions, smoothed_signal = detect_repetitions_from_signal(
        signal,
        prominence=prominence,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        positive_distance=positive_distance,
        negative_distance=negative_distance,
        peak_height_threshold=peak_height_threshold,
        exercise_type=exercise_type,
    )

    # Mostrar información sobre repeticiones detectadas
    if repetitions:
        logger.info(f"Detectadas {len(repetitions)} repeticiones en {len(data)} frames")
        for i, rep in enumerate(repetitions):
            logger.debug(
                f"Repetición {i+1}: Inicio={rep['start_frame']}, "
                f"Medio={rep['mid_frame']}, Fin={rep['end_frame']}"
            )
    else:
        logger.warning("No se detectaron repeticiones")

    return repetitions, smoothed_signal


#############################################################################
# FUNCIONES DE VISUALIZACIÓN
#############################################################################


def visualize_repetitions(
    signal,
    repetitions,
    exercise_name=None,
    exercise_type="press",
    output_path=None,
    landmark_name="unknown",
    smoothed_signal=None,
):
    """
    Visualiza las repeticiones detectadas en una señal.

    Args:
        signal: Array con valores de la señal original
        repetitions: Lista de diccionarios con info de repeticiones
        exercise_name: Nombre del ejercicio para el título
        exercise_type: Tipo de ejercicio ('press' o 'pull')
        output_path: Ruta donde guardar la imagen (None = no guardar)
        landmark_name: Nombre del landmark para la etiqueta
        smoothed_signal: Señal suavizada (si es None, se usa la original)

    Returns:
        Figure de matplotlib
    """
    # Usar señal suavizada si está disponible, sino la original
    if smoothed_signal is None:
        smoothed_signal = signal

    fig = plt.figure(figsize=(12, 6))

    # Graficar la señal
    plt.plot(smoothed_signal, linewidth=2, label=f"Posición {landmark_name}")

    # Colores para los eventos
    colors = {"start": "green", "mid": "red", "end": "blue"}

    # Añadir líneas verticales y etiquetas para cada repetición
    for i, rep in enumerate(repetitions):
        # Líneas para inicio, medio y fin
        plt.axvline(
            x=rep["start_frame"],
            color=colors["start"],
            linestyle="--",
            alpha=0.7,
            label="Inicio" if i == 0 else "",
        )

        plt.axvline(
            x=rep["mid_frame"],
            color=colors["mid"],
            linestyle="--",
            alpha=0.7,
            label="Punto medio" if i == 0 else "",
        )

        plt.axvline(
            x=rep["end_frame"],
            color=colors["end"],
            linestyle="--",
            alpha=0.7,
            label="Fin" if i == 0 else "",
        )

        # Añadir número de repetición
        mid_x = (rep["start_frame"] + rep["end_frame"]) / 2
        y_pos = np.max(smoothed_signal) * 0.9
        plt.text(
            mid_x,
            y_pos,
            f"Rep {i+1}",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
        )

        # Sombrear el área de la repetición
        plt.axvspan(
            rep["start_frame"],
            rep["end_frame"],
            alpha=0.1,
            color=f"C{i%10}",
            label=f"Repetición {i+1}" if i < 3 else "",
        )

    # Mejorar presentación
    plt.title(
        f"Detección de Repeticiones - {exercise_name or 'Ejercicio'} ({exercise_type})"
    )
    plt.xlabel("Frame")
    plt.ylabel("Posición (invertida)")
    plt.grid(True, alpha=0.3)

    # Leyenda con las primeras repeticiones solamente (para no saturar)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")

    # Guardar si se especifica directorio
    if output_path is not None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        logger.info(f"Gráfica guardada en: {output_path}")

    plt.tight_layout()

    return fig


#############################################################################
# FUNCIONES DE ANÁLISIS DE REPETICIONES
#############################################################################


def calculate_repetition_metrics(repetitions, signal, fps=30.0, exercise_type="press"):
    """
    Calcula métricas de calidad de las repeticiones.

    Args:
        repetitions: Lista de diccionarios con info de repeticiones
        signal: Array con valores de la señal
        fps: Frames por segundo del video
        exercise_type: Tipo de ejercicio ('press' o 'pull')

    Returns:
        Lista de diccionarios con métricas por repetición
    """
    metrics = []

    for i, rep in enumerate(repetitions):
        start_frame = rep["start_frame"]
        mid_frame = rep["mid_frame"]
        end_frame = rep["end_frame"]

        # Duración total en segundos
        duration = (end_frame - start_frame) / fps

        # Duración de fase concéntrica y excéntrica
        if exercise_type == "press":
            # En press: subida = concéntrica, bajada = excéntrica
            concentric_duration = (mid_frame - start_frame) / fps
            eccentric_duration = (end_frame - mid_frame) / fps
        else:
            # En pull: bajada = concéntrica, subida = excéntrica
            concentric_duration = (end_frame - mid_frame) / fps
            eccentric_duration = (mid_frame - start_frame) / fps

        # Amplitud (diferencia entre valor mínimo y máximo)
        amplitude = np.max(signal[start_frame:end_frame]) - np.min(
            signal[start_frame:end_frame]
        )

        # Simetría (ratio entre fase concéntrica y excéntrica)
        if eccentric_duration > 0:
            con_ecc_ratio = concentric_duration / eccentric_duration
        else:
            con_ecc_ratio = np.nan

        # Velocidad media concéntrica y excéntrica
        if concentric_duration > 0:
            concentric_speed = amplitude / concentric_duration
        else:
            concentric_speed = np.nan

        if eccentric_duration > 0:
            eccentric_speed = amplitude / eccentric_duration
        else:
            eccentric_speed = np.nan

        # Añadir métricas para esta repetición
        metrics.append(
            {
                "repetition": i + 1,
                "start_frame": start_frame,
                "mid_frame": mid_frame,
                "end_frame": end_frame,
                "duration": duration,
                "concentric_duration": concentric_duration,
                "eccentric_duration": eccentric_duration,
                "amplitude": amplitude,
                "con_ecc_ratio": con_ecc_ratio,
                "concentric_speed": concentric_speed,
                "eccentric_speed": eccentric_speed,
            }
        )

    return metrics


def identify_phases_in_repetition(
    repetition, signal, exercise_type="press", num_phases=4
):
    """
    Identifica fases más detalladas dentro de una repetición.

    Args:
        repetition: Diccionario con info de repetición (start_frame, mid_frame, end_frame)
        signal: Array con valores de la señal
        exercise_type: Tipo de ejercicio ('press' o 'pull')
        num_phases: Número de fases a identificar (2, 3 o 4)

    Returns:
        Lista de diccionarios con info de fases
    """
    start_frame = repetition["start_frame"]
    mid_frame = repetition["mid_frame"]
    end_frame = repetition["end_frame"]

    if num_phases == 2:
        # 2 fases: concéntrica y excéntrica
        phases = [
            {"name": "concéntrica", "start": start_frame, "end": mid_frame},
            {"name": "excéntrica", "start": mid_frame, "end": end_frame},
        ]

    elif num_phases == 3:
        # 3 fases: inicial, media, final
        third_length = (end_frame - start_frame) / 3
        phase1_end = int(start_frame + third_length)
        phase2_end = int(start_frame + 2 * third_length)

        phases = [
            {"name": "inicial", "start": start_frame, "end": phase1_end},
            {"name": "media", "start": phase1_end, "end": phase2_end},
            {"name": "final", "start": phase2_end, "end": end_frame},
        ]

    elif num_phases == 4:
        # 4 fases: inicial concéntrica, final concéntrica, inicial excéntrica, final excéntrica
        conc_mid = start_frame + (mid_frame - start_frame) // 2
        ecc_mid = mid_frame + (end_frame - mid_frame) // 2

        phases = [
            {"name": "inicial_concéntrica", "start": start_frame, "end": conc_mid},
            {"name": "final_concéntrica", "start": conc_mid, "end": mid_frame},
            {"name": "inicial_excéntrica", "start": mid_frame, "end": ecc_mid},
            {"name": "final_excéntrica", "start": ecc_mid, "end": end_frame},
        ]

    else:
        # Por defecto, 2 fases
        phases = [
            {"name": "concéntrica", "start": start_frame, "end": mid_frame},
            {"name": "excéntrica", "start": mid_frame, "end": end_frame},
        ]

    # Invertir nombres si el ejercicio es tipo pull
    if exercise_type != "press":
        for phase in phases:
            if "concéntrica" in phase["name"]:
                phase["name"] = phase["name"].replace("concéntrica", "excéntrica")
            elif "excéntrica" in phase["name"]:
                phase["name"] = phase["name"].replace("excéntrica", "concéntrica")

    return phases


def calculate_phase_metrics(phases, signal, fps=30.0):
    """
    Calcula métricas para cada fase de una repetición.

    Args:
        phases: Lista de diccionarios con info de fases
        signal: Array con valores de la señal
        fps: Frames por segundo del video

    Returns:
        Lista de diccionarios con métricas por fase
    """
    phase_metrics = []

    for phase in phases:
        start_frame = phase["start"]
        end_frame = phase["end"]
        name = phase["name"]

        # Extraer segmento de señal para esta fase
        phase_signal = signal[start_frame:end_frame]

        # Duración
        duration = (end_frame - start_frame) / fps

        # Valores extremos
        min_val = np.min(phase_signal)
        max_val = np.max(phase_signal)

        # Cambio neto
        net_change = phase_signal[-1] - phase_signal[0]

        # Velocidad media
        speed = net_change / duration if duration > 0 else np.nan

        # Aceleración media (derivada segunda)
        velocity = np.gradient(phase_signal) * fps
        acceleration = np.gradient(velocity) * fps
        avg_acceleration = np.mean(acceleration)

        # Añadir métricas para esta fase
        phase_metrics.append(
            {
                "name": name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration": duration,
                "min_val": min_val,
                "max_val": max_val,
                "net_change": net_change,
                "speed": speed,
                "avg_acceleration": avg_acceleration,
            }
        )

    return phase_metrics


def calculate_exercise_total_duration(user_repetitions, expert_repetitions):
    """
    Calcula duración total del ejercicio basada en repeticiones reales.

    Args:
        user_repetitions: Lista de repeticiones del usuario
        expert_repetitions: Lista de repeticiones del experto

    Returns:
        dict: {
            'user_frames': frames totales de ejercicio del usuario,
            'expert_frames_scaled': frames esperados del experto para N reps usuario,
            'speed_ratio': expert_frames / user_frames,
            'user_reps': número de repeticiones del usuario,
            'expert_reps': número de repeticiones del experto
        }
    """
    if not user_repetitions or not expert_repetitions:
        return {
            "user_frames": 0,
            "expert_frames_scaled": 0,
            "speed_ratio": 1.0,
            "user_reps": 0,
            "expert_reps": 0,
        }

    # Usuario: desde primera hasta última repetición
    user_start = min(rep["start_frame"] for rep in user_repetitions)
    user_end = max(rep["end_frame"] for rep in user_repetitions)
    user_frames = user_end - user_start

    # Experto: promedio de duración por repetición × número de reps usuario
    expert_durations = [
        rep["end_frame"] - rep["start_frame"] for rep in expert_repetitions
    ]
    expert_avg_duration = np.mean(expert_durations)
    expert_frames_scaled = expert_avg_duration * len(user_repetitions)

    speed_ratio = expert_frames_scaled / user_frames if user_frames > 0 else 1.0

    return {
        "user_frames": int(user_frames),
        "expert_frames_scaled": int(expert_frames_scaled),
        "speed_ratio": speed_ratio,
        "user_reps": len(user_repetitions),
        "expert_reps": len(expert_repetitions),
    }
