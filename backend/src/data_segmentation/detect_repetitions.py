import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import logging
import os
from typing import List, Dict, Tuple, Optional, Union, Any

# Importar funciones de configuración si están disponibles
try:
    from config_utils import load_exercise_config
except ImportError:
    # Definir una versión simplificada si el módulo no está disponible
    def load_exercise_config(exercise_name, config_path=None):
        logging.warning(
            f"No se pudo importar config_utils. Usando configuración por defecto para {exercise_name}"
        )
        return {"sync_config": {}}


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
    config_path: str = "config_expanded.json",
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
        config (dict, optional): Configuración personalizada. Si es None, se usa ejercicio o valores por defecto.
        exercise_name (str, optional): Nombre del ejercicio para cargar configuración.
        config_path (str): Ruta al archivo de configuración expandida.
        output_dir (str, optional): Directorio para guardar visualizaciones. Si None, usa directorio actual.

    Returns:
        list[dict]: Lista de diccionarios con información sobre las repeticiones.
    """
    # Cargar configuración basada en nombre de ejercicio si no hay config explícita
    if config is None and exercise_name is not None:
        try:
            logger.debug(f"Cargando configuración para ejercicio: {exercise_name}")
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

    # Construir las columnas a utilizar
    landmark_columns = []
    for landmark in division_landmarks:
        col = f"{landmark}_{axis}"
        if col in data.columns:
            landmark_columns.append(col)

    # Si no hay columnas encontradas, usar las muñecas por defecto
    if not landmark_columns:
        default_cols = [f"landmark_right_wrist_{axis}", f"landmark_left_wrist_{axis}"]
        available_cols = [col for col in default_cols if col in data.columns]

        if not available_cols:
            # Si no hay landmarks de muñecas, buscar cualquier landmark disponible
            all_landmark_cols = [
                col
                for col in data.columns
                if col.startswith("landmark_") and col.endswith(f"_{axis}")
            ]
            if all_landmark_cols:
                landmark_columns = [all_landmark_cols[0]]  # Tomar el primero disponible
                logger.warning(f"Usando {landmark_columns[0]} como alternativa")
            else:
                raise ValueError(
                    f"No se encontraron columnas de landmarks para el eje {axis}"
                )
        else:
            landmark_columns = available_cols
            logger.debug(f"Usando columnas por defecto: {landmark_columns}")

    # Validar calidad de datos para los landmarks seleccionados
    nan_ratio = data[landmark_columns].isna().mean().mean()
    if nan_ratio > 0.2:  # Más de 20% NaNs
        logger.warning(
            f"Alto porcentaje de valores NaN ({nan_ratio:.1%}) en landmarks. "
            "La detección de repeticiones puede ser imprecisa."
        )

    # Extraer la posición del landmark (invertida para tener valor alto = posición alta)
    # Nota: en MediaPipe, las coordenadas Y crecen hacia abajo, por eso se invierte
    landmark_position = -(data[landmark_columns].min(axis=1).values)

    # Suavizar la señal
    if len(landmark_position) < smoothing_window:
        # Reducir ventana si la señal es más corta
        actual_window = min(
            (
                len(landmark_position) - 2
                if len(landmark_position) % 2 == 0
                else len(landmark_position) - 1
            ),
            5,
        )
        if actual_window < 3:
            # No suavizar si la señal es muy corta
            smoothed_position = landmark_position
            logger.warning("Señal demasiado corta para suavizar")
        else:
            smoothed_position = savgol_filter(
                landmark_position, actual_window, min(polyorder, actual_window - 1)
            )
    else:
        smoothed_position = savgol_filter(
            landmark_position, smoothing_window, polyorder
        )

    # Añadir valores artificiales al inicio y al final para facilitar detección
    padding_value = smoothed_position.mean()
    padded_signal = np.concatenate(
        ([padding_value - 10], smoothed_position, [padding_value - 10])
    )

    # Encontrar picos (puntos de inicio/fin de repetición)
    peaks, _ = find_peaks(
        padded_signal,
        prominence=prominence,
        distance=negative_distance,
        height=peak_height_threshold,
    )

    # Encontrar valles (puntos medios de repetición)
    valleys, _ = find_peaks(
        -padded_signal,
        prominence=prominence,
        distance=positive_distance,
        height=peak_height_threshold,
    )

    # Ajustar los índices al rango original
    peaks = peaks - 1
    valleys = valleys - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(smoothed_position))]
    valleys = valleys[(valleys >= 0) & (valleys < len(smoothed_position))]

    # Log de resultados iniciales
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
            # Para ejercicios de tracción, el pico es el punto medio
            # Inversión de lógica: buscamos picos adicionales entre valles
            # Por simplicidad y consistencia, seguimos usando un único valle como punto medio
            valley_candidates = valleys[(valleys > start_frame) & (valleys < end_frame)]

        if len(valley_candidates) > 0:
            mid_frame = valley_candidates[0]
        else:
            # Si no hay valle detectado, usar el punto con valor mínimo/máximo entre los picos
            segment = smoothed_position[start_frame:end_frame]
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

    # Visualizar los resultados si se solicita
    if plot_graph:
        visualize_repetitions(
            smoothed_position=smoothed_position,
            repetitions=repetitions,
            exercise_name=exercise_name,
            exercise_type=exercise_type,
            output_dir=output_dir,
            landmark_name=landmark_columns[0] if landmark_columns else "unknown",
        )

    return repetitions


def visualize_repetitions(
    smoothed_position: np.ndarray,
    repetitions: List[Dict[str, int]],
    exercise_name: Optional[str] = None,
    exercise_type: str = "press",
    output_dir: Optional[str] = None,
    landmark_name: str = "unknown",
):
    """
    Visualiza las repeticiones detectadas.

    Args:
        smoothed_position: Señal suavizada
        repetitions: Lista de repeticiones detectadas
        exercise_name: Nombre del ejercicio para el título
        exercise_type: Tipo de ejercicio ('press' o 'pull')
        output_dir: Directorio donde guardar la imagen. Si es None, no guarda.
        landmark_name: Nombre del landmark para etiqueta
    """
    try:
        plt.figure(figsize=(12, 6))

        # Graficar la señal
        plt.plot(smoothed_position, linewidth=2, label=f"Posición {landmark_name}")

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
            y_pos = np.max(smoothed_position) * 0.9
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
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            exercise_str = exercise_name or "exercise"
            filename = f"repetitions_{exercise_str}_{len(repetitions)}reps.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            logger.info(f"Gráfica guardada en: {filepath}")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error al visualizar repeticiones: {e}")


# Función original para mantener compatibilidad
def detect_repetitions_legacy(
    data,
    prominence=0.2,
    smoothing_window=11,
    polyorder=2,
    positive_distance=20,
    negative_distance=50,
    peak_height_threshold=-0.8,
    plot_graph=True,
):
    """
    Versión original de la función para compatibilidad.
    Esta función sigue la implementación original exactamente.
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
        height=peak_height_threshold,
    )

    # Encontrar valles usando la señal extendida (invertida)
    valleys, _ = find_peaks(
        -padded_signal,
        prominence=prominence,
        distance=positive_distance,
        height=peak_height_threshold,
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

    # Graficar la señal y las líneas en los frames importantes
    if plot_graph:
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_wrist_y, label="Posición Y suavizada")

        # Agregar líneas verticales en los picos y valles
        for repetition in repetitions:
            if not np.isnan(repetition["start_frame"]):
                plt.axvline(
                    x=repetition["start_frame"],
                    color="g",
                    linestyle="--",
                    label=(
                        "Pico Inicio"
                        if repetition["start_frame"] == repetitions[0]["start_frame"]
                        else ""
                    ),
                )
            if not np.isnan(repetition["mid_frame"]):
                plt.axvline(
                    x=repetition["mid_frame"],
                    color="r",
                    linestyle="--",
                    label=(
                        "Valle"
                        if repetition["mid_frame"] == repetitions[0]["mid_frame"]
                        else ""
                    ),
                )
            if not np.isnan(repetition["end_frame"]):
                plt.axvline(
                    x=repetition["end_frame"],
                    color="b",
                    linestyle="--",
                    label=(
                        "Pico Fin"
                        if repetition["end_frame"] == repetitions[0]["end_frame"]
                        else ""
                    ),
                )

        plt.title("Detección de Repeticiones")
        plt.xlabel("Frame")
        plt.ylabel("Posición Y suavizada")
        plt.legend(loc="upper right")
        plt.show()

    return repetitions
