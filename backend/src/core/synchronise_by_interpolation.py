import numpy as np
import pandas as pd
import sys
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Importamos funciones de configuración
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import load_exercise_config

# Configuración del logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Importamos la función de detección de repeticiones del módulo original
from src.core.detect_repetitions import detect_repetitions

###############################################################################
# 1. FUNCIONES DE VALIDACIÓN Y PREPROCESAMIENTO
###############################################################################


def _validate_input_data(
    user_data: pd.DataFrame, expert_data: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """
    Valida que los DataFrames de entrada tengan la estructura correcta y contengan los landmarks necesarios.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        config: Diccionario de configuración

    Raises:
        ValueError: Si hay problemas con los datos de entrada
    """
    # Verificar que los DataFrames no estén vacíos
    if user_data.empty or expert_data.empty:
        raise ValueError("Los DataFrames de usuario o experto están vacíos")

    # Verificar que los landmarks requeridos estén presentes
    required_landmarks = config.get(
        "landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
    )

    for landmark in required_landmarks:
        for suffix in ["_x", "_y", "_z"]:
            column = f"{landmark}{suffix}"
            if column not in user_data.columns:
                raise ValueError(
                    f"Columna {column} no encontrada en los datos del usuario"
                )
            if column not in expert_data.columns:
                raise ValueError(
                    f"Columna {column} no encontrada en los datos del experto"
                )

    # Verificar que no haya demasiados NaN en columnas críticas
    for landmark in required_landmarks:
        for suffix in ["_x", "_y", "_z"]:
            column = f"{landmark}{suffix}"
            user_nans = user_data[column].isna().mean()
            expert_nans = expert_data[column].isna().mean()

            if user_nans > 0.3:  # Más del 30% de valores NaN
                logger.warning(
                    f"La columna {column} en datos de usuario tiene {user_nans:.1%} valores NaN"
                )
            if expert_nans > 0.3:
                logger.warning(
                    f"La columna {column} en datos de experto tiene {expert_nans:.1%} valores NaN"
                )


def _preprocess_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesa el DataFrame aplicando filtros de suavizado y rellenando NaNs.

    Args:
        df: DataFrame a preprocesar
        config: Configuración con parámetros de preprocesamiento

    Returns:
        DataFrame preprocesado
    """
    # Copiar para no modificar el original
    result = df.copy()

    # Parámetros de preprocesamiento
    window_size = config.get("smoothing_window", 7)
    poly_order = config.get("poly_order", 2)

    # Asegurarse que window_size es impar
    if window_size % 2 == 0:
        window_size += 1

    # Aplicar suavizado Savitzky-Golay a columnas específicas
    landmark_columns = [
        col
        for col in df.columns
        if col.startswith("landmark_") and col.endswith(("_x", "_y", "_z"))
    ]

    for col in landmark_columns:
        # Rellenar NaNs con interpolación antes de suavizar
        result[col] = result[col].interpolate(method="linear", limit_direction="both")

        # Aplicar suavizado solo si hay suficientes puntos
        if len(result) > window_size:
            result[col] = savgol_filter(result[col], window_size, poly_order)

    return result


###############################################################################
# 2. DETECCIÓN Y EMPAREJAMIENTO DE REPETICIONES
###############################################################################


def _get_matched_repetitions(
    user_data: pd.DataFrame, expert_data: pd.DataFrame, config: Dict[str, Any]
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Detecta las repeticiones en los datos del usuario y experto,
    y los empareja de forma inteligente.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        config: Configuración para la detección de repeticiones

    Returns:
        Lista de tuplas (repetición_usuario, repetición_experto) emparejadas
    """
    # Parámetros para la detección de repeticiones
    prominence = config.get("rep_prominence", 0.2)
    smoothing_window = config.get("rep_smoothing_window", 11)
    polyorder = config.get("rep_polyorder", 2)
    positive_distance = config.get("rep_positive_distance", 20)
    negative_distance = config.get("rep_negative_distance", 50)
    peak_height_threshold = config.get("rep_peak_height_threshold", -0.8)

    # Detectar repeticiones
    logger.info("Detectando repeticiones en datos de usuario...")
    user_repetitions = detect_repetitions(
        user_data,
        prominence=prominence,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        positive_distance=positive_distance,
        negative_distance=negative_distance,
        peak_height_threshold=peak_height_threshold,
        plot_graph=False,
    )

    logger.info("Detectando repeticiones en datos de experto...")
    expert_repetitions = detect_repetitions(
        expert_data,
        prominence=prominence,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        positive_distance=positive_distance,
        negative_distance=negative_distance,
        peak_height_threshold=peak_height_threshold,
        plot_graph=False,
    )

    if not user_repetitions:
        raise ValueError("No se detectaron repeticiones en los datos del usuario")

    if not expert_repetitions:
        raise ValueError("No se detectaron repeticiones en los datos del experto")

    logger.info(
        f"Detectadas {len(user_repetitions)} repeticiones de usuario y {len(expert_repetitions)} de experto"
    )

    # Estrategia de emparejamiento basada en la configuración
    matching_strategy = config.get("matching_strategy", "first_only")

    if matching_strategy == "best_example":
        # Usar la mejor repetición del experto para todas las del usuario
        best_expert_rep = _find_best_expert_repetition(expert_repetitions, expert_data)
        return [(user_rep, best_expert_rep) for user_rep in user_repetitions]

    elif matching_strategy == "one_to_one":
        # Emparejar repeticiones 1:1 hasta donde alcancen
        max_pairs = min(len(user_repetitions), len(expert_repetitions))
        return [(user_repetitions[i], expert_repetitions[i]) for i in range(max_pairs)]

    elif matching_strategy == "similarity":
        # Emparejar basado en similitud de duración/forma
        return _match_repetitions_by_similarity(
            user_repetitions, expert_repetitions, user_data, expert_data
        )

    else:  # 'first_only', comportamiento original
        # Usar la primera repetición del experto para todas
        if not expert_repetitions:
            raise ValueError("No se detectaron repeticiones en los datos del experto")

        default_expert_rep = expert_repetitions[0]
        logger.info(
            f"Usando primera repetición del experto para todas ({len(user_repetitions)}) las del usuario"
        )
        return [(user_rep, default_expert_rep) for user_rep in user_repetitions]


def _find_best_expert_repetition(
    expert_repetitions: List[Dict[str, int]], expert_data: pd.DataFrame
) -> Dict[str, int]:
    """
    Encuentra la "mejor" repetición del experto basándose en criterios como
    duración, limpieza de movimiento, etc.

    Args:
        expert_repetitions: Lista de repeticiones detectadas del experto
        expert_data: DataFrame con datos del experto

    Returns:
        La repetición del experto considerada como mejor ejemplo
    """
    # Si solo hay una repetición, devolver esa
    if len(expert_repetitions) == 1:
        return expert_repetitions[0]

    # Criterios para evaluar la calidad de una repetición
    rep_scores = []

    for rep in expert_repetitions:
        start_frame = rep["start_frame"]
        mid_frame = rep.get("mid_frame")
        end_frame = rep["end_frame"]

        # Si no hay mid_frame, asignar puntuación baja
        if np.isnan(mid_frame):
            rep_scores.append(-1)
            continue

        # Calcular la duración
        duration = end_frame - start_frame

        # Verificar la suavidad del movimiento usando la varianza de las derivadas
        rep_data = expert_data.iloc[int(start_frame) : int(end_frame)]
        wrist_cols = [
            col for col in rep_data.columns if "wrist" in col and col.endswith("_y")
        ]

        if not wrist_cols:
            rep_scores.append(0)
            continue

        smoothness = 0
        for col in wrist_cols:
            # Calcular derivada primera
            deriv = np.diff(rep_data[col].values)
            # Calcular varianza (menos varianza = más suave)
            smoothness -= np.var(deriv)

        # Combinar criterios: preferimos repeticiones de duración media y movimiento suave
        # La duración ideal está alrededor de 60-100 frames
        duration_score = (
            -abs(duration - 80) / 20
        )  # Penalizar lejanía de la duración ideal

        # Combinación de criterios
        total_score = duration_score + smoothness
        rep_scores.append(total_score)

    # Devolver la repetición con mayor puntuación
    best_rep_idx = np.argmax(rep_scores)
    return expert_repetitions[best_rep_idx]


def _match_repetitions_by_similarity(
    user_repetitions: List[Dict[str, int]],
    expert_repetitions: List[Dict[str, int]],
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Empareja repeticiones del usuario con las del experto basándose en similitud.

    Args:
        user_repetitions: Lista de repeticiones del usuario
        expert_repetitions: Lista de repeticiones del experto
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto

    Returns:
        Lista de pares (repetición_usuario, repetición_experto) emparejados
    """
    # Matriz de similitud entre cada par de repeticiones
    similarity_matrix = np.zeros((len(user_repetitions), len(expert_repetitions)))

    for i, user_rep in enumerate(user_repetitions):
        u_start = int(user_rep["start_frame"])
        u_end = int(user_rep["end_frame"])
        u_duration = u_end - u_start

        for j, expert_rep in enumerate(expert_repetitions):
            e_start = int(expert_rep["start_frame"])
            e_end = int(expert_rep["end_frame"])
            e_duration = e_end - e_start

            # Diferencia relativa en duración
            duration_diff = abs(u_duration - e_duration) / max(u_duration, e_duration)

            # Extraer trayectoria de las muñecas (normalizada en tiempo)
            u_wrist_y = user_data.iloc[u_start:u_end]["landmark_right_wrist_y"].values
            u_wrist_y = u_wrist_y - u_wrist_y.min()  # Normalizar a 0
            u_wrist_y = (
                u_wrist_y / u_wrist_y.max() if u_wrist_y.max() > 0 else u_wrist_y
            )  # Normalizar a 1

            e_wrist_y = expert_data.iloc[e_start:e_end]["landmark_right_wrist_y"].values
            e_wrist_y = e_wrist_y - e_wrist_y.min()  # Normalizar a 0
            e_wrist_y = (
                e_wrist_y / e_wrist_y.max() if e_wrist_y.max() > 0 else e_wrist_y
            )  # Normalizar a 1

            # Remuestrear para comparar directamente
            u_time = np.linspace(0, 1, len(u_wrist_y))
            e_time = np.linspace(0, 1, len(e_wrist_y))

            common_length = 50  # Longitud común para comparación
            common_time = np.linspace(0, 1, common_length)

            u_interp = interp1d(
                u_time, u_wrist_y, bounds_error=False, fill_value="extrapolate"
            )(common_time)
            e_interp = interp1d(
                e_time, e_wrist_y, bounds_error=False, fill_value="extrapolate"
            )(common_time)

            # Calcular distancia media entre trayectorias
            traj_distance = np.mean(np.abs(u_interp - e_interp))

            # Puntuación de similitud (menor es mejor)
            similarity_score = duration_diff + traj_distance
            similarity_matrix[i, j] = similarity_score

    # Asignar repeticiones greedily
    matched_pairs = []
    unmatched_user_reps = list(range(len(user_repetitions)))

    while unmatched_user_reps:
        best_score = float("inf")
        best_pair = None

        for i in unmatched_user_reps:
            best_exp_j = np.argmin(similarity_matrix[i, :])
            score = similarity_matrix[i, best_exp_j]

            if score < best_score:
                best_score = score
                best_pair = (i, best_exp_j)

        if best_pair:
            i, j = best_pair
            matched_pairs.append((user_repetitions[i], expert_repetitions[j]))
            unmatched_user_reps.remove(i)

            # Penalizar esta columna para evitar reusar el mismo experto
            similarity_matrix[:, j] += 100
        else:
            break

    return matched_pairs


###############################################################################
# 3. IDENTIFICACIÓN Y DIVISIÓN DE FASES
###############################################################################


def _identify_phases(
    data: pd.DataFrame, repetition: Dict[str, int], config: Dict[str, Any]
) -> List[Tuple[int, int]]:
    """
    Identifica las fases del movimiento dentro de una repetición.
    Por defecto divide en fase de subida y bajada.

    Args:
        data: DataFrame con datos del movimiento
        repetition: Diccionario con info de la repetición
        config: Configuración para la identificación de fases

    Returns:
        Lista de tuplas (frame_inicio, frame_fin) para cada fase
    """
    start_frame = repetition.get("start_frame", 0)
    mid_frame = repetition.get("mid_frame", None)
    end_frame = repetition.get("end_frame", 0)

    # Si tenemos el frame medio, dividir en subida y bajada
    if mid_frame is not None and not np.isnan(mid_frame):
        # Fase de subida: inicio → medio
        # Fase de bajada: medio → fin
        return [(int(start_frame), int(mid_frame)), (int(mid_frame), int(end_frame))]
    else:
        # Si no tenemos mid_frame, analizar la curva para dividir
        phase_strategy = config.get("phase_strategy", "auto")

        if phase_strategy == "auto":
            # Fase automática mediante análisis de la curva
            segment = data.iloc[int(start_frame) : int(end_frame)].copy()

            # Usar la combinación de ambas muñecas para robustez
            wrist_y = (
                -segment[["landmark_right_wrist_y", "landmark_left_wrist_y"]]
                .min(axis=1)
                .values
            )

            # Suavizar para encontrar picos/valles más robustos
            wrist_y_smooth = savgol_filter(
                wrist_y,
                min(11, len(wrist_y) - 1 if len(wrist_y) % 2 == 0 else len(wrist_y)),
                2,
            )

            # Encontrar el punto más bajo/más alto para dividir en fases
            if config.get("exercise_type", "press") == "press":  # Ejercicio de empuje
                lowest_point_idx = np.argmin(wrist_y_smooth)
                mid_frame_estimated = start_frame + lowest_point_idx
            else:  # Ejercicio de tracción
                highest_point_idx = np.argmax(wrist_y_smooth)
                mid_frame_estimated = start_frame + highest_point_idx

            # Comprobar que el punto medio está dentro de la ventana
            if mid_frame_estimated <= start_frame or mid_frame_estimated >= end_frame:
                # Fallback: dividir por la mitad
                mid_frame_estimated = start_frame + (end_frame - start_frame) // 2
                logger.warning(
                    f"No se pudo identificar la fase automáticamente. Dividiendo por la mitad."
                )

            return [
                (int(start_frame), int(mid_frame_estimated)),
                (int(mid_frame_estimated), int(end_frame)),
            ]

        elif phase_strategy == "single":
            # Tratar toda la repetición como una sola fase
            return [(int(start_frame), int(end_frame))]

        elif phase_strategy == "thirds":
            # Dividir en tres partes iguales
            third_length = (end_frame - start_frame) / 3
            third1 = start_frame + third_length
            third2 = start_frame + 2 * third_length
            return [
                (int(start_frame), int(third1)),
                (int(third1), int(third2)),
                (int(third2), int(end_frame)),
            ]

        else:
            # Por defecto, dividir en dos partes iguales
            midpoint = start_frame + (end_frame - start_frame) / 2
            return [(int(start_frame), int(midpoint)), (int(midpoint), int(end_frame))]


def _divide_segment_by_height(
    data: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    num_divisions: int,
    landmarks: List[str] = None,
    axis: str = "y",
) -> List[int]:
    """
    Divide un segmento en subsegmentos basados en la altura de landmarks específicos.

    Args:
        data: DataFrame con datos
        start_frame: Frame de inicio del segmento
        end_frame: Frame de fin del segmento
        num_divisions: Número de divisiones a crear
        landmarks: Lista de landmarks a utilizar (por defecto, muñecas)
        axis: Eje a considerar ('y' por defecto)

    Returns:
        Lista de frames que dividen el segmento
    """
    if start_frame >= end_frame or start_frame < 0 or end_frame > len(data):
        raise ValueError(
            f"Rango inválido: start_frame={start_frame}, end_frame={end_frame}"
        )

    # Segmento de datos entre start_frame y end_frame
    segment = data.iloc[start_frame:end_frame].reset_index(drop=True)

    # Por defecto, usar las muñecas
    if landmarks is None:
        landmarks = ["landmark_right_wrist", "landmark_left_wrist"]

    # Recopilar columnas relevantes
    height_columns = []
    for landmark in landmarks:
        col = f"{landmark}_{axis}"
        if col in segment.columns:
            height_columns.append(col)

    if not height_columns:
        raise ValueError(
            f"No se encontraron columnas de altura para los landmarks especificados"
        )

    # Para ejercicios de press, invertir el eje Y
    heights = -segment[height_columns].min(axis=1)

    if heights.empty:
        raise ValueError(f"Alturas vacías en segmento de {start_frame} a {end_frame}.")

    # Calcular alturas de división
    division_heights = np.linspace(heights.iloc[0], heights.iloc[-1], num_divisions + 1)

    # Identificar los frames de las divisiones
    division_frames_relative = [
        (heights - target_height).abs().idxmin()
        for target_height in division_heights[1:-1]
    ]

    division_frames = (
        [start_frame]
        + [start_frame + rel for rel in division_frames_relative]
        + [end_frame]
    )

    # Asegurar que los frames están ordenados y son únicos
    division_frames = sorted(list(set(division_frames)))

    return division_frames


def _divide_segment_adaptative(
    data: pd.DataFrame, start_frame: int, end_frame: int, config: Dict[str, Any]
) -> List[int]:
    """
    Divide un segmento en subsegmentos utilizando la estrategia especificada.

    Args:
        data: DataFrame con datos
        start_frame: Frame de inicio del segmento
        end_frame: Frame de fin del segmento
        config: Diccionario de configuración con parámetros de división

    Returns:
        Lista de frames que dividen el segmento
    """
    division_strategy = config.get("division_strategy", "height")
    num_divisions = config.get("num_divisions", 7)

    if division_strategy == "height":
        # División basada en altura
        landmarks = config.get(
            "division_landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
        )
        axis = config.get("division_axis", "y")
        return _divide_segment_by_height(
            data, start_frame, end_frame, num_divisions, landmarks, axis
        )

    elif division_strategy == "equal":
        # División en partes iguales
        step = (end_frame - start_frame) / num_divisions
        return [int(start_frame + i * step) for i in range(num_divisions + 1)]

    elif division_strategy == "acceleration":
        # División basada en cambios de aceleración
        segment = data.iloc[start_frame:end_frame].copy()
        landmarks = config.get(
            "division_landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
        )
        axis = config.get("division_axis", "y")

        # Calcular derivada segunda (aceleración)
        accel_columns = []
        for landmark in landmarks:
            col = f"{landmark}_{axis}"
            if col in segment.columns:
                # Calcular aceleración
                vel = np.gradient(segment[col].values)
                accel = np.gradient(vel)
                segment[f"{col}_accel"] = accel
                accel_columns.append(f"{col}_accel")

        if not accel_columns:
            # Fallback a división igual
            return [
                int(start_frame + i * (end_frame - start_frame) / num_divisions)
                for i in range(num_divisions + 1)
            ]

        # Usar la suma de aceleraciones
        accel_sum = segment[accel_columns].abs().sum(axis=1)

        # Encontrar puntos de mayor cambio de aceleración
        sorted_indices = np.argsort(accel_sum.values)[
            ::-1
        ]  # Ordenar por magnitud de cambio

        # Tomar los N-1 puntos más significativos
        significant_points = sorted_indices[: num_divisions - 1]
        significant_points = np.sort(significant_points)  # Reordenar por tiempo

        # Convertir a frames absolutos
        division_frames = (
            [start_frame]
            + [start_frame + idx for idx in significant_points]
            + [end_frame]
        )

        return division_frames

    else:
        # Por defecto, división equitativa
        step = (end_frame - start_frame) / num_divisions
        return [int(start_frame + i * step) for i in range(num_divisions + 1)]


###############################################################################
# 4. INTERPOLACIÓN Y SINCRONIZACIÓN
###############################################################################


def _interpolate_segment(
    original_data: pd.DataFrame, target_length: int, method: str = "linear"
) -> pd.DataFrame:
    """
    Interpola un segmento de datos para ajustarlo a una longitud objetivo.

    Args:
        original_data: DataFrame con datos originales
        target_length: Longitud objetivo
        method: Método de interpolación ('linear', 'cubic', etc.)

    Returns:
        DataFrame con datos interpolados
    """
    if original_data.empty:
        raise ValueError("Segmento vacío recibido en interpolate_segment.")

    if target_length <= 1:
        raise ValueError(f"Longitud objetivo inválida: {target_length}")

    # Si ya tiene la longitud correcta, devolver sin cambios
    if len(original_data) == target_length:
        return original_data.copy()

    original_frames = np.linspace(0, 1, len(original_data))
    target_frames = np.linspace(0, 1, target_length)

    interpolated_data = pd.DataFrame()

    # Agrupar columnas similares para optimizar (opcional)
    for col in original_data.select_dtypes(include=[np.number]).columns:
        # Saltear si todos son NaN
        if original_data[col].isna().all():
            interpolated_data[col] = [np.nan] * target_length
            continue

        # Valores no NaN para interpolación
        valid_idx = ~original_data[col].isna()
        valid_frames = original_frames[valid_idx]
        valid_values = original_data[col].dropna().values

        # En caso de que no haya valores suficientes
        if len(valid_values) <= 1:
            interpolated_data[col] = [
                valid_values[0] if len(valid_values) > 0 else np.nan
            ] * target_length
            continue

        # Crear el interpolador con los valores válidos
        try:
            interpolator = interp1d(
                valid_frames,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated_data[col] = interpolator(target_frames)
        except Exception as e:
            logger.warning(f"Error al interpolar columna {col}: {e}")
            # Fallback a método más simple
            interpolated_data[col] = np.interp(
                target_frames,
                valid_frames,
                valid_values,
                left=valid_values[0],
                right=valid_values[-1],
            )

    return interpolated_data


def _synchronize_subsegments(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    user_frames: List[int],
    expert_frames: List[int],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sincroniza subsegmentos entre usuario y experto mediante interpolación.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        user_frames: Lista de frames del usuario que delimitan subsegmentos
        expert_frames: Lista de frames del experto que delimitan subsegmentos
        config: Configuración para la sincronización

    Returns:
        Tupla (user_processed, expert_processed) con los segmentos sincronizados
    """
    user_segments = []
    expert_segments = []

    # Método de interpolación
    interp_method = config.get("interp_method", "linear")

    # Procesar cada par de subsegmentos
    for i in range(len(user_frames) - 1):
        user_sub_segment = user_data.iloc[user_frames[i] : user_frames[i + 1]]
        expert_sub_segment = expert_data.iloc[expert_frames[i] : expert_frames[i + 1]]

        if user_sub_segment.empty or expert_sub_segment.empty:
            logger.warning(
                f"Segmento vacío encontrado: usuario={user_sub_segment.empty}, "
                f"experto={expert_sub_segment.empty}"
            )
            continue

        # Decidir si interpolar experto al usuario o viceversa
        # Por defecto adaptamos el experto a la longitud del usuario
        adapt_direction = config.get("adapt_direction", "expert_to_user")

        if adapt_direction == "expert_to_user":
            # Interpolar expert_sub_segment para que coincida con la longitud de user_sub_segment
            interpolated_sub_segment = _interpolate_segment(
                expert_sub_segment, len(user_sub_segment), interp_method
            )
            user_segments.append(user_sub_segment)
            expert_segments.append(interpolated_sub_segment)

        elif adapt_direction == "user_to_expert":
            # Interpolar user_sub_segment para que coincida con la longitud de expert_sub_segment
            interpolated_sub_segment = _interpolate_segment(
                user_sub_segment, len(expert_sub_segment), interp_method
            )
            user_segments.append(interpolated_sub_segment)
            expert_segments.append(expert_sub_segment)

        elif adapt_direction == "both_to_average":
            # Interpolar ambos a la longitud promedio
            avg_length = (len(user_sub_segment) + len(expert_sub_segment)) // 2
            interpolated_user = _interpolate_segment(
                user_sub_segment, avg_length, interp_method
            )
            interpolated_expert = _interpolate_segment(
                expert_sub_segment, avg_length, interp_method
            )
            user_segments.append(interpolated_user)
            expert_segments.append(interpolated_expert)

        else:
            # Default: adaptar experto al usuario
            interpolated_sub_segment = _interpolate_segment(
                expert_sub_segment, len(user_sub_segment), interp_method
            )
            user_segments.append(user_sub_segment)
            expert_segments.append(interpolated_sub_segment)

    # Concatenar los segmentos
    if not user_segments or not expert_segments:
        raise ValueError("No se generaron segmentos válidos durante la sincronización")

    user_processed = pd.concat(user_segments).reset_index(drop=True)
    expert_processed = pd.concat(expert_segments).reset_index(drop=True)

    return user_processed, expert_processed


def _process_repetition_pair(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    user_rep: Dict[str, int],
    expert_rep: Dict[str, int],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Procesa un par de repeticiones (usuario-experto) dividiéndolas en fases
    y sincronizando los subsegmentos.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        user_rep: Diccionario con info de repetición del usuario
        expert_rep: Diccionario con info de repetición del experto
        config: Configuración para el procesamiento

    Returns:
        Tupla (user_processed, expert_processed) con las repeticiones sincronizadas
    """
    # Identificar fases para usuario y experto
    user_phases = _identify_phases(user_data, user_rep, config)
    expert_phases = _identify_phases(expert_data, expert_rep, config)

    # Si hay diferente número de fases, usar el mínimo
    min_phases = min(len(user_phases), len(expert_phases))
    user_phases = user_phases[:min_phases]
    expert_phases = expert_phases[:min_phases]

    # Procesar cada fase por separado
    user_processed_segments = []
    expert_processed_segments = []

    for (user_phase_start, user_phase_end), (
        expert_phase_start,
        expert_phase_end,
    ) in zip(user_phases, expert_phases):
        # Dividir cada fase en subsegmentos
        user_frames = _divide_segment_adaptative(
            user_data, user_phase_start, user_phase_end, config
        )

        expert_frames = _divide_segment_adaptative(
            expert_data, expert_phase_start, expert_phase_end, config
        )

        # Asegurar que ambos tienen el mismo número de divisiones
        min_divisions = min(len(user_frames), len(expert_frames)) - 1
        user_frames = user_frames[: min_divisions + 1]
        expert_frames = expert_frames[: min_divisions + 1]

        # Sincronizar los subsegmentos
        user_phase_proc, expert_phase_proc = _synchronize_subsegments(
            user_data, expert_data, user_frames, expert_frames, config
        )

        user_processed_segments.append(user_phase_proc)
        expert_processed_segments.append(expert_phase_proc)

    # Concatenar todas las fases procesadas
    user_processed = pd.concat(user_processed_segments).reset_index(drop=True)
    expert_processed = pd.concat(expert_processed_segments).reset_index(drop=True)

    return user_processed, expert_processed


def _process_all_repetitions(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    repetition_pairs: List[Tuple[Dict[str, int], Dict[str, int]]],
    config: Dict[str, Any],
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Procesa todas las repeticiones emparejadas en paralelo para mejorar rendimiento.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        repetition_pairs: Lista de pares (repetición_usuario, repetición_experto)
        config: Configuración para el procesamiento

    Returns:
        Tupla (user_processed_segments, expert_processed_segments) con listas de segmentos procesados
    """
    user_processed_segments = []
    expert_processed_segments = []

    # Verificar si debemos usar paralelismo
    use_parallel = config.get("use_parallel", False) and len(repetition_pairs) > 1

    if use_parallel:
        # Procesamiento en paralelo para múltiples repeticiones
        with ProcessPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
            futures = []

            for user_rep, expert_rep in repetition_pairs:
                future = executor.submit(
                    _process_repetition_pair,
                    user_data.copy(),  # Copiar para evitar problemas con paralelismo
                    expert_data.copy(),
                    user_rep,
                    expert_rep,
                    config,
                )
                futures.append(future)

            # Recopilar resultados
            for future in as_completed(futures):
                try:
                    user_proc, expert_proc = future.result()
                    user_processed_segments.append(user_proc)
                    expert_processed_segments.append(expert_proc)
                except Exception as e:
                    logger.error(f"Error al procesar repetición en paralelo: {e}")

    else:
        # Procesamiento secuencial
        for user_rep, expert_rep in repetition_pairs:
            try:
                user_proc, expert_proc = _process_repetition_pair(
                    user_data, expert_data, user_rep, expert_rep, config
                )
                user_processed_segments.append(user_proc)
                expert_processed_segments.append(expert_proc)
            except Exception as e:
                logger.error(f"Error al procesar repetición: {e}")

    return user_processed_segments, expert_processed_segments


def _combine_and_validate(
    user_segments: List[pd.DataFrame], expert_segments: List[pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combina los segmentos procesados y verifica que los resultados sean válidos.

    Args:
        user_segments: Lista de DataFrames con segmentos de usuario procesados
        expert_segments: Lista de DataFrames con segmentos de experto procesados

    Returns:
        Tupla (final_user_data, final_expert_data) con los datos sincronizados completos
    """
    if not user_segments or not expert_segments:
        raise ValueError("No hay segmentos para combinar")

    # Verificar que para cada segmento de usuario hay uno de experto
    if len(user_segments) != len(expert_segments):
        raise ValueError(
            f"Número diferente de segmentos: usuario={len(user_segments)}, "
            f"experto={len(expert_segments)}"
        )

    # Verificar que los segmentos correspondientes tienen el mismo tamaño
    for i, (user_seg, expert_seg) in enumerate(zip(user_segments, expert_segments)):
        if len(user_seg) != len(expert_seg):
            raise ValueError(
                f"El segmento {i} tiene diferentes longitudes: "
                f"usuario={len(user_seg)}, experto={len(expert_seg)}"
            )

    # Combinar todos los segmentos
    final_user_data = pd.concat(user_segments, axis=0).reset_index(drop=True)
    final_expert_data = pd.concat(expert_segments, axis=0).reset_index(drop=True)

    # Verificar que ambos tienen el mismo número total de frames
    if len(final_user_data) != len(final_expert_data):
        raise ValueError(
            f"Desajuste en el número de frames: usuario={len(final_user_data)}, "
            f"experto={len(final_expert_data)}"
        )

    # Renumerar los frames si existe la columna
    if "frame" in final_user_data.columns and "frame" in final_expert_data.columns:
        new_frames = np.arange(len(final_user_data))
        final_user_data["frame"] = new_frames
        final_expert_data["frame"] = new_frames

    logger.info(
        f"Sincronización completada: {len(final_user_data)} frames sincronizados"
    )

    return final_user_data, final_expert_data


###############################################################################
# 5. FUNCIÓN PRINCIPAL DE SINCRONIZACIÓN
###############################################################################


def synchronize_data(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    exercise_name: Optional[str] = None,
    config_path: str = "config_expanded.json",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sincroniza los datos del experto con los del usuario utilizando una estrategia
    flexible y configurable.

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto
        config: Diccionario con configuración personalizada (opcional)
        exercise_name: Nombre del ejercicio para cargar configuración (opcional)
        config_path: Ruta al archivo de configuración (opcional)

    Returns:
        Tupla (user_data_sync, expert_data_sync) con los datos sincronizados
    """
    logger.info("Iniciando proceso de sincronización...")

    # Cargar configuración desde archivo si se proporciona un nombre de ejercicio
    if exercise_name and not config:
        try:
            logger.info(f"Cargando configuración para ejercicio: {exercise_name}")
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada: {config}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración del ejercicio: {e}")
            logger.warning("Usando configuración por defecto")
            config = None

    # Configuración por defecto
    if config is None:
        config = {
            "landmarks": ["landmark_right_wrist", "landmark_left_wrist"],
            "num_divisions": 7,
            "interp_method": "linear",
            "division_strategy": "height",
            "matching_strategy": "first_only",
            "adapt_direction": "expert_to_user",
            "exercise_type": "press",
            "use_parallel": False,
            "max_workers": 4,
            "phase_strategy": "auto",
            "rep_prominence": 0.2,
            "rep_smoothing_window": 11,
            "rep_polyorder": 2,
            "rep_positive_distance": 20,
            "rep_negative_distance": 50,
            "rep_peak_height_threshold": -0.8,
            "smoothing_window": 7,
            "poly_order": 2,
        }

    # 1. Validar datos de entrada
    _validate_input_data(user_data, expert_data, config)

    # 2. Preprocesar datos (opcional)
    if config.get("preprocess", True):
        logger.info("Preprocesando datos...")
        user_data_prep = _preprocess_dataframe(user_data, config)
        expert_data_prep = _preprocess_dataframe(expert_data, config)
    else:
        user_data_prep = user_data.copy()
        expert_data_prep = expert_data.copy()

    # 3. Detectar y emparejar repeticiones
    logger.info("Detectando y emparejando repeticiones...")
    repetition_pairs = _get_matched_repetitions(
        user_data_prep, expert_data_prep, config
    )

    # 4. Procesar todas las repeticiones
    logger.info(f"Procesando {len(repetition_pairs)} pares de repeticiones...")
    user_segments, expert_segments = _process_all_repetitions(
        user_data_prep, expert_data_prep, repetition_pairs, config
    )

    # 5. Combinar y validar
    logger.info("Combinando resultados...")
    final_user_data, final_expert_data = _combine_and_validate(
        user_segments, expert_segments
    )

    logger.info("Sincronización completada exitosamente.")
    return final_user_data, final_expert_data


# Función legacy para mantener compatibilidad con el código existente
def synchronize_data_by_height(user_data, expert_data, num_divisions=7):
    """
    Función de compatibilidad con el código anterior.
    Llama a la nueva implementación con la configuración adecuada.
    """
    config = {
        "landmarks": ["landmark_right_wrist", "landmark_left_wrist"],
        "num_divisions": num_divisions,
        "interp_method": "linear",
        "division_strategy": "height",
        "matching_strategy": "first_only",
        "adapt_direction": "expert_to_user",
        "exercise_type": "press",
        "use_parallel": False,
    }

    return synchronize_data(user_data, expert_data, config)
