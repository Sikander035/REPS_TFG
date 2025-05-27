"""
Utilities for synchronizing data between user and expert exercise recordings.

This module provides utility functions for temporal alignment and synchronization
of exercise data captured from users and experts.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging

# Configuración de logging
logger = logging.getLogger(__name__)

#############################################################################
# CORE VALIDATION AND PREPROCESSING
#############################################################################


def validate_input_data(user_data, expert_data, config=None):
    """
    Validates that input DataFrames have the correct structure and required landmarks.
    """
    config = config or {}

    # Verificaciones básicas
    if user_data.empty or expert_data.empty:
        raise ValueError("User or expert DataFrames are empty")

    # Obtener landmarks requeridos
    required_landmarks = config.get(
        "landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
    )

    # Verificar que los landmarks existen
    for landmark in required_landmarks:
        for suffix in ["_x", "_y", "_z"]:
            column = f"{landmark}{suffix}"
            if column not in user_data.columns:
                raise ValueError(f"Column {column} not found in user data")
            if column not in expert_data.columns:
                raise ValueError(f"Column {column} not found in expert data")

    # Verificación de NaN excesivos
    for landmark in required_landmarks:
        for suffix in ["_x", "_y", "_z"]:
            column = f"{landmark}{suffix}"
            user_nans = user_data[column].isna().mean()
            expert_nans = expert_data[column].isna().mean()

            if user_nans > 0.3:
                logger.warning(
                    f"Column {column} in user data has {user_nans:.1%} NaN values"
                )
            if expert_nans > 0.3:
                logger.warning(
                    f"Column {column} in expert data has {expert_nans:.1%} NaN values"
                )


def preprocess_dataframe(df, config=None):
    """
    Preprocesses DataFrame by applying smoothing and filling NaN values.
    """
    config = config or {}
    result = df.copy()

    # Parámetros de preprocesamiento
    window_size = config.get("smoothing_window", 7)
    poly_order = config.get("poly_order", 2)

    # Asegurar que window_size es impar
    if window_size % 2 == 0:
        window_size += 1

    # Suavizar columnas de landmarks
    landmark_columns = [
        col
        for col in df.columns
        if col.startswith("landmark_") and col.endswith(("_x", "_y", "_z"))
    ]

    for col in landmark_columns:
        # Interpolar NaN primero
        result[col] = result[col].interpolate(method="linear", limit_direction="both")

        # Aplicar suavizado si hay suficientes puntos
        if len(result) > window_size:
            result[col] = savgol_filter(result[col], window_size, poly_order)

    return result


#############################################################################
# REPETITION MATCHING
#############################################################################


def match_repetitions(
    user_repetitions, expert_repetitions, user_data, expert_data, config=None
):
    """
    Matches repetitions between user and expert based on the specified strategy.

    Args:
        user_repetitions: List of user repetitions
        expert_repetitions: List of expert repetitions
        user_data: DataFrame with user data (needed for similarity matching)
        expert_data: DataFrame with expert data (needed for similarity matching)
        config: Configuration with matching strategy

    Returns:
        List of tuples (user_repetition, expert_repetition) that are matched
    """
    config = config or {}
    matching_strategy = config.get("matching_strategy", "first_only")

    logger.info(f"Matching repetitions using strategy: {matching_strategy}")

    if matching_strategy == "best_example":
        # Usar mejor repetición del experto para todas las del usuario
        best_expert_rep = _find_best_expert_repetition(expert_repetitions, expert_data)
        return [(user_rep, best_expert_rep) for user_rep in user_repetitions]

    elif matching_strategy == "one_to_one":
        # Emparejar 1:1 hasta agotar
        max_pairs = min(len(user_repetitions), len(expert_repetitions))
        return [(user_repetitions[i], expert_repetitions[i]) for i in range(max_pairs)]

    elif matching_strategy == "similarity":
        # Emparejar por similitud
        return _match_repetitions_by_similarity(
            user_repetitions, expert_repetitions, user_data, expert_data
        )

    else:  # 'first_only' (comportamiento predeterminado)
        # Usar primera repetición del experto para todas las del usuario
        default_expert_rep = expert_repetitions[0]
        logger.info(
            f"Using first expert repetition for all ({len(user_repetitions)}) user repetitions"
        )
        return [(user_rep, default_expert_rep) for user_rep in user_repetitions]


def _find_best_expert_repetition(expert_repetitions, expert_data):
    """Helper function to find the best expert repetition."""
    # Si solo hay una repetición, devolverla
    if len(expert_repetitions) == 1:
        return expert_repetitions[0]

    # Criterios para evaluar calidad de repetición
    rep_scores = []

    for rep in expert_repetitions:
        start_frame = rep["start_frame"]
        mid_frame = rep.get("mid_frame")
        end_frame = rep["end_frame"]

        # Si no hay mid_frame, asignar puntuación baja
        if mid_frame is None or np.isnan(mid_frame):
            rep_scores.append(-1)
            continue

        # Calcular duración
        duration = end_frame - start_frame

        # Verificar suavidad del movimiento
        rep_data = expert_data.iloc[int(start_frame) : int(end_frame)]
        wrist_cols = [
            col for col in rep_data.columns if "wrist" in col and col.endswith("_y")
        ]

        if not wrist_cols:
            rep_scores.append(0)
            continue

        smoothness = 0
        for col in wrist_cols:
            # Calcular derivada
            deriv = np.diff(rep_data[col].values)
            # Calcular varianza (menos varianza = más suave)
            smoothness -= np.var(deriv)

        # Puntuación para duración: preferir duraciones medias
        duration_score = -abs(duration - 80) / 20

        # Combinar criterios
        total_score = duration_score + smoothness
        rep_scores.append(total_score)

    # Retornar repetición con mayor puntuación
    best_rep_idx = np.argmax(rep_scores)
    return expert_repetitions[best_rep_idx]


def _match_repetitions_by_similarity(
    user_repetitions, expert_repetitions, user_data, expert_data
):
    """Helper function to match repetitions by similarity."""
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

            # Diferencia relativa de duración
            duration_diff = abs(u_duration - e_duration) / max(u_duration, e_duration)

            # Extraer trayectoria de muñeca (normalizada en tiempo)
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

            # Remuestrear para comparación directa
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

            # Calcular distancia promedio entre trayectorias
            traj_distance = np.mean(np.abs(u_interp - e_interp))

            # Puntuación de similitud (menor es mejor)
            similarity_score = duration_diff + traj_distance
            similarity_matrix[i, j] = similarity_score

    # Asignar repeticiones de forma voraz
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

            # Penalizar esta columna para evitar reutilizar mismo experto
            similarity_matrix[:, j] += 100
        else:
            break

    return matched_pairs


#############################################################################
# PHASE IDENTIFICATION AND SEGMENTATION
#############################################################################


def identify_phases(data, repetition, config=None):
    """
    Identifies movement phases within a repetition.
    Default is to divide into upward and downward phases.
    """
    config = config or {}
    start_frame = repetition.get("start_frame", 0)
    mid_frame = repetition.get("mid_frame", None)
    end_frame = repetition.get("end_frame", 0)

    # Si tenemos el frame medio, dividir en fases ascendente y descendente
    if mid_frame is not None and not np.isnan(mid_frame):
        # Fase ascendente: inicio → medio
        # Fase descendente: medio → fin
        return [(int(start_frame), int(mid_frame)), (int(mid_frame), int(end_frame))]
    else:
        # Si no hay mid_frame, analizar curva para dividir
        phase_strategy = config.get("phase_strategy", "auto")

        if phase_strategy == "auto":
            # Detección automática de fases mediante análisis de curvas
            segment = data.iloc[int(start_frame) : int(end_frame)].copy()

            # Usar combinación de ambas muñecas para robustez
            wrist_y = (
                -segment[["landmark_right_wrist_y", "landmark_left_wrist_y"]]
                .min(axis=1)
                .values
            )

            # Suavizar para encontrar picos/valles más robustos
            window_size = min(
                11, len(wrist_y) - 1 if len(wrist_y) % 2 == 0 else len(wrist_y)
            )
            wrist_y_smooth = savgol_filter(wrist_y, window_size, 2)

            # Encontrar punto más bajo/alto para dividir en fases
            if config.get("exercise_type", "press") == "press":  # Ejercicio de empuje
                lowest_point_idx = np.argmin(wrist_y_smooth)
                mid_frame_estimated = start_frame + lowest_point_idx
            else:  # Ejercicio de tracción
                highest_point_idx = np.argmax(wrist_y_smooth)
                mid_frame_estimated = start_frame + highest_point_idx

            # Verificar que el punto medio está dentro de la ventana
            if mid_frame_estimated <= start_frame or mid_frame_estimated >= end_frame:
                # Fallback: dividir por la mitad
                mid_frame_estimated = start_frame + (end_frame - start_frame) // 2
                logger.warning(
                    "Could not automatically identify phase. Dividing in half."
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
            # Por defecto: dividir en dos partes iguales
            midpoint = start_frame + (end_frame - start_frame) / 2
            return [(int(start_frame), int(midpoint)), (int(midpoint), int(end_frame))]


def divide_segment_adaptative(data, start_frame, end_frame, config=None):
    """
    Divides a segment into subsegments using the specified strategy.
    """
    config = config or {}
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
        return _divide_segment_by_acceleration(
            data, start_frame, end_frame, num_divisions, config
        )

    else:
        # Por defecto: división en partes iguales
        step = (end_frame - start_frame) / num_divisions
        return [int(start_frame + i * step) for i in range(num_divisions + 1)]


def _divide_segment_by_height(
    data, start_frame, end_frame, num_divisions, landmarks=None, axis="y"
):
    """Helper function to divide segment based on landmark height."""
    if start_frame >= end_frame or start_frame < 0 or end_frame > len(data):
        raise ValueError(
            f"Invalid range: start_frame={start_frame}, end_frame={end_frame}"
        )

    # Segmento de datos entre start_frame y end_frame
    segment = data.iloc[start_frame:end_frame].reset_index(drop=True)

    # Por defecto, usar muñecas
    if landmarks is None:
        landmarks = ["landmark_right_wrist", "landmark_left_wrist"]

    # Recopilar columnas relevantes
    height_columns = []
    for landmark in landmarks:
        col = f"{landmark}_{axis}"
        if col in segment.columns:
            height_columns.append(col)

    if not height_columns:
        raise ValueError("No height columns found for specified landmarks")

    # Para ejercicios de press, invertir eje Y
    heights = -segment[height_columns].min(axis=1)

    if heights.empty:
        raise ValueError(f"Empty heights in segment from {start_frame} to {end_frame}.")

    # Calcular alturas de división
    division_heights = np.linspace(heights.iloc[0], heights.iloc[-1], num_divisions + 1)

    # Identificar frames de división
    division_frames_relative = [
        (heights - target_height).abs().idxmin()
        for target_height in division_heights[1:-1]
    ]

    division_frames = (
        [start_frame]
        + [start_frame + rel for rel in division_frames_relative]
        + [end_frame]
    )

    # Asegurar que los frames estén ordenados y sean únicos
    division_frames = sorted(list(set(division_frames)))

    return division_frames


def _divide_segment_by_acceleration(
    data, start_frame, end_frame, num_divisions, config=None
):
    """Helper function to divide segment based on acceleration changes."""
    config = config or {}
    segment = data.iloc[start_frame:end_frame].copy()
    landmarks = config.get(
        "division_landmarks", ["landmark_right_wrist", "landmark_left_wrist"]
    )
    axis = config.get("division_axis", "y")

    # Calcular segunda derivada (aceleración)
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
        # Fallback a división equitativa
        return [
            int(start_frame + i * (end_frame - start_frame) / num_divisions)
            for i in range(num_divisions + 1)
        ]

    # Usar suma de aceleraciones
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
        [start_frame] + [start_frame + idx for idx in significant_points] + [end_frame]
    )

    return division_frames


#############################################################################
# INTERPOLATION AND SYNCHRONIZATION
#############################################################################


def interpolate_segment(original_data, target_length, method="linear"):
    """
    Interpolates a data segment to fit a target length.
    """
    if original_data.empty:
        raise ValueError("Empty segment received in interpolate_segment.")

    if target_length <= 1:
        raise ValueError(f"Invalid target length: {target_length}")

    # Si ya tiene la longitud correcta, devolver sin cambios
    if len(original_data) == target_length:
        return original_data.copy()

    original_frames = np.linspace(0, 1, len(original_data))
    target_frames = np.linspace(0, 1, target_length)

    interpolated_data = pd.DataFrame()

    # Interpolar cada columna numérica
    for col in original_data.select_dtypes(include=[np.number]).columns:
        # Omitir si todas son NaN
        if original_data[col].isna().all():
            interpolated_data[col] = [np.nan] * target_length
            continue

        # Valores no-NaN para interpolación
        valid_idx = ~original_data[col].isna()
        valid_frames = original_frames[valid_idx]
        valid_values = original_data[col].dropna().values

        # Si no hay suficientes valores
        if len(valid_values) <= 1:
            value = valid_values[0] if len(valid_values) > 0 else np.nan
            interpolated_data[col] = [value] * target_length
            continue

        # Crear interpolador con valores válidos
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
            logger.warning(f"Error interpolating column {col}: {e}")
            # Fallback a método más simple
            interpolated_data[col] = np.interp(
                target_frames,
                valid_frames,
                valid_values,
                left=valid_values[0],
                right=valid_values[-1],
            )

    return interpolated_data


def synchronize_subsegments(
    user_data, expert_data, user_frames, expert_frames, config=None
):
    """
    Synchronizes subsegments between user and expert through interpolation.
    """
    config = config or {}
    user_segments = []
    expert_segments = []

    # Método de interpolación
    interp_method = config.get("interp_method", "linear")

    # Dirección de adaptación
    adapt_direction = config.get("adapt_direction", "expert_to_user")

    # Procesar cada par de subsegmentos
    for i in range(len(user_frames) - 1):
        user_sub_segment = user_data.iloc[user_frames[i] : user_frames[i + 1]]
        expert_sub_segment = expert_data.iloc[expert_frames[i] : expert_frames[i + 1]]

        if user_sub_segment.empty or expert_sub_segment.empty:
            logger.warning(
                f"Empty segment found: user={user_sub_segment.empty}, "
                f"expert={expert_sub_segment.empty}"
            )
            continue

        if adapt_direction == "expert_to_user":
            # Interpolar expert_sub_segment para que coincida con la longitud de user_sub_segment
            interpolated_sub_segment = interpolate_segment(
                expert_sub_segment, len(user_sub_segment), interp_method
            )
            user_segments.append(user_sub_segment)
            expert_segments.append(interpolated_sub_segment)

        elif adapt_direction == "user_to_expert":
            # Interpolar user_sub_segment para que coincida con la longitud de expert_sub_segment
            interpolated_sub_segment = interpolate_segment(
                user_sub_segment, len(expert_sub_segment), interp_method
            )
            user_segments.append(interpolated_sub_segment)
            expert_segments.append(expert_sub_segment)

        elif adapt_direction == "both_to_average":
            # Interpolar ambos a longitud promedio
            avg_length = (len(user_sub_segment) + len(expert_sub_segment)) // 2
            interpolated_user = interpolate_segment(
                user_sub_segment, avg_length, interp_method
            )
            interpolated_expert = interpolate_segment(
                expert_sub_segment, avg_length, interp_method
            )
            user_segments.append(interpolated_user)
            expert_segments.append(interpolated_expert)

        else:
            # Por defecto: adaptar experto a usuario
            interpolated_sub_segment = interpolate_segment(
                expert_sub_segment, len(user_sub_segment), interp_method
            )
            user_segments.append(user_sub_segment)
            expert_segments.append(interpolated_sub_segment)

    # Concatenar segmentos
    if not user_segments or not expert_segments:
        raise ValueError("No valid segments generated during synchronization")

    user_processed = pd.concat(user_segments).reset_index(drop=True)
    expert_processed = pd.concat(expert_segments).reset_index(drop=True)

    return user_processed, expert_processed


def process_repetition_pair(user_data, expert_data, user_rep, expert_rep, config=None):
    """
    Processes a pair of repetitions (user-expert) by dividing into phases
    and synchronizing the subsegments.
    """
    config = config or {}

    # Identificar fases para usuario y experto
    user_phases = identify_phases(user_data, user_rep, config)
    expert_phases = identify_phases(expert_data, expert_rep, config)

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
        user_frames = divide_segment_adaptative(
            user_data, user_phase_start, user_phase_end, config
        )

        expert_frames = divide_segment_adaptative(
            expert_data, expert_phase_start, expert_phase_end, config
        )

        # Asegurar que ambos tengan el mismo número de divisiones
        min_divisions = min(len(user_frames), len(expert_frames)) - 1
        user_frames = user_frames[: min_divisions + 1]
        expert_frames = expert_frames[: min_divisions + 1]

        # Sincronizar los subsegmentos
        user_phase_proc, expert_phase_proc = synchronize_subsegments(
            user_data, expert_data, user_frames, expert_frames, config
        )

        user_processed_segments.append(user_phase_proc)
        expert_processed_segments.append(expert_phase_proc)

    # Concatenar todas las fases procesadas
    user_processed = pd.concat(user_processed_segments).reset_index(drop=True)
    expert_processed = pd.concat(expert_processed_segments).reset_index(drop=True)

    return user_processed, expert_processed


def process_all_repetitions(user_data, expert_data, repetition_pairs, config=None):
    """
    Processes all paired repetitions sequentially.
    """
    config = config or {}
    user_processed_segments = []
    expert_processed_segments = []

    # Procesamiento secuencial
    for user_rep, expert_rep in repetition_pairs:
        try:
            user_proc, expert_proc = process_repetition_pair(
                user_data, expert_data, user_rep, expert_rep, config
            )
            user_processed_segments.append(user_proc)
            expert_processed_segments.append(expert_proc)
        except Exception as e:
            logger.error(f"Error processing repetition: {e}")

    return user_processed_segments, expert_processed_segments


def combine_and_validate(user_segments, expert_segments):
    """
    Combines processed segments and verifies results are valid.
    """
    if not user_segments or not expert_segments:
        raise ValueError("No segments to combine")

    # Verificar que para cada segmento de usuario hay un segmento de experto
    if len(user_segments) != len(expert_segments):
        raise ValueError(
            f"Different number of segments: user={len(user_segments)}, "
            f"expert={len(expert_segments)}"
        )

    # Verificar que los segmentos correspondientes tengan el mismo tamaño
    for i, (user_seg, expert_seg) in enumerate(zip(user_segments, expert_segments)):
        if len(user_seg) != len(expert_seg):
            raise ValueError(
                f"Segment {i} has different lengths: "
                f"user={len(user_seg)}, expert={len(expert_seg)}"
            )

    # Combinar todos los segmentos
    final_user_data = pd.concat(user_segments, axis=0).reset_index(drop=True)
    final_expert_data = pd.concat(expert_segments, axis=0).reset_index(drop=True)

    # Verificar que tienen el mismo número total de frames
    if len(final_user_data) != len(final_expert_data):
        raise ValueError(
            f"Mismatch in number of frames: user={len(final_user_data)}, "
            f"expert={len(final_expert_data)}"
        )

    # Renumerar frames si existe la columna
    if "frame" in final_user_data.columns and "frame" in final_expert_data.columns:
        new_frames = np.arange(len(final_user_data))
        final_user_data["frame"] = new_frames
        final_expert_data["frame"] = new_frames

    logger.info(
        f"Synchronization completed: {len(final_user_data)} frames synchronized"
    )

    return final_user_data, final_expert_data
