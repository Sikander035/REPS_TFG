import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from data_segmentation.detect_repetitions import detect_repetitions


def interpolate_segment(original_data, target_length):
    """
    Interpola un segmento de datos para ajustarlo a una longitud objetivo.
    """
    if original_data.empty:
        raise ValueError("Segmento vacío recibido en interpolate_segment.")

    original_frames = np.linspace(0, 1, len(original_data))
    target_frames = np.linspace(0, 1, target_length)

    interpolated_data = pd.DataFrame()
    for col in original_data.select_dtypes(include=[np.number]).columns:
        interpolator = interp1d(
            original_frames,
            original_data[col].dropna(),
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated_data[col] = interpolator(target_frames)
    return interpolated_data


def divide_by_height(data, start_frame, end_frame, num_divisions):
    """
    Divide un tramo de datos en subsegmentos basados en divisiones iguales de altura.
    Además, grafica el segmento completo, marcando el inicio, el final, y las divisiones.
    """
    if start_frame >= end_frame or start_frame < 0 or end_frame > len(data):
        raise ValueError(
            f"Rango inválido: start_frame={start_frame}, end_frame={end_frame}"
        )

    # Segmento de datos entre start_frame y end_frame
    segment = data.iloc[start_frame:end_frame].reset_index(drop=True)
    heights = -segment[["landmark_right_wrist_y", "landmark_left_wrist_y"]].min(axis=1)

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

    return division_frames


def synchronize_data_by_height(user_data, example_data, num_divisions=7):
    """
    Sincroniza los datos dividiendo cada tramo en segmentos basados en alturas.
    """
    print("Detectando repeticiones...")
    user_repetitions = detect_repetitions(user_data)
    example_repetition = detect_repetitions(example_data)

    if not user_repetitions or not example_repetition:
        raise ValueError(
            "No se detectaron repeticiones en uno o ambos conjuntos de datos."
        )

    print(f"Repeticiones usuario: {len(user_repetitions)}")
    print(f"Repeticiones ejemplo: {len(example_repetition)}")

    # Replicar la primera repetición del ejemplo tantas veces como repeticiones de usuario
    example_repetitions = [example_repetition[0]] * len(user_repetitions)

    processed_user_data = []
    processed_example_data = []

    for user_rep, example_rep in zip(user_repetitions, example_repetitions):
        print(f"Procesando repetición: usuario {user_rep}, ejemplo {example_rep}")

        # Dividir segmentos por alturas en subida y bajada
        user_up_frames = divide_by_height(
            user_data, user_rep["start_frame"], user_rep["mid_frame"], num_divisions
        )
        user_down_frames = divide_by_height(
            user_data, user_rep["mid_frame"], user_rep["end_frame"], num_divisions
        )

        example_up_frames = divide_by_height(
            example_data,
            example_rep["start_frame"],
            example_rep["mid_frame"],
            num_divisions,
        )
        example_down_frames = divide_by_height(
            example_data,
            example_rep["mid_frame"],
            example_rep["end_frame"],
            num_divisions,
        )

        # Asegurarse de que los tramos no estén vacíos
        print(f"Tramos de usuario (subida): {user_up_frames}")
        print(f"Tramos de ejemplo (subida): {example_up_frames}")
        print(f"Tramos de usuario (bajada): {user_down_frames}")
        print(f"Tramos de ejemplo (bajada): {example_down_frames}")

        interpolated_segments = []

        # Interpolar segmentos de subida
        for i in range(len(user_up_frames) - 1):
            user_sub_segment = user_data.iloc[user_up_frames[i] : user_up_frames[i + 1]]
            example_sub_segment = example_data.iloc[
                example_up_frames[i] : example_up_frames[i + 1]
            ]

            if user_sub_segment.empty or example_sub_segment.empty:
                print(
                    f"Advertencia: un segmento vacío encontrado: usuario={user_sub_segment.empty}, ejemplo={example_sub_segment.empty}"
                )
                continue  # Ignorar este segmento

            interpolated_sub_segment = interpolate_segment(
                example_sub_segment, len(user_sub_segment)
            )
            interpolated_segments.append(interpolated_sub_segment)

        # Interpolar segmentos de bajada
        for i in range(len(user_down_frames) - 1):
            user_sub_segment = user_data.iloc[
                user_down_frames[i] : user_down_frames[i + 1]
            ]
            example_sub_segment = example_data.iloc[
                example_down_frames[i] : example_down_frames[i + 1]
            ]

            if user_sub_segment.empty or example_sub_segment.empty:
                print(
                    f"Advertencia: un segmento vacío encontrado: usuario={user_sub_segment.empty}, ejemplo={example_sub_segment.empty}"
                )
                continue  # Ignorar este segmento

            interpolated_sub_segment = interpolate_segment(
                example_sub_segment, len(user_sub_segment)
            )
            interpolated_segments.append(interpolated_sub_segment)

        # Concatenar resultados
        interpolated_example = pd.concat(interpolated_segments, axis=0).reset_index(
            drop=True
        )
        processed_user_data.append(
            user_data.iloc[user_rep["start_frame"] : user_rep["end_frame"]].reset_index(
                drop=True
            )
        )
        processed_example_data.append(interpolated_example)

    # Concatenar todas las repeticiones
    final_user_data = pd.concat(processed_user_data, axis=0).reset_index(drop=True)
    final_example_data = pd.concat(processed_example_data, axis=0).reset_index(
        drop=True
    )

    # Verificar si ambos tienen el mismo número de frames
    if len(final_user_data) != len(final_example_data):
        raise ValueError(
            f"Desajuste en el número de frames: usuario={len(final_user_data)}, "
            f"ejemplo={len(final_example_data)}"
        )

    # Renumerar los frames
    final_user_data["frame"] = np.arange(len(final_user_data))
    final_example_data["frame"] = np.arange(len(final_example_data))

    print("Sincronización completada exitosamente.")
    return final_user_data, final_example_data
