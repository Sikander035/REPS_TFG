import pandas as pd
import numpy as np


def validate_connections(reference_lengths, skeleton_keys):
    """
    Valida que todas las conexiones especificadas en reference_lengths sean válidas
    y que ambos puntos de cada conexión existan en el esqueleto.

    Parameters:
        reference_lengths (list): Lista de tuplas (conexión, longitud deseada).
        skeleton_keys (set): Conjunto de nombres de joints en el esqueleto.

    Raises:
        ValueError: Si alguna conexión tiene joints no presentes en el esqueleto.
    """
    for (
        joint_a,
        joint_b,
    ), _ in reference_lengths:  # Desempaquetar las conexiones de la lista
        if joint_a not in skeleton_keys or joint_b not in skeleton_keys:
            raise ValueError(
                f"Conexión inválida: {joint_a} o {joint_b} no están en el esqueleto."
            )


def normalize_skeleton_frame(frame_data, reference_lengths):
    """
    Normaliza un esqueleto en un frame ajustando las proporciones según longitudes de referencia.

    Parameters:
        frame_data (pd.Series): Datos de un frame con coordenadas de landmarks.
        reference_lengths (dict): Diccionario con las articulaciones y sus longitudes deseadas.

    Returns:
        pd.Series: Frame con los landmarks normalizados.
    """
    # Extraer coordenadas de los landmarks ignorando visibilidad
    skeleton = {
        col.rsplit("_", 1)[0]: [
            frame_data[f"{col.rsplit('_', 1)[0]}_x"],
            frame_data[f"{col.rsplit('_', 1)[0]}_y"],
            frame_data[f"{col.rsplit('_', 1)[0]}_z"],
        ]
        for col in frame_data.index
        if col.endswith("_x") and not col.startswith("frame")
    }

    # Validar que todas las conexiones de reference_lengths son válidas
    validate_connections(reference_lengths, set(skeleton.keys()))

    # Crear una copia del esqueleto para modificaciones
    normalized_skeleton = skeleton.copy()

    # Normalizar las conexiones especificadas
    for (joint_a, joint_b), target_length in reference_lengths:
        point_a = np.array(skeleton[joint_a])
        point_b = np.array(skeleton[joint_b])

        point_a_normalized = np.array(normalized_skeleton[joint_a])

        vector_ab = point_b - point_a
        current_length = np.linalg.norm(vector_ab)

        if current_length == 0:
            print(f"Advertencia: Longitud cero entre {joint_a} y {joint_b}.")
            continue

        unit_vector = vector_ab / current_length
        new_vector_ab = unit_vector * target_length

        normalized_skeleton[joint_b] = point_a_normalized + new_vector_ab

    # Actualizar los datos del frame con los landmarks normalizados
    for joint, coords in normalized_skeleton.items():
        frame_data[f"{joint}_x"], frame_data[f"{joint}_y"], frame_data[f"{joint}_z"] = (
            coords
        )

    return frame_data


def normalize_skeleton(data, reference_lengths):
    """
    Normaliza los datos de esqueletos ajustando las proporciones según longitudes de referencia.

    Parameters:
        data (pd.DataFrame): Datos de entrada con frames y coordenadas de landmarks.
        reference_lengths (dict): Diccionario con las articulaciones y sus longitudes deseadas.

    Returns:
        pd.DataFrame: Datos normalizados.
    """
    # Filtrar columnas relevantes (frame y coordenadas x, y, z)
    frame_column = data["frame"]  # Guardar la columna frame
    data = data.filter(regex=r"landmark_.*_[xyz]$", axis=1)

    # Aplicar normalización frame por frame
    normalized_data = data.apply(
        lambda row: normalize_skeleton_frame(row, reference_lengths), axis=1
    )

    # Restaurar la columna frame en el resultado normalizado
    normalized_data.insert(0, "frame", frame_column)

    return normalized_data
