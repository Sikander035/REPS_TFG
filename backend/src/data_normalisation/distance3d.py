import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta al archivo CSV
archivo_csv = (
    "out/normalized_VIDEOPRESS_joints.csv"  # Cambia esto por la ruta de tu archivo
)

# Cargar datos del CSV
datos = pd.read_csv(archivo_csv)


# Función para calcular la distancia tridimensional entre dos puntos
def calcular_distancia_3d(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# Lista de conexiones a graficar
conexiones = [
    ("landmark_left_shoulder", "landmark_right_shoulder"),
    ("landmark_left_shoulder", "landmark_left_elbow"),
    ("landmark_right_shoulder", "landmark_right_elbow"),
    ("landmark_left_elbow", "landmark_left_wrist"),
    ("landmark_right_elbow", "landmark_right_wrist"),
    ("landmark_left_shoulder", "landmark_left_hip"),
    ("landmark_right_shoulder", "landmark_right_hip"),
    ("landmark_left_hip", "landmark_right_hip"),
]

# Diccionario para almacenar las distancias de cada conexión
distancias_por_conexion = {conexion: [] for conexion in conexiones}

# Recorrer los frames únicos
for frame_id in datos["frame"].unique():
    # Filtrar datos del frame actual
    frame_datos = datos[datos["frame"] == frame_id]

    for conexion in conexiones:
        punto1, punto2 = conexion

        # Obtener coordenadas de los puntos
        x1, y1, z1 = (
            frame_datos[f"{punto1}_x"].values[0],
            frame_datos[f"{punto1}_y"].values[0],
            frame_datos[f"{punto1}_z"].values[0],
        )
        x2, y2, z2 = (
            frame_datos[f"{punto2}_x"].values[0],
            frame_datos[f"{punto2}_y"].values[0],
            frame_datos[f"{punto2}_z"].values[0],
        )

        # Calcular la distancia tridimensional y agregarla al diccionario
        distancia = calcular_distancia_3d(x1, y1, z1, x2, y2, z2)
        distancias_por_conexion[conexion].append(distancia)

# Crear un gráfico para cada conexión
plt.figure(figsize=(12, 8))
for conexion, distancias in distancias_por_conexion.items():
    plt.plot(
        range(len(distancias)),
        distancias,
        label=f"{conexion[0]} ↔ {conexion[1]}",
        linewidth=1.5,
    )

# Calcula la variación de la distancia entre las conexiones
for conexion, distancias in distancias_por_conexion.items():
    variacion = (max(distancias) - min(distancias)) / max(distancias)
    print(
        f"Variación de la distancia entre {conexion[0]} y {conexion[1]}: {variacion:.2f}"
    )

# Configuración del gráfico
plt.xlabel("Frame")
plt.ylabel("Distancia (normalizada)")
plt.title("Evolución de la distancia 3D entre conexiones")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
