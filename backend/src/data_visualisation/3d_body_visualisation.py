import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ruta al archivo CSV
archivo_csv = (
    "out/normalized_VIDEOPRESS_joints.csv"  # Cambia esto por la ruta de tu archivo
)

# Cargar datos del CSV
datos = pd.read_csv(archivo_csv)


# Aplicar suavizado con un filtro de media móvil
def suavizar_datos(datos, ventana=5):
    """
    Suaviza las coordenadas de los landmarks usando un filtro de media móvil.

    Parameters:
        datos (pd.DataFrame): Datos del CSV con coordenadas.
        ventana (int): Tamaño de la ventana de suavizado.

    Returns:
        pd.DataFrame: Datos suavizados.
    """
    columnas = [col for col in datos.columns if col.endswith(("_x", "_y"))]
    for col in columnas:
        datos[col] = (
            datos[col].rolling(window=ventana, center=True, min_periods=1).mean()
        )
    return datos


# Suavizar los datos
datos = suavizar_datos(datos, ventana=5)

# Definir las conexiones entre los puntos para el esqueleto
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

# Parámetros de visualización
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Dibujar el esqueleto cuadro por cuadro
for frame_id in datos["frame"].unique():
    frame_datos = datos[datos["frame"] == frame_id]

    # Diccionario para almacenar las coordenadas 3D de los landmarks
    landmarks = {}

    # Leer las posiciones de los landmarks
    for columna in frame_datos.columns:
        if "_x" in columna:
            base = columna[:-2]
            x = frame_datos[f"{base}_x"].values[0]
            y = frame_datos[f"{base}_y"].values[0]
            z = frame_datos[f"{base}_z"].values[0]
            landmarks[base] = np.array([x, y, z])

    # Calcular el punto medio de las caderas
    cadera_izquierda = landmarks["landmark_left_hip"]
    cadera_derecha = landmarks["landmark_right_hip"]
    centro_caderas = (cadera_izquierda + cadera_derecha) / 2

    # Ajustar el punto de centrado más abajo (e.g., 0.2 unidades debajo de las caderas)
    punto_centro = centro_caderas - np.array([0, 0.45, 0.2])

    # Recolocar todos los puntos para centrar el esqueleto en el nuevo punto
    for key in landmarks.keys():
        landmarks[key] -= punto_centro

    # Limpiar el gráfico para el nuevo frame
    ax.cla()
    ax.set_xlim(-0.8, 0.8)  # Ajustar rangos según los datos
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame_id}")

    # Dibujar conexiones
    for punto1, punto2 in conexiones:
        if punto1 in landmarks and punto2 in landmarks:
            x_coords = [landmarks[punto1][0], landmarks[punto2][0]]
            y_coords = [landmarks[punto1][1], landmarks[punto2][1]]
            z_coords = [landmarks[punto1][2], landmarks[punto2][2]]
            ax.plot(x_coords, y_coords, z_coords, color="blue", linewidth=2)

    # Dibujar puntos
    for punto, coords in landmarks.items():
        ax.scatter(coords[0], coords[1], coords[2], color="green", s=50)

    # Mostrar el frame
    plt.pause(0.01)

plt.show()
