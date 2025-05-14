import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio
from moviepy.editor import ImageSequenceClip

# Ruta a los archivos CSV
archivo_csv1 = "out/normalized_VIDEOPRESS_joints.csv"  # Cambia esto por la ruta de tu primer archivo
archivo_csv2 = (
    "out/VIDEOPRESS_joints.csv"  # Cambia esto por la ruta de tu segundo archivo
)

# Cargar datos de ambos CSV
datos1 = pd.read_csv(archivo_csv1)
datos2 = pd.read_csv(archivo_csv2)


# Aplicar suavizado con un filtro de media móvil
def suavizar_datos(datos, ventana=5):
    """
    Suaviza las coordenadas de los landmarks usando un filtro de media móvil.
    """
    columnas = [col for col in datos.columns if col.endswith(("_x", "_y", "_z"))]
    for col in columnas:
        datos[col] = (
            datos[col].rolling(window=ventana, center=True, min_periods=1).mean()
        )
    return datos


# Suavizar los datos
datos1 = suavizar_datos(datos1, ventana=5)
datos2 = suavizar_datos(datos2, ventana=5)

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

# Crear una carpeta temporal para guardar los frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Generar y guardar cada frame
for frame_id in datos1["frame"].unique():
    frame_datos1 = datos1[datos1["frame"] == frame_id]
    frame_datos2 = datos2[datos2["frame"] == frame_id]

    # Diccionario para almacenar las coordenadas 3D de los landmarks
    landmarks1 = {}
    landmarks2 = {}

    # Leer las posiciones de los landmarks para el primer esqueleto
    for columna in frame_datos1.columns:
        if "_x" in columna:
            base = columna[:-2]
            x = frame_datos1[f"{base}_x"].values[0]
            y = frame_datos1[f"{base}_y"].values[0]
            z = frame_datos1[f"{base}_z"].values[0]
            landmarks1[base] = np.array([x, y, z])

    # Leer las posiciones de los landmarks para el segundo esqueleto
    for columna in frame_datos2.columns:
        if "_x" in columna:
            base = columna[:-2]
            x = frame_datos2[f"{base}_x"].values[0]
            y = frame_datos2[f"{base}_y"].values[0]
            z = frame_datos2[f"{base}_z"].values[0]
            landmarks2[base] = np.array([x, y, z])

    # Calcular el punto medio de las caderas para el primer esqueleto
    cadera_izquierda1 = landmarks1["landmark_left_hip"]
    cadera_derecha1 = landmarks1["landmark_right_hip"]
    centro_caderas1 = (cadera_izquierda1 + cadera_derecha1) / 2

    # Calcular el punto medio de las caderas para el segundo esqueleto
    cadera_izquierda2 = landmarks2["landmark_left_hip"]
    cadera_derecha2 = landmarks2["landmark_right_hip"]
    centro_caderas2 = (cadera_izquierda2 + cadera_derecha2) / 2

    # Ajustar el punto de centrado más abajo para ambos esqueletos
    punto_centro1 = centro_caderas1 - np.array([0, 0.45, 0.2])
    punto_centro2 = centro_caderas2 - np.array([0, 0.45, 0.2])

    # Recolocar todos los puntos para centrar el primer esqueleto
    for key in landmarks1.keys():
        landmarks1[key] -= punto_centro1

    # Recolocar todos los puntos para centrar el segundo esqueleto
    for key in landmarks2.keys():
        landmarks2[key] -= punto_centro2

    # Mover los esqueletos más a la izquierda
    desplazamiento_x = 0.39  # Ajusta este valor para mover más a la izquierda
    for key in landmarks1.keys():
        landmarks1[key][0] -= desplazamiento_x
    for key in landmarks2.keys():
        landmarks2[key][0] -= desplazamiento_x

    # Crear la figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-1, 1)

    # Ocultar los ejes, la cuadrícula y los números
    ax.axis("off")

    # Fijar la vista con elevación y azimuth
    ax.view_init(elev=-88, azim=-90)

    # Dibujar conexiones para el primer esqueleto
    for punto1, punto2 in conexiones:
        if punto1 in landmarks1 and punto2 in landmarks1:
            x_coords = [landmarks1[punto1][0], landmarks1[punto2][0]]
            y_coords = [landmarks1[punto1][1], landmarks1[punto2][1]]
            z_coords = [landmarks1[punto1][2], landmarks1[punto2][2]]
            ax.plot(
                x_coords,
                y_coords,
                z_coords,
                color="blue",
                linewidth=2,
                label="Esqueleto 1",
            )

    # Dibujar conexiones para el segundo esqueleto
    for punto1, punto2 in conexiones:
        if punto1 in landmarks2 and punto2 in landmarks2:
            x_coords = [landmarks2[punto1][0], landmarks2[punto2][0]]
            y_coords = [landmarks2[punto1][1], landmarks2[punto2][1]]
            z_coords = [landmarks2[punto1][2], landmarks2[punto2][2]]
            ax.plot(
                x_coords,
                y_coords,
                z_coords,
                color="red",
                linewidth=2,
                label="Esqueleto 2",
            )

    # Dibujar puntos para el primer esqueleto
    for punto, coords in landmarks1.items():
        ax.scatter(coords[0], coords[1], coords[2], color="blue", s=50)

    # Dibujar puntos para el segundo esqueleto
    for punto, coords in landmarks2.items():
        ax.scatter(coords[0], coords[1], coords[2], color="red", s=50)

    # Añadir un cuadrado azul y el texto correspondiente
    fig.patches.append(
        Rectangle(
            (0.05, 0.9085),
            0.03,
            0.03,
            color="blue",
            transform=fig.transFigure,
            figure=fig,
        )
    )
    fig.text(0.09, 0.91, "Esqueleto 1", color="black", fontsize=10)

    # Añadir un cuadrado rojo y el texto correspondiente
    fig.patches.append(
        Rectangle(
            (0.05, 0.858),
            0.03,
            0.03,
            color="red",
            transform=fig.transFigure,
            figure=fig,
        )
    )
    fig.text(0.09, 0.86, "Esqueleto 2", color="black", fontsize=10)

    # Guardar el frame como imagen
    plt.savefig(f"{output_dir}/frame_{frame_id:04d}.png", bbox_inches="tight")
    plt.close(fig)

# Crear un video a partir de los frames
frame_files = sorted(
    [
        f"{output_dir}/{fname}"
        for fname in os.listdir(output_dir)
        if fname.endswith(".png")
    ]
)
clip = ImageSequenceClip(frame_files, fps=30)  # Ajustar fps según preferencia
clip.write_videofile(
    "out/dual_skeleton_animation.mp4",
    codec="libx264",
    audio=False,
    fps=30,
    preset="ultrafast",
)


# Eliminar los frames temporales
for fname in frame_files:
    os.remove(fname)
os.rmdir(output_dir)
