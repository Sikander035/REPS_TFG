import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import imageio
from moviepy.editor import ImageSequenceClip


def generate_dual_skeleton_video(user_data, example_data, ruta_video_output):
    """
    Genera un video que compara visualmente dos esqueletos en 3D a partir de archivos CSV.

    Parámetros:
        user_data: Landmarks del primer esqueleto.
        example_data: Landmarks del segundo esqueleto.
        ruta_video_output (str): Ruta donde se guardará el video generado.
    """

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

    user_data = suavizar_datos(user_data, ventana=5)
    example_data = suavizar_datos(example_data, ventana=5)

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
    for frame_id in user_data["frame"].unique():
        frame_user_data = user_data[user_data["frame"] == frame_id]
        frame_example_data = example_data[example_data["frame"] == frame_id]

        # Diccionario para almacenar las coordenadas 3D de los landmarks
        landmarks1 = {}
        landmarks2 = {}

        # Leer las posiciones de los landmarks para el primer esqueleto
        for columna in frame_user_data.columns:
            if "_x" in columna:
                base = columna[:-2]
                x = frame_user_data[f"{base}_x"].values[0]
                y = frame_user_data[f"{base}_y"].values[0]
                z = frame_user_data[f"{base}_z"].values[0]
                landmarks1[base] = np.array([x, y, z])

        # Leer las posiciones de los landmarks para el segundo esqueleto
        for columna in frame_example_data.columns:
            if "_x" in columna:
                base = columna[:-2]
                x = frame_example_data[f"{base}_x"].values[0]
                y = frame_example_data[f"{base}_y"].values[0]
                z = frame_example_data[f"{base}_z"].values[0]
                landmarks2[base] = np.array([x, y, z])

        # Calcular el punto medio de las caderas para ambos esqueletos
        centro_caderas1 = (
            landmarks1["landmark_left_hip"] + landmarks1["landmark_right_hip"]
        ) / 2
        centro_caderas2 = (
            landmarks2["landmark_left_hip"] + landmarks2["landmark_right_hip"]
        ) / 2

        # Ajustar el punto de centrado más abajo para ambos esqueletos
        punto_centro1 = centro_caderas1 - np.array([0, 0.45, 0.2])
        punto_centro2 = centro_caderas2 - np.array([0, 0.45, 0.2])

        # Recolocar todos los puntos para centrar y desplazar los esqueletos
        desplazamiento_x = 0.39  # Ajusta este valor para mover más a la izquierda
        for key in landmarks1.keys():
            landmarks1[key] -= punto_centro1
            landmarks1[key][0] -= desplazamiento_x
        for key in landmarks2.keys():
            landmarks2[key] -= punto_centro2
            landmarks2[key][0] -= desplazamiento_x

        # Crear la figura 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-1, 1)
        ax.axis("off")
        ax.view_init(elev=-88, azim=-90)

        # Dibujar conexiones para ambos esqueletos
        for punto1, punto2 in conexiones:
            if punto1 in landmarks1 and punto2 in landmarks1:
                ax.plot(
                    [landmarks1[punto1][0], landmarks1[punto2][0]],
                    [landmarks1[punto1][1], landmarks1[punto2][1]],
                    [landmarks1[punto1][2], landmarks1[punto2][2]],
                    color="blue",
                    linewidth=2,
                )
            if punto1 in landmarks2 and punto2 in landmarks2:
                ax.plot(
                    [landmarks2[punto1][0], landmarks2[punto2][0]],
                    [landmarks2[punto1][1], landmarks2[punto2][1]],
                    [landmarks2[punto1][2], landmarks2[punto2][2]],
                    color="red",
                    linewidth=2,
                )

        # Dibujar puntos para ambos esqueletos
        for punto, coords in landmarks1.items():
            ax.scatter(coords[0], coords[1], coords[2], color="blue", s=50)
        for punto, coords in landmarks2.items():
            ax.scatter(coords[0], coords[1], coords[2], color="red", s=50)

        # Añadir leyenda visual
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
        fig.text(0.09, 0.91, "Tu ejecución", color="black", fontsize=10)
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
        fig.text(0.09, 0.86, "Ejecución correcta", color="black", fontsize=10)

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
    clip = ImageSequenceClip(frame_files, fps=30)
    clip.write_videofile(
        ruta_video_output,
        codec="libx264",
        audio=False,
        fps=30,  # Ajustar fps según preferencia
        preset="ultrafast",
    )

    # Eliminar los frames temporales
    for fname in frame_files:
        os.remove(fname)
    os.rmdir(output_dir)
