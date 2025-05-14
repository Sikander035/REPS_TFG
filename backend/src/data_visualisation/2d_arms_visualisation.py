import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ruta al archivo CSV
archivo_csv = (
    "out/normalized_VIDEOPRESS_joints.csv"  # Cambia esto por la ruta de tu archivo
)

# Cargar datos del CSV
datos = pd.read_csv(archivo_csv)


# Parámetros de visualización
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)

# Dibujar el brazo izquierdo y derecho cuadro por cuadro
for frame_id in datos["frame"].unique():
    frame_datos = datos[datos["frame"] == frame_id]

    # Extraer posiciones
    landmarks = {}
    for columna in frame_datos.columns:
        if "_x" in columna:
            base = columna[:-2]
            x = frame_datos[f"{base}_x"].values[0]
            y = frame_datos[f"{base}_y"].values[0]
            landmarks[base] = np.array([x, y])

    # Centrar los brazos en sus respectivos hombros
    centro_hombro_izquierdo = landmarks["landmark_left_shoulder"]
    centro_hombro_derecho = landmarks["landmark_right_shoulder"]

    brazo_izquierdo = {
        "wrist": landmarks["landmark_left_wrist"] - centro_hombro_izquierdo,
        "elbow": landmarks["landmark_left_elbow"] - centro_hombro_izquierdo,
        "shoulder": np.array([0, 0]),  # El centro es el propio hombro
    }

    brazo_derecho = {
        "wrist": landmarks["landmark_right_wrist"] - centro_hombro_derecho,
        "elbow": landmarks["landmark_right_elbow"] - centro_hombro_derecho,
        "shoulder": np.array([0, 0]),  # El centro es el propio hombro
    }

    # Limpiar gráfico para el nuevo frame
    ax.cla()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"Frame {frame_id}")

    # Dibujar brazo izquierdo
    x_coords = [
        brazo_izquierdo["wrist"][0],
        brazo_izquierdo["elbow"][0],
        brazo_izquierdo["shoulder"][0],
    ]
    y_coords = [
        brazo_izquierdo["wrist"][1],
        brazo_izquierdo["elbow"][1],
        brazo_izquierdo["shoulder"][1],
    ]
    ax.plot(x_coords, y_coords, color="blue", marker="o", label="Brazo Izquierdo")

    # Dibujar brazo derecho
    x_coords = [
        brazo_derecho["wrist"][0],
        brazo_derecho["elbow"][0],
        brazo_derecho["shoulder"][0],
    ]
    y_coords = [
        brazo_derecho["wrist"][1],
        brazo_derecho["elbow"][1],
        brazo_derecho["shoulder"][1],
    ]
    ax.plot(x_coords, y_coords, color="green", marker="o", label="Brazo Derecho")

    # Leyenda
    ax.legend()

    # Mostrar frame
    plt.pause(0.1)

plt.show()
