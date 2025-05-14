import pandas as pd
import numpy as np
import cv2

# Ruta al archivo CSV
archivo_csv = (
    "out/normalized_VIDEOPRESS_joints.csv"  # Cambia esto por la ruta de tu archivo
)

# Cargar datos del CSV
datos = pd.read_csv(archivo_csv)

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

# Parámetros del lienzo
ancho, alto = 640, 480  # Tamaño de la ventana
color_fondo = (0, 0, 0)  # Negro
color_puntos = (0, 255, 0)  # Verde
color_lineas = (255, 0, 0)  # Azul
radio_puntos = 5


# Función para normalizar coordenadas al tamaño del lienzo
def normalizar_coordenadas(x, y, ancho, alto):
    x = int(x * ancho)
    y = int(y * alto)
    return x, y


# Dibujar el esqueleto cuadro por cuadro
for frame_id in datos["frame"].unique():
    frame_datos = datos[datos["frame"] == frame_id]

    # Crear un lienzo negro
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)

    # Diccionario para almacenar las coordenadas 2D normalizadas de los landmarks
    landmarks = {}

    # Leer las posiciones de los landmarks y normalizar
    for columna in frame_datos.columns:
        if "_x" in columna:
            base = columna[:-2]
            x = frame_datos[f"{base}_x"].values[0]
            y = frame_datos[f"{base}_y"].values[0]
            landmarks[base] = normalizar_coordenadas(x, y, ancho, alto)

    # Dibujar conexiones
    for punto1, punto2 in conexiones:
        if punto1 in landmarks and punto2 in landmarks:
            cv2.line(lienzo, landmarks[punto1], landmarks[punto2], color_lineas, 2)

    # Dibujar puntos
    for punto, (x, y) in landmarks.items():
        cv2.circle(lienzo, (x, y), radio_puntos, color_puntos, -1)

    # Mostrar el frame
    cv2.imshow("Esqueleto", lienzo)

    # Esperar un momento para mostrar el cuadro
    if cv2.waitKey(40) & 0xFF == ord("q"):  # Presiona 'q' para salir
        break

# Cerrar ventanas al terminar
cv2.destroyAllWindows()
