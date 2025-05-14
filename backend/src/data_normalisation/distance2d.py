import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta al archivo CSV
archivo_csv = (
    "out/normalized2D_VIDEOPRESS_joints.csv"  # Cambia esto por la ruta de tu archivo
)

# Cargar datos del CSV
datos = pd.read_csv(archivo_csv)


# Función para calcular la distancia entre dos puntos
def calcular_distancia(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Puntos de interés: "landmark_left_wrist" y "landmark_left_elbow"
punto1 = "landmark_left_wrist"
punto2 = "landmark_left_elbow"

# Lista para almacenar las distancias por frame
distancias = []

# Recorrer los frames únicos
for frame_id in datos["frame"].unique():
    # Filtrar datos del frame actual
    frame_datos = datos[datos["frame"] == frame_id]

    # Obtener coordenadas de los puntos
    x1, y1 = frame_datos[f"{punto1}_x"].values[0], frame_datos[f"{punto1}_y"].values[0]
    x2, y2 = frame_datos[f"{punto2}_x"].values[0], frame_datos[f"{punto2}_y"].values[0]

    # Calcular la distancia y agregarla a la lista
    distancia = calcular_distancia(x1, y1, x2, y2)
    distancias.append(distancia)

# Crear un gráfico de la distancia a lo largo de los frames
plt.figure(figsize=(10, 6))
plt.plot(
    range(len(distancias)),
    distancias,
    label=f"Distancia: {punto1} ↔ {punto2}",
    color="blue",
)
plt.xlabel("Frame")
plt.ylabel("Distancia (normalizada)")
plt.title("Evolución de la distancia entre puntos")
plt.legend()
plt.grid(True)
plt.show()

# Calcular la media de las distancias y comprobar cuanto se aleja de la media como máximo
media = np.mean(distancias)

# Máximo en valores absolutos
maximo = np.max(np.abs(distancias))
print(f"Media de las distancias: {media}")
print(f"Distancia máxima (en valor absoluto): {maximo}")
print(f"Porcentaje de variación: {(maximo - media) / media * 100:.2f}%")
