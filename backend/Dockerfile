FROM python:3.12.7-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    libavcodec-extra \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip install --upgrade pip

# REPLICAR EXACTAMENTE tu entorno local que funciona:

# 1. NumPy con tu versión exacta
RUN pip install numpy==1.26.4

# 2. Protobuf con tu versión exacta (¡CLAVE!)
RUN pip install protobuf==4.25.7

# 3. OpenCV - Versión EXACTA 4.10.0.84 (ambos paquetes compatibles)
RUN pip install opencv-python==4.10.0.84
RUN pip install opencv-contrib-python==4.10.0.84

# 4. MediaPipe con tu versión (ahora debería funcionar)
RUN pip install mediapipe==0.10.20

# 5. Resto de dependencias de tu app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Mostrar versiones para confirmar
RUN pip list | grep -E "(mediapipe|opencv|protobuf|numpy)"

# TEST CRÍTICO: Verificar versiones exactas
RUN python -c 'import cv2; import mediapipe as mp; print(f"OpenCV: {cv2.__version__}"); assert cv2.__version__.startswith("4.10.0"), f"ERROR: OpenCV {cv2.__version__} no es 4.10.0.x"; print("✅ OpenCV versión correcta: 4.10.0.x"); mp_pose = mp.solutions.pose; pose = mp_pose.Pose(); print("✅ MediaPipe Pose funciona"); pose.close(); print("✅ ¡Setup perfecto!")'

COPY . .

EXPOSE 8000

CMD ["python", "./api/main.py"]