FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiar los requisitos de Python
COPY requirements.txt .

# Instalar los paquetes de Python
RUN pip install --no-cache-dir -r ./requirements.txt

# Copiar el resto del código al contenedor
COPY . .

# Exponer el puerto de la aplicación
EXPOSE 8000

# Comando por defecto para iniciar la aplicación
CMD ["python", "./api/main.py"]
