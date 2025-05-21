# backend/src/core/data_extraction/pose_landmarker.py
import os
import logging
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2

# Configuración del logging
logger = logging.getLogger(__name__)


class PoseLandmarker:
    """
    Clase que encapsula el ciclo de vida del modelo de detección de poses de MediaPipe.
    Implementada con patrón singleton para mantener una única instancia.
    """

    # Variable de clase para almacenar la instancia singleton
    _instance = None

    @classmethod
    def get_instance(cls, model_path=None):
        """
        Método de clase para obtener la instancia singleton.

        Args:
            model_path: Ruta al modelo de pose (opcional)

        Returns:
            Instancia única de PoseLandmarker
        """
        if cls._instance is None:
            cls._instance = cls(model_path)
        elif model_path is not None and cls._instance.model_path != model_path:
            cls._instance.load_model(model_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        Reinicia la instancia singleton, liberando recursos.
        Útil cuando se necesita forzar una recarga completa.
        """
        if cls._instance is not None:
            cls._instance.release_resources()
            cls._instance = None

    def __init__(self, model_path=None):
        """
        Inicializa el PoseLandmarker, opcionalmente cargando un modelo.

        Args:
            model_path (str, optional): Ruta al archivo del modelo.
        """
        self.model_path = model_path
        self.landmarker = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Carga el modelo de detección de poses desde la ruta especificada.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

        BaseOptions = mp_python.BaseOptions
        PoseLandmarker = mp_vision.PoseLandmarker
        PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
        VisionRunningMode = mp_vision.RunningMode

        try:
            # Liberar recursos previos si existe una instancia anterior
            if self.landmarker:
                logger.info(
                    "Liberando recursos del modelo anterior antes de cargar uno nuevo"
                )
                self.release_resources()

            logger.info(f"Cargando modelo desde: {model_path}")
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_buffer=open(model_path, "rb").read()
                ),
                running_mode=VisionRunningMode.VIDEO,
            )

            self.landmarker = PoseLandmarker.create_from_options(options)
            self.model_path = model_path
            logger.info("Modelo cargado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            self.landmarker = None
            raise

    def detect_pose(self, frame, timestamp_ms):
        """
        Detecta poses en un frame de video.
        """
        if self.landmarker is None:
            raise ValueError("El modelo no está cargado. Llame a load_model primero.")

        # Convertir a formato RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detectar poses
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def release_resources(self):
        """
        Libera recursos asociados con el modelo.
        """
        if self.landmarker:
            self.landmarker = None
            logger.info("Recursos del modelo liberados")

    def __del__(self):
        """
        Destructor de la clase, asegurando la liberación de recursos.
        """
        self.release_resources()
