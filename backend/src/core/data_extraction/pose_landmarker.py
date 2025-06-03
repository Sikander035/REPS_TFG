# backend/src/core/data_extraction/pose_landmarker.py - THREAD-SAFE VERSION
import os
import logging
import threading
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2

# Configuración del logging
logger = logging.getLogger(__name__)


class PoseLandmarker:
    """
    Clase que encapsula el ciclo de vida del modelo de detección de poses de MediaPipe.
    Implementada con patrón singleton THREAD-SAFE para evitar conflictos en FastAPI.
    """

    # Variable de clase para almacenar la instancia singleton
    _instance = None
    _lock = threading.Lock()  # Lock para thread safety

    @classmethod
    def get_instance(cls, model_path=None):
        """
        Método de clase para obtener la instancia singleton de forma thread-safe.

        Args:
            model_path: Ruta al modelo de pose (opcional)

        Returns:
            Instancia única de PoseLandmarker
        """
        # Double-checked locking pattern para thread safety
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path)
                    logger.info("🔧 Nueva instancia PoseLandmarker creada")

        # Si se proporciona un model_path diferente, recargar modelo
        elif model_path is not None and cls._instance.model_path != model_path:
            with cls._lock:
                if cls._instance.model_path != model_path:
                    cls._instance.load_model(model_path)
                    logger.info(f"🔧 Modelo recargado: {model_path}")

        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        Reinicia la instancia singleton, liberando recursos.
        THREAD-SAFE - útil para evitar conflictos entre procesamiento de trabajos.
        """
        with cls._lock:
            if cls._instance is not None:
                logger.info("🧹 Reseteando instancia PoseLandmarker...")
                cls._instance.release_resources()
                cls._instance = None
                logger.info("✅ Instancia PoseLandmarker reseteada")

    def __init__(self, model_path=None):
        """
        Inicializa el PoseLandmarker, opcionalmente cargando un modelo.
        THREAD-SAFE constructor.

        Args:
            model_path (str, optional): Ruta al archivo del modelo.
        """
        self.model_path = model_path
        self.landmarker = None
        self._model_lock = (
            threading.Lock()
        )  # Lock específico para operaciones del modelo

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Carga el modelo de detección de poses desde la ruta especificada.
        THREAD-SAFE model loading.
        """
        with self._model_lock:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

            BaseOptions = mp_python.BaseOptions
            PoseLandmarker = mp_vision.PoseLandmarker
            PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
            VisionRunningMode = mp_vision.RunningMode

            try:
                # Liberar recursos previos si existe una instancia anterior
                if self.landmarker:
                    logger.debug("🧹 Liberando modelo anterior...")
                    self.release_resources()

                logger.info(f"📥 Cargando modelo desde: {model_path}")

                # Leer modelo en memoria
                with open(model_path, "rb") as f:
                    model_data = f.read()

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_buffer=model_data),
                    running_mode=VisionRunningMode.VIDEO,
                )

                self.landmarker = PoseLandmarker.create_from_options(options)
                self.model_path = model_path
                logger.info("✅ Modelo cargado exitosamente")
                return True

            except Exception as e:
                logger.error(f"❌ Error al cargar el modelo: {e}")
                self.landmarker = None
                raise

    def detect_pose(self, frame, timestamp_ms):
        """
        Detecta poses en un frame de video.
        THREAD-SAFE pose detection.
        """
        with self._model_lock:
            if self.landmarker is None:
                raise ValueError(
                    "El modelo no está cargado. Llame a load_model primero."
                )

            try:
                # Convertir a formato RGB para MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                # Detectar poses
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                return result

            except Exception as e:
                logger.error(f"❌ Error en detección de pose: {e}")
                raise

    def release_resources(self):
        """
        Libera recursos asociados con el modelo.
        THREAD-SAFE resource cleanup.
        """
        with self._model_lock:
            if self.landmarker:
                try:
                    # MediaPipe no tiene método close explícito, pero asignar None libera recursos
                    self.landmarker = None
                    logger.debug("🧹 Recursos del modelo liberados")
                except Exception as e:
                    logger.warning(f"⚠️ Error liberando recursos: {e}")

    def is_loaded(self):
        """
        Verifica si el modelo está cargado.
        THREAD-SAFE status check.
        """
        with self._model_lock:
            return self.landmarker is not None

    def get_model_info(self):
        """
        Obtiene información del modelo cargado.
        THREAD-SAFE info retrieval.
        """
        with self._model_lock:
            return {
                "model_path": self.model_path,
                "is_loaded": self.landmarker is not None,
                "instance_id": id(self),
            }

    def __del__(self):
        """
        Destructor de la clase, asegurando la liberación de recursos.
        THREAD-SAFE destructor.
        """
        try:
            self.release_resources()
        except Exception as e:
            # Ignorar errores en destructor para evitar problemas al cerrar
            pass

    def __repr__(self):
        """Representación string de la instancia."""
        return (
            f"PoseLandmarker(model_path='{self.model_path}', loaded={self.is_loaded()})"
        )
