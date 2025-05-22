# config_manager.py
import json
import os
import logging
import sys

# Configurar rutas absolutas para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Gestor de configuración implementado como un Singleton real.
    Carga archivos de configuración una sola vez y mantiene
    configuraciones persistentes como atributos.
    """

    # Instancia única (singleton)
    _instance = None

    def __new__(cls, config_path=None):
        """Crea o devuelve la instancia singleton."""
        if cls._instance is None:
            logger.debug("Creando nueva instancia de ConfigManager")
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        """Inicializa el ConfigManager solo una vez."""
        # Evitar inicialización múltiple
        if self._initialized:
            return

        logger.info(f"Inicializando ConfigManager con configuración: {config_path}")
        self._initialized = True

        # Atributos de estado interno
        self._loaded_files = {}  # Archivos completos cargados
        self._exercise_configs = {}  # Configuraciones de ejercicios procesadas
        self._landmark_mapping = self._init_landmark_mapping()

        # Cargar configuración inicial si se proporciona
        if config_path:
            self.load_config_file(config_path)

    def _init_landmark_mapping(self):
        """Inicializa el mapeo de landmarks (solo una vez)."""
        return {
            0: "nose",
            1: "left_eye_inner",
            2: "left_eye",
            3: "left_eye_outer",
            4: "right_eye_inner",
            5: "right_eye",
            6: "right_eye_outer",
            7: "left_ear",
            8: "right_ear",
            9: "mouth_left",
            10: "mouth_right",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            17: "left_pinky",
            18: "right_pinky",
            19: "left_index",
            20: "right_index",
            21: "left_thumb",
            22: "right_thumb",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
            29: "left_heel",
            30: "right_heel",
            31: "left_foot_index",
            32: "right_foot_index",
        }

    def load_config_file(self, config_path):
        """
        Carga un archivo de configuración en memoria.

        Args:
            config_path: Ruta al archivo de configuración

        Returns:
            bool: True si se cargó correctamente, False en caso contrario

        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        # Verificar si ya se cargó este archivo
        if config_path in self._loaded_files:
            logger.debug(f"Archivo ya cargado: {config_path}")
            return True

        # Verificar existencia del archivo
        if not os.path.exists(config_path):
            alt_path = os.path.join(os.path.dirname(__file__), config_path)
            if not os.path.exists(alt_path):
                raise FileNotFoundError(
                    f"Archivo de configuración no encontrado en '{config_path}' ni en '{alt_path}'"
                )
            config_path = alt_path

        # Cargar el archivo
        try:
            logger.info(f"Cargando archivo de configuración: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Almacenar el archivo cargado
            self._loaded_files[config_path] = config_data
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Error al decodificar JSON desde {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar configuración desde {config_path}: {e}")
            raise

    def get_exercise_config(self, exercise_name, config_path):
        """
        Obtiene la configuración de un ejercicio específico.
        La carga solo una vez y la mantiene en memoria.

        Args:
            exercise_name: Nombre del ejercicio
            config_path: Ruta al archivo de configuración

        Returns:
            dict: Configuración completa del ejercicio

        Raises:
            ValueError: Si el ejercicio no existe en la configuración
        """
        # Normalizar nombre de ejercicio
        exercise_name = exercise_name.strip().lower().replace(" ", "_")

        # Clave única para cada combinación de ejercicio y archivo
        config_key = f"{exercise_name}:{config_path}"

        # Verificar si ya tenemos esta configuración
        if config_key in self._exercise_configs:
            return self._exercise_configs[config_key]

        # Cargar archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        # Obtener datos del archivo
        config_data = self._loaded_files[config_path]

        # Verificar si el ejercicio existe
        if exercise_name not in config_data:
            available = ", ".join(config_data.keys())
            raise ValueError(
                f"Ejercicio '{exercise_name}' no encontrado. Ejercicios disponibles: {available}"
            )

        # Obtener y procesar la configuración del ejercicio
        exercise_config = config_data[exercise_name]

        # Verificar campos obligatorios
        if "landmarks" not in exercise_config:
            raise ValueError(
                f"Configuración de '{exercise_name}' no contiene landmarks"
            )

        if "sync_config" not in exercise_config:
            raise ValueError(
                f"Configuración de '{exercise_name}' no contiene sync_config"
            )

        # Verificar campos en sync_config
        sync_config = exercise_config["sync_config"]
        required_fields = [
            "landmarks",
            "num_divisions",
            "interp_method",
            "division_strategy",
            "matching_strategy",
            "phase_strategy",
            "exercise_type",
            "division_landmarks",
            "division_axis",
            "adapt_direction",
        ]

        missing_fields = [f for f in required_fields if f not in sync_config]
        if missing_fields:
            raise ValueError(
                f"Configuración de '{exercise_name}' incompleta. "
                f"Faltan campos: {', '.join(missing_fields)}"
            )

        # Crear resultado final
        result = {
            "landmarks": exercise_config["landmarks"],
            "sync_config": exercise_config["sync_config"],
        }

        # Guardar en atributos internos y retornar
        self._exercise_configs[config_key] = result
        return result

    def get_landmark_mapping(self):
        """Devuelve el mapeo de landmarks."""
        return self._landmark_mapping

    def convert_landmark_indices_to_names(self, landmark_indices):
        """Convierte índices de landmarks a nombres."""
        return [
            f"landmark_{self._landmark_mapping[idx]}"
            for idx in landmark_indices
            if idx in self._landmark_mapping
        ]

    def get_available_exercises(self, config_path):
        """
        Devuelve la lista de ejercicios disponibles en un archivo de configuración.

        Args:
            config_path: Ruta al archivo de configuración

        Returns:
            list: Lista de nombres de ejercicios disponibles
        """
        # Cargar el archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        # Devolver las claves (nombres de ejercicios)
        return list(self._loaded_files[config_path].keys())

    def get_global_visualization_config(self, config_path):
        """
        Obtiene la configuración global de visualización.

        Args:
            config_path: Ruta al archivo de configuración

        Returns:
            dict: Configuración de visualización global

        Raises:
            ValueError: Si no se encuentra la configuración requerida
        """
        # Cargar el archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        config_data = self._loaded_files[config_path]

        # Verificar que existe la configuración global de visualización
        if "global_visualization" not in config_data:
            raise ValueError(
                "Configuración 'global_visualization' no encontrada en el archivo"
            )

        viz_config = config_data["global_visualization"]

        # Verificar campos obligatorios
        required_fields = [
            "user_color",
            "expert_color",
            "user_alpha",
            "expert_alpha",
            "user_thickness",
            "expert_thickness",
            "show_labels",
            "show_progress",
            "text_info",
            "resize_factor",
        ]

        missing_fields = [field for field in required_fields if field not in viz_config]
        if missing_fields:
            raise ValueError(
                f"Campos obligatorios faltantes en 'global_visualization': {', '.join(missing_fields)}"
            )

        return viz_config

    def get_global_connections(self, config_path):
        """
        Obtiene las conexiones globales para dibujar esqueletos.

        Args:
            config_path: Ruta al archivo de configuración

        Returns:
            list: Lista de tuplas con conexiones entre landmarks

        Raises:
            ValueError: Si no se encuentra la configuración requerida
        """
        # Cargar el archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        config_data = self._loaded_files[config_path]

        # Verificar que existe la configuración de conexiones globales
        if "global_connections" not in config_data:
            raise ValueError(
                "Configuración 'global_connections' no encontrada en el archivo"
            )

        connections_data = config_data["global_connections"]

        # Verificar que es una lista y no está vacía
        if not isinstance(connections_data, list) or len(connections_data) == 0:
            raise ValueError("'global_connections' debe ser una lista no vacía")

        # Verificar que cada conexión tiene exactamente 2 elementos
        for i, conn in enumerate(connections_data):
            if not isinstance(conn, list) or len(conn) != 2:
                raise ValueError(
                    f"Conexión {i} debe ser una lista de exactamente 2 elementos"
                )
            if not all(isinstance(item, str) for item in conn):
                raise ValueError(f"Conexión {i} debe contener solo strings")

        # Convertir listas a tuplas
        return [tuple(conn) for conn in connections_data]

    def clear_cache(self):
        """Limpia todas las configuraciones cargadas (útil para pruebas)."""
        self._loaded_files.clear()
        self._exercise_configs.clear()
        logger.info("Caché de configuraciones limpiada")


# Instancia global para facilitar el uso
config_manager = ConfigManager()


# Funciones de compatibilidad para código existente
def load_exercise_config(exercise_name, config_path):
    """Función de compatibilidad que utiliza la instancia singleton."""
    return config_manager.get_exercise_config(exercise_name, config_path)


def get_landmark_mapping():
    """Función de compatibilidad para mapeo de landmarks."""
    return config_manager.get_landmark_mapping()


def convert_landmark_indices_to_names(landmark_indices):
    """Función de compatibilidad para convertir índices a nombres."""
    return config_manager.convert_landmark_indices_to_names(landmark_indices)
