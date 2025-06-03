# config_manager.py - THREAD-SAFE VERSION
import json
import os
import logging
import sys
import threading

# Configurar rutas absolutas para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Gestor de configuración implementado como un Singleton real THREAD-SAFE.
    Carga archivos de configuración una sola vez y mantiene
    configuraciones persistentes como atributos.
    """

    # Instancia única (singleton)
    _instance = None
    _lock = threading.Lock()  # Lock para thread safety

    def __new__(cls, config_path=None):
        """Crea o devuelve la instancia singleton de forma thread-safe."""
        # Double-checked locking pattern
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.debug("Creating new ConfigManager instance (thread-safe)")
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        """Inicializa el ConfigManager solo una vez de forma thread-safe."""
        # Evitar inicialización múltiple
        if self._initialized:
            return

        logger.info(f"Initializing ConfigManager with configuration: {config_path}")

        with self._lock:
            if self._initialized:
                return

            self._initialized = True

            # Atributos de estado interno con locks individuales
            self._loaded_files = {}  # Archivos completos cargados
            self._exercise_configs = {}  # Configuraciones de ejercicios procesadas
            self._landmark_mapping = self._init_landmark_mapping()

            # Lock para operaciones de archivos
            self._files_lock = threading.Lock()
            self._configs_lock = threading.Lock()

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
        Carga un archivo de configuración en memoria de forma thread-safe.

        Args:
            config_path: Ruta al archivo de configuración

        Returns:
            bool: True si se cargó correctamente

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo tiene formato inválido
        """
        with self._files_lock:
            # Verificar si ya se cargó este archivo
            if config_path in self._loaded_files:
                logger.debug(f"File already loaded: {config_path}")
                return True

            # Verificar existencia del archivo
            if not os.path.exists(config_path):
                alt_path = os.path.join(os.path.dirname(__file__), config_path)
                if not os.path.exists(alt_path):
                    raise FileNotFoundError(
                        f"Configuration file not found at '{config_path}' or '{alt_path}'"
                    )
                config_path = alt_path

            # Cargar el archivo
            try:
                logger.info(f"Loading configuration file: {config_path}")
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                # Almacenar el archivo cargado
                self._loaded_files[config_path] = config_data
                return True

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in configuration file {config_path}: {e}"
                )
            except Exception as e:
                raise ValueError(f"Error loading configuration from {config_path}: {e}")

    def get_exercise_config(self, exercise_name, config_path):
        """
        Obtiene la configuración de un ejercicio específico de forma thread-safe.

        Args:
            exercise_name: Nombre del ejercicio
            config_path: Ruta al archivo de configuración

        Returns:
            dict: Configuración completa del ejercicio

        Raises:
            ValueError: Si el ejercicio no existe o la configuración está incompleta
        """
        # Normalizar nombre de ejercicio
        exercise_name = exercise_name.strip().lower().replace(" ", "_")

        # Clave única para cada combinación de ejercicio y archivo
        config_key = f"{exercise_name}:{config_path}"

        with self._configs_lock:
            # Verificar si ya tenemos esta configuración
            if config_key in self._exercise_configs:
                return self._exercise_configs[config_key]

        # Cargar archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            # Obtener datos del archivo
            config_data = self._loaded_files[config_path]

        # Verificar si el ejercicio existe
        if exercise_name not in config_data:
            available = ", ".join(
                [
                    k
                    for k in config_data.keys()
                    if not k.startswith("global_")
                    and k != "penalty_config"
                    and k != "skill_levels"
                    and k != "scoring_weights"
                    and k != "exercise_landmarks_config"
                ]
            )
            raise ValueError(
                f"Exercise '{exercise_name}' not found in configuration. Available exercises: {available}"
            )

        # Obtener la configuración completa del ejercicio
        exercise_config = config_data[exercise_name]

        # Validar configuración completa
        self._validate_complete_exercise_config(exercise_config, exercise_name)

        with self._configs_lock:
            # Guardar en atributos internos y retornar
            self._exercise_configs[config_key] = exercise_config

        return exercise_config

    def get_exercise_landmarks_config(self, exercise_name, config_path):
        """
        Obtiene la configuración de landmarks para un ejercicio específico de forma thread-safe.
        """
        exercise_name = exercise_name.strip().lower().replace(" ", "_")

        # Cargar archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            config_data = self._loaded_files[config_path]

        # Verificar que existe exercise_landmarks_config
        if "exercise_landmarks_config" not in config_data:
            raise ValueError(
                "'exercise_landmarks_config' section not found in configuration file"
            )

        landmarks_config = config_data["exercise_landmarks_config"]

        # Verificar que existe el ejercicio
        if exercise_name not in landmarks_config:
            available = ", ".join(landmarks_config.keys())
            raise ValueError(
                f"Landmarks configuration for exercise '{exercise_name}' not found. Available: {available}"
            )

        return landmarks_config[exercise_name]

    def get_penalty_config(self, exercise_name, metric_type, metric_name, config_path):
        """
        Obtiene la configuración de penalty para una métrica específica de forma thread-safe.
        """
        # Cargar archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            config_data = self._loaded_files[config_path]

        # Verificar que existe penalty_config
        if "penalty_config" not in config_data:
            raise ValueError("'penalty_config' section not found in configuration file")

        penalty_config = config_data["penalty_config"]

        if metric_type == "universal":
            if "universal_metrics" not in penalty_config:
                raise ValueError(
                    "'penalty_config.universal_metrics' not found in configuration"
                )

            universal_penalties = penalty_config["universal_metrics"]
            if metric_name not in universal_penalties:
                available = ", ".join(universal_penalties.keys())
                raise ValueError(
                    f"Penalty for universal metric '{metric_name}' not found. Available: {available}"
                )

            penalty = universal_penalties[metric_name]
            logger.debug(f"Penalty for {metric_name} (universal): {penalty}")
            return penalty

        elif metric_type == "specific":
            if "specific_metrics" not in penalty_config:
                raise ValueError(
                    "'penalty_config.specific_metrics' not found in configuration"
                )

            specific_penalties = penalty_config["specific_metrics"]

            if exercise_name not in specific_penalties:
                available = ", ".join(specific_penalties.keys())
                raise ValueError(
                    f"Penalty configuration for exercise '{exercise_name}' not found. Available: {available}"
                )

            exercise_penalties = specific_penalties[exercise_name]
            if metric_name not in exercise_penalties:
                available = ", ".join(exercise_penalties.keys())
                raise ValueError(
                    f"Penalty for specific metric '{metric_name}' in exercise '{exercise_name}' not found. Available: {available}"
                )

            penalty = exercise_penalties[metric_name]
            logger.debug(f"Penalty for {metric_name} ({exercise_name}): {penalty}")
            return penalty

        else:
            raise ValueError(
                f"Invalid metric_type: {metric_type}. Must be 'universal' or 'specific'"
            )

    def get_sensitivity_factor(self, metric_name, exercise_name, config_path):
        """
        Obtiene el factor de sensibilidad para una métrica específica de forma thread-safe.
        """
        exercise_config = self.get_exercise_config(exercise_name, config_path)

        if "analysis_config" not in exercise_config:
            raise ValueError(
                f"'analysis_config' not found for exercise '{exercise_name}'"
            )

        analysis_config = exercise_config["analysis_config"]

        if "sensitivity_factors" not in analysis_config:
            raise ValueError(
                f"'sensitivity_factors' not found in analysis_config for exercise '{exercise_name}'"
            )

        sensitivity_factors = analysis_config["sensitivity_factors"]

        if metric_name not in sensitivity_factors:
            available = ", ".join(sensitivity_factors.keys())
            raise ValueError(
                f"Sensitivity factor for metric '{metric_name}' not found in exercise '{exercise_name}'. Available: {available}"
            )

        factor = sensitivity_factors[metric_name]
        if not isinstance(factor, (int, float)) or factor <= 0:
            raise ValueError(
                f"Invalid sensitivity factor for '{metric_name}': {factor}. Must be a positive number"
            )

        return factor

    def get_scoring_weights(self, exercise_name, config_path):
        """
        Obtiene los pesos de scoring para un ejercicio de forma thread-safe.
        """
        exercise_config = self.get_exercise_config(exercise_name, config_path)

        # Prioridad: pesos específicos del ejercicio
        if "scoring_weights" in exercise_config:
            weights = exercise_config["scoring_weights"]
            if not isinstance(weights, dict) or not weights:
                raise ValueError(
                    f"Invalid 'scoring_weights' for exercise '{exercise_name}': must be a non-empty dict"
                )
            return weights

        # Fallback: pesos globales
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            config_data = self._loaded_files[config_path]

        if "scoring_weights" in config_data:
            weights = config_data["scoring_weights"]
            if not isinstance(weights, dict) or not weights:
                raise ValueError(
                    "Invalid global 'scoring_weights': must be a non-empty dict"
                )
            return weights

        raise ValueError(
            f"No scoring weights found for exercise '{exercise_name}' or globally"
        )

    def get_analysis_threshold(self, threshold_name, exercise_name, config_path):
        """
        Obtiene un umbral específico de análisis de forma thread-safe.
        """
        exercise_config = self.get_exercise_config(exercise_name, config_path)

        # Buscar en analysis_config del ejercicio
        if (
            "analysis_config" in exercise_config
            and threshold_name in exercise_config["analysis_config"]
        ):
            threshold = exercise_config["analysis_config"][threshold_name]
            if not isinstance(threshold, (int, float)):
                raise ValueError(
                    f"Invalid threshold '{threshold_name}' for exercise '{exercise_name}': {threshold}"
                )
            return threshold

        # Buscar en configuración global
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            config_data = self._loaded_files[config_path]

        if (
            "global_analysis_config" in config_data
            and threshold_name in config_data["global_analysis_config"]
        ):
            threshold = config_data["global_analysis_config"][threshold_name]
            if not isinstance(threshold, (int, float)):
                raise ValueError(
                    f"Invalid global threshold '{threshold_name}': {threshold}"
                )
            return threshold

        raise ValueError(
            f"Threshold '{threshold_name}' not found for exercise '{exercise_name}' or globally"
        )

    def get_global_visualization_config(self, config_path):
        """
        Obtiene la configuración global de visualización de forma thread-safe.
        """
        # Cargar el archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            config_data = self._loaded_files[config_path]

        # Verificar que existe la configuración global de visualización
        if "global_visualization" not in config_data:
            raise ValueError("'global_visualization' not found in configuration file")

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
                f"Missing required fields in 'global_visualization': {', '.join(missing_fields)}"
            )

        return viz_config

    def get_global_connections(self, config_path):
        """
        Obtiene las conexiones globales para dibujar esqueletos de forma thread-safe.
        """
        # Cargar el archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            config_data = self._loaded_files[config_path]

        # Verificar que existe la configuración de conexiones globales
        if "global_connections" not in config_data:
            raise ValueError("'global_connections' not found in configuration file")

        connections_data = config_data["global_connections"]

        # Verificar que es una lista y no está vacía
        if not isinstance(connections_data, list) or len(connections_data) == 0:
            raise ValueError("'global_connections' must be a non-empty list")

        # Verificar que cada conexión tiene exactamente 2 elementos
        for i, conn in enumerate(connections_data):
            if not isinstance(conn, list) or len(conn) != 2:
                raise ValueError(f"Connection {i} must be a list of exactly 2 elements")
            if not all(isinstance(item, str) for item in conn):
                raise ValueError(f"Connection {i} must contain only strings")

        # Convertir listas a tuplas
        return [tuple(conn) for conn in connections_data]

    def get_available_exercises(self, config_path):
        """
        Devuelve la lista de ejercicios disponibles en un archivo de configuración de forma thread-safe.
        """
        # Cargar el archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        with self._files_lock:
            # Filtrar solo ejercicios (excluir configuraciones globales)
            config_data = self._loaded_files[config_path]

        excluded_keys = {
            "global_visualization",
            "global_connections",
            "global_analysis_config",
            "penalty_config",
            "scoring_weights",
            "skill_levels",
            "exercise_landmarks_config",
        }

        return [k for k in config_data.keys() if k not in excluded_keys]

    def clear_cache(self):
        """Limpia todas las configuraciones cargadas de forma thread-safe (útil para pruebas)."""
        with self._files_lock:
            self._loaded_files.clear()

        with self._configs_lock:
            self._exercise_configs.clear()

        logger.info("🧹 Configuration cache cleared (thread-safe)")

    def _validate_complete_exercise_config(self, exercise_config, exercise_name):
        """
        Valida que la configuración del ejercicio esté completa y sea válida.
        """
        # Campos obligatorios de primer nivel
        required_fields = [
            "landmarks",
            "sync_config",
            "analysis_config",
            "scoring_weights",
        ]

        for field in required_fields:
            if field not in exercise_config:
                raise ValueError(
                    f"Missing required field '{field}' in exercise '{exercise_name}'"
                )

        # Validar landmarks
        landmarks = exercise_config["landmarks"]
        if not isinstance(landmarks, list) or len(landmarks) == 0:
            raise ValueError(
                f"'landmarks' for exercise '{exercise_name}' must be a non-empty list"
            )

        # Validar sync_config
        self._validate_sync_config(exercise_config["sync_config"], exercise_name)

        # Validar analysis_config
        self._validate_analysis_config(
            exercise_config["analysis_config"], exercise_name
        )

        # Validar scoring_weights
        self._validate_scoring_weights(
            exercise_config["scoring_weights"], exercise_name
        )

    def _validate_sync_config(self, sync_config, exercise_name):
        """Valida que sync_config esté completo."""
        if not isinstance(sync_config, dict):
            raise ValueError(
                f"'sync_config' for exercise '{exercise_name}' must be a dict"
            )

        required_sync_fields = [
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

        for field in required_sync_fields:
            if field not in sync_config:
                raise ValueError(
                    f"Missing required field 'sync_config.{field}' in exercise '{exercise_name}'"
                )

    def _validate_analysis_config(self, analysis_config, exercise_name):
        """Valida que analysis_config esté completo."""
        if not isinstance(analysis_config, dict):
            raise ValueError(
                f"'analysis_config' for exercise '{exercise_name}' must be a dict"
            )

        # sensitivity_factors es obligatorio
        if "sensitivity_factors" not in analysis_config:
            raise ValueError(
                f"Missing required field 'analysis_config.sensitivity_factors' in exercise '{exercise_name}'"
            )

        sensitivity_factors = analysis_config["sensitivity_factors"]
        if not isinstance(sensitivity_factors, dict) or not sensitivity_factors:
            raise ValueError(
                f"'sensitivity_factors' for exercise '{exercise_name}' must be a non-empty dict"
            )

        # Validar que todos los valores sean numéricos positivos
        for factor_name, factor_value in sensitivity_factors.items():
            if not isinstance(factor_value, (int, float)) or factor_value <= 0:
                raise ValueError(
                    f"Invalid sensitivity factor '{factor_name}' for exercise '{exercise_name}': {factor_value}. Must be a positive number"
                )

    def _validate_scoring_weights(self, scoring_weights, exercise_name):
        """Valida que scoring_weights esté completo."""
        if not isinstance(scoring_weights, dict) or not scoring_weights:
            raise ValueError(
                f"'scoring_weights' for exercise '{exercise_name}' must be a non-empty dict"
            )

        # Verificar que todos los valores sean numéricos
        for weight_name, weight_value in scoring_weights.items():
            if not isinstance(weight_value, (int, float)) or weight_value < 0:
                raise ValueError(
                    f"Invalid scoring weight '{weight_name}' for exercise '{exercise_name}': {weight_value}. Must be a non-negative number"
                )

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
