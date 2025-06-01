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
        CORREGIDO: Incluye TODOS los campos del ejercicio, no solo landmarks y sync_config.

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

        # Obtener la configuración completa del ejercicio
        exercise_config = config_data[exercise_name]

        # CORREGIDO: Validar campos obligatorios básicos
        self._validate_basic_exercise_config(exercise_config, exercise_name)

        # CORREGIDO: Validar analysis_config si existe
        if "analysis_config" in exercise_config:
            self._validate_analysis_config(
                exercise_config["analysis_config"], exercise_name
            )

        # CORREGIDO: Retornar TODA la configuración del ejercicio, no solo campos específicos
        result = exercise_config.copy()  # Copia completa de todos los campos

        # Guardar en atributos internos y retornar
        self._exercise_configs[config_key] = result
        return result

    def get_penalty_config(self, exercise_name, metric_type, metric_name, config_path):
        """
        Obtiene la configuración de penalty para una métrica específica.

        Args:
            exercise_name: Nombre del ejercicio ("press_militar", "sentadilla", etc.)
            metric_type: Tipo de métrica ("universal" o "specific")
            metric_name: Nombre específico de la métrica ("amplitude", "elbow_abduction", etc.)
            config_path: Ruta al archivo de configuración

        Returns:
            int: Valor de penalty configurado o valor por defecto
        """
        # Cargar archivo si es necesario
        if config_path not in self._loaded_files:
            self.load_config_file(config_path)

        config_data = self._loaded_files[config_path]

        # Valores por defecto si no hay configuración
        default_penalties = {
            "universal": {
                "amplitude": 40,
                "symmetry": 30,
                "trajectory": 30,
                "speed": 25,
            },
            "specific": {
                "elbow_abduction": 25,
                "scapular_stability": 35,
                "squat_depth": 40,
                "knee_tracking": 30,
                "swing_control": 35,
                "scapular_retraction": 30,
            },
        }

        try:
            # Buscar en penalty_config
            penalty_config = config_data.get("penalty_config", {})

            if metric_type == "universal":
                # Para métricas universales
                universal_penalties = penalty_config.get("universal_metrics", {})
                penalty = universal_penalties.get(metric_name)

                if penalty is not None:
                    logger.debug(f"Penalty para {metric_name} (universal): {penalty}")
                    return penalty
                else:
                    # Fallback a valor por defecto
                    default_penalty = default_penalties["universal"].get(
                        metric_name, 30
                    )
                    logger.warning(
                        f"Penalty no configurado para {metric_name} (universal). Usando default: {default_penalty}"
                    )
                    return default_penalty

            elif metric_type == "specific":
                # Para métricas específicas
                specific_penalties = penalty_config.get("specific_metrics", {})
                exercise_penalties = specific_penalties.get(exercise_name, {})
                penalty = exercise_penalties.get(metric_name)

                if penalty is not None:
                    logger.debug(
                        f"Penalty para {metric_name} ({exercise_name}): {penalty}"
                    )
                    return penalty
                else:
                    # Fallback a valor por defecto
                    default_penalty = default_penalties["specific"].get(metric_name, 30)
                    logger.warning(
                        f"Penalty no configurado para {metric_name} ({exercise_name}). Usando default: {default_penalty}"
                    )
                    return default_penalty
            else:
                logger.error(
                    f"Tipo de métrica inválido: {metric_type}. Debe ser 'universal' o 'specific'"
                )
                return 30  # Valor por defecto genérico

        except Exception as e:
            logger.error(f"Error obteniendo penalty config: {e}")
            # Retornar valor por defecto según el tipo
            if metric_type == "universal":
                return default_penalties["universal"].get(metric_name, 30)
            else:
                return default_penalties["specific"].get(metric_name, 30)

    def _validate_basic_exercise_config(self, exercise_config, exercise_name):
        """
        Valida que la configuración básica del ejercicio sea correcta.

        Args:
            exercise_config: Configuración del ejercicio
            exercise_name: Nombre del ejercicio para mensajes de error

        Raises:
            ValueError: Si faltan campos obligatorios
        """
        # Verificar campos obligatorios básicos
        if "landmarks" not in exercise_config:
            raise ValueError(
                f"Configuración de '{exercise_name}' no contiene landmarks"
            )

        if "sync_config" not in exercise_config:
            raise ValueError(
                f"Configuración de '{exercise_name}' no contiene sync_config"
            )

        # Verificar que landmarks es una lista no vacía
        landmarks = exercise_config["landmarks"]
        if not isinstance(landmarks, list) or len(landmarks) == 0:
            raise ValueError(
                f"Configuración de '{exercise_name}': landmarks debe ser una lista no vacía"
            )

        # Verificar que sync_config es un diccionario
        sync_config = exercise_config["sync_config"]
        if not isinstance(sync_config, dict):
            raise ValueError(
                f"Configuración de '{exercise_name}': sync_config debe ser un diccionario"
            )

        # Verificar campos obligatorios en sync_config
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

        missing_fields = [f for f in required_sync_fields if f not in sync_config]
        if missing_fields:
            raise ValueError(
                f"Configuración de '{exercise_name}' incompleta. "
                f"Faltan campos en sync_config: {', '.join(missing_fields)}"
            )

        logger.debug(
            f"Configuración básica de '{exercise_name}' validada correctamente"
        )

    def _validate_analysis_config(self, analysis_config, exercise_name):
        """
        Valida que analysis_config tenga la estructura correcta.

        Args:
            analysis_config: Configuración de análisis
            exercise_name: Nombre del ejercicio para mensajes de error

        Raises:
            ValueError: Si analysis_config tiene estructura incorrecta
        """
        if not isinstance(analysis_config, dict):
            raise ValueError(
                f"Configuración de '{exercise_name}': analysis_config debe ser un diccionario"
            )

        # Verificar sensitivity_factors si existe
        if "sensitivity_factors" in analysis_config:
            sensitivity_factors = analysis_config["sensitivity_factors"]

            if not isinstance(sensitivity_factors, dict):
                raise ValueError(
                    f"Configuración de '{exercise_name}': sensitivity_factors debe ser un diccionario"
                )

            # Verificar que los valores sean numéricos
            for factor_name, factor_value in sensitivity_factors.items():
                if not isinstance(factor_value, (int, float)):
                    raise ValueError(
                        f"Configuración de '{exercise_name}': sensitivity_factors['{factor_name}'] "
                        f"debe ser numérico, recibido: {type(factor_value)}"
                    )

                if factor_value <= 0:
                    raise ValueError(
                        f"Configuración de '{exercise_name}': sensitivity_factors['{factor_name}'] "
                        f"debe ser positivo, recibido: {factor_value}"
                    )

            logger.debug(
                f"sensitivity_factors de '{exercise_name}' validado: {len(sensitivity_factors)} factores"
            )

        # Verificar otros campos de análisis si existen
        valid_analysis_fields = [
            "sensitivity_factors",
            "min_elbow_angle",
            "max_elbow_angle",
            "rom_threshold",
            "bottom_diff_threshold",
            "abduction_angle_threshold",
            "symmetry_threshold",
            "lateral_dev_threshold",
            "frontal_dev_threshold",
            "velocity_ratio_threshold",
            "scapular_stability_threshold",
            "scoring_weights",
        ]

        # Advertir sobre campos desconocidos (no es error, solo advertencia)
        unknown_fields = [
            f for f in analysis_config.keys() if f not in valid_analysis_fields
        ]
        if unknown_fields:
            logger.warning(
                f"Configuración de '{exercise_name}': campos desconocidos en analysis_config: {unknown_fields}"
            )

        logger.debug(f"analysis_config de '{exercise_name}' validado correctamente")

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
