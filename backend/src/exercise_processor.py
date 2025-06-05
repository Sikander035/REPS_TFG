import os
import sys
import logging
import pandas as pd
import traceback

# Añadir path raíz del proyecto para imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# CRÍTICO: Configurar matplotlib ANTES de cualquier import que lo use
try:
    from src.config.matplotlib_config import (
        configure_matplotlib_for_threading,
        cleanup_matplotlib_resources,
    )

    configure_matplotlib_for_threading()
except ImportError:
    # Si no existe matplotlib_config, seguimos sin configuración especial
    def cleanup_matplotlib_resources():
        pass

    def configure_matplotlib_for_threading():
        pass


# Imports del primer código
from src.core.data_extraction.data_extraction import extract_landmarks_from_video
from src.core.data_segmentation.detect_repetitions import detect_repetitions
from src.core.data_synchronization.synchronize_data import synchronize_data
from src.core.data_transformation.min_square_normalization import (
    normalize_skeletons_with_affine_method,
)
from src.core.data_transformation.align_data import align_skeletons_dataframe
from src.feedback.dual_body_visualization import generate_dual_skeleton_video
from src.config.config_manager import config_manager
from src.feedback.analysis_report import (
    run_exercise_analysis,
    generate_analysis_report,
)
from src.feedback.analysis_graphics import generate_radar_data
from src.feedback.analysis_llm import generate_trainer_feedback

logger = logging.getLogger(__name__)


class ExerciseProcessor:
    """
    Procesador principal de ejercicios con cleanup automático de singletons.
    Compatible con lógica y rutas del primer código.
    """

    def __init__(
        self,
        user_video_path,
        expert_csv_path,
        exercise_name,
        output_dir,
        config_path,
        model_path,
    ):
        self.user_video_path = user_video_path
        self.expert_csv_path = expert_csv_path
        self.exercise_name = exercise_name
        self.output_dir = output_dir
        self.config_path = config_path
        self.model_path = model_path

        # Cargar configuración (igual que en el primero)
        try:
            config_manager.load_config_file(config_path)
            logger.info("Configuración cargada correctamente")
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            raise

        # Crear carpetas de salida
        os.makedirs(output_dir, exist_ok=True)
        self.analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)

        # Inicializar variables de datos
        self.user_data = None
        self.expert_data = None
        self.user_repetitions = None
        self.expert_repetitions = None
        self.synchronized_user_data = None
        self.synchronized_expert_data = None
        self.normalized_expert_data = None
        self.aligned_expert_data = None
        self.analysis_results = None

        logger.info(f"🔧 ExerciseProcessor inicializado para {exercise_name}")

    def cleanup_singletons(self):
        """
        Limpia singletons para evitar conflictos entre ejecuciones.
        CRÍTICO para segundo procesamiento.
        """
        try:
            logger.info("🧹 Iniciando cleanup de singletons...")

            # 1. Limpiar matplotlib
            cleanup_matplotlib_resources()

            # 2. Resetear PoseLandmarker singleton (si existe)
            try:
                from src.core.data_extraction.pose_landmarker import PoseLandmarker

                PoseLandmarker.reset_instance()
                logger.debug("✅ PoseLandmarker singleton reseteado")
            except Exception as e:
                logger.warning(f"⚠️ Error reseteando PoseLandmarker: {e}")

            # 3. Limpiar cache del config_manager (si existe)
            try:
                config_manager.clear_cache()
                logger.debug("✅ ConfigManager cache limpiado")
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando ConfigManager: {e}")

            # 4. Forzar garbage collection
            import gc

            gc.collect()

            logger.info("✅ Cleanup de singletons completado")

        except Exception as e:
            logger.error(f"❌ Error en cleanup de singletons: {e}")

    def extract_landmarks_user_only(self):
        """Extract landmarks from user video only."""
        logger.info("📊 Extrayendo landmarks del video del usuario...")

        # Cleanup antes de extracción (especialmente importante en segunda ejecución)
        self.cleanup_singletons()

        user_csv_path = os.path.join(self.output_dir, "user_landmarks.csv")

        try:
            self.user_data = extract_landmarks_from_video(
                video_path=self.user_video_path,
                output_csv_path=user_csv_path,
                exercise=self.exercise_name,
                config_path=self.config_path,
                model_path=self.model_path,
                force_model_reload=True,  # Forzar recarga del modelo
            )

            if self.user_data is None or self.user_data.empty:
                raise ValueError(
                    "No se pudieron extraer landmarks del video del usuario"
                )

            logger.info(f"✅ Landmarks extraídos: {len(self.user_data)} frames")

        except Exception as e:
            logger.error(f"❌ Error extrayendo landmarks: {e}")
            raise

        finally:
            # Cleanup después de extracción
            cleanup_matplotlib_resources()

    def load_expert_data(self):
        """Load expert data from CSV."""
        logger.info("📂 Cargando datos del experto desde CSV...")

        try:
            if not os.path.exists(self.expert_csv_path):
                raise FileNotFoundError(
                    f"CSV del experto no encontrado: {self.expert_csv_path}"
                )

            self.expert_data = pd.read_csv(self.expert_csv_path)

            if self.expert_data.empty:
                raise ValueError("El CSV del experto está vacío")

            logger.info(
                f"✅ Datos del experto cargados: {len(self.expert_data)} frames"
            )

        except Exception as e:
            logger.error(f"❌ Error cargando datos del experto: {e}")
            raise

    def detect_repetitions(self):
        """Detect repetitions in both user and expert data."""
        logger.info("🔄 Detectando repeticiones...")

        try:
            # Detectar repeticiones del usuario
            self.user_repetitions = detect_repetitions(self.user_data)

            # Detectar repeticiones del experto
            self.expert_repetitions = detect_repetitions(self.expert_data)

            if not self.user_repetitions:
                raise ValueError("No se detectaron repeticiones en el usuario")
            if not self.expert_repetitions:
                raise ValueError("No se detectaron repeticiones en el experto")

            logger.info(
                f"✅ Repeticiones detectadas - Usuario: {len(self.user_repetitions)}, Experto: {len(self.expert_repetitions)}"
            )

        except Exception as e:
            logger.error(f"❌ Error detectando repeticiones: {e}")
            raise

    def synchronize_data(self):
        """Synchronize user and expert data."""
        logger.info("🔄 Sincronizando datos...")

        try:
            self.synchronized_user_data, self.synchronized_expert_data = (
                synchronize_data(
                    self.user_data,
                    self.expert_data,
                    exercise_name=self.exercise_name,
                    config_path=self.config_path,
                    user_repetitions=self.user_repetitions,
                    expert_repetitions=self.expert_repetitions,
                )
            )

            if (
                self.synchronized_user_data is None
                or self.synchronized_user_data.empty
                or self.synchronized_expert_data is None
                or self.synchronized_expert_data.empty
            ):
                raise ValueError("Los datos sincronizados están vacíos")

            logger.info(
                f"✅ Datos sincronizados: {len(self.synchronized_user_data)} frames"
            )

        except Exception as e:
            logger.error(f"❌ Error sincronizando datos: {e}")
            raise

    def normalize_skeletons(self):
        """Normalize expert skeleton to user proportions."""
        logger.info("📏 Normalizando esqueletos...")

        try:
            self.normalized_expert_data = normalize_skeletons_with_affine_method(
                self.synchronized_user_data, self.synchronized_expert_data
            )

            if self.normalized_expert_data is None or self.normalized_expert_data.empty:
                raise ValueError("Los datos normalizados están vacíos")

            logger.info(
                f"✅ Esqueletos normalizados: {len(self.normalized_expert_data)} frames"
            )

        except Exception as e:
            logger.error(f"❌ Error normalizando esqueletos: {e}")
            raise

    def align_skeletons(self):
        """Align normalized expert skeleton with user skeleton."""
        logger.info("🎯 Alineando esqueletos...")

        try:
            self.aligned_expert_data = align_skeletons_dataframe(
                self.synchronized_user_data, self.normalized_expert_data
            )

            if self.aligned_expert_data is None or self.aligned_expert_data.empty:
                raise ValueError("Los datos alineados están vacíos")

            logger.info(
                f"✅ Esqueletos alineados: {len(self.aligned_expert_data)} frames"
            )

        except Exception as e:
            logger.error(f"❌ Error alineando esqueletos: {e}")
            raise

    def run_analysis(self):
        """Run detailed technique analysis."""
        logger.info("📊 Ejecutando análisis detallado...")

        try:
            # Cleanup antes del análisis para evitar conflictos matplotlib
            cleanup_matplotlib_resources()

            self.analysis_results = run_exercise_analysis(
                user_data=self.synchronized_user_data,
                expert_data=self.aligned_expert_data,
                exercise_name=self.exercise_name,
                config_path=self.config_path,
                user_repetitions=self.user_repetitions,
                expert_repetitions=self.expert_repetitions,
            )

            if not self.analysis_results:
                raise ValueError("El análisis no produjo resultados")

            # Generar reporte JSON
            report_path = os.path.join(
                self.analysis_dir, f"{self.exercise_name}_report.json"
            )
            generate_analysis_report(
                analysis_results=self.analysis_results,
                exercise_name=self.exercise_name,
                output_path=report_path,
            )

            # Generar visualizaciones (thread-safe)
            generate_radar_data(
                analysis_results=self.analysis_results,
                exercise_name=self.exercise_name,
                output_dir=self.analysis_dir,
            )

            logger.info(
                f"✅ Análisis completado - Score: {self.analysis_results.get('score', 0):.1f}/100"
            )

        except Exception as e:
            logger.error(f"❌ Error en análisis: {e}")
            raise

        finally:
            # Cleanup final después del análisis
            cleanup_matplotlib_resources()

    def generate_video(self):
        """Generate comparison video."""
        logger.info("🎥 Generando video comparativo...")

        try:
            video_output_path = os.path.join(self.output_dir, "comparison_video.mp4")

            # Determinar rango de frames del ejercicio
            if self.user_repetitions:
                min_frame = min(rep["start_frame"] for rep in self.user_repetitions)
                max_frame = max(rep["end_frame"] for rep in self.user_repetitions)
                exercise_frame_range = (min_frame, max_frame)
            else:
                exercise_frame_range = None

            result_path = generate_dual_skeleton_video(
                original_video_path=self.user_video_path,
                user_data=self.synchronized_user_data,
                expert_data=self.aligned_expert_data,
                output_video_path=video_output_path,
                config_path=self.config_path,
                original_user_data=self.user_data,
                exercise_frame_range=exercise_frame_range,
            )

            logger.info(f"✅ Video generado: {result_path}")
            return result_path

        except Exception as e:
            logger.error(f"❌ Error generando video: {e}")
            raise

    def generate_feedback(self):
        """Generate personalized feedback using LLM."""
        logger.info("🤖 Generando feedback personalizado...")

        try:
            # Ruta al reporte JSON
            report_path = os.path.join(
                self.analysis_dir, f"{self.exercise_name}_report.json"
            )

            if not os.path.exists(report_path):
                raise FileNotFoundError(f"Reporte no encontrado: {report_path}")

            # Ruta de salida del feedback
            feedback_path = os.path.join(self.analysis_dir, "personalized_feedback.txt")

            # Generar feedback con DeepSeek
            api_key = os.getenv("DEEPSEEK_API_KEY")

            feedback = generate_trainer_feedback(
                informe_path=report_path,
                output_path=feedback_path,
                api_key=api_key,
            )

            logger.info(f"✅ Feedback generado: {feedback_path}")
            return feedback_path

        except Exception as e:
            logger.error(f"❌ Error generando feedback: {e}")
            # Generar feedback básico si falla
            try:
                fallback_feedback = f"""
🤖 ANÁLISIS AUTOMÁTICO DE TU {self.exercise_name.upper()}

Puntuación General: {self.analysis_results.get('score', 0):.1f}/100
Nivel: {self.analysis_results.get('level', 'No determinado')}

El análisis detallado se completó pero no se pudo generar feedback personalizado.
Revisa los gráficos y reporte JSON para obtener información específica sobre tu técnica.

Error técnico: {str(e)}
"""

                fallback_path = os.path.join(
                    self.analysis_dir, "personalized_feedback.txt"
                )
                with open(fallback_path, "w", encoding="utf-8") as f:
                    f.write(fallback_feedback)

                return fallback_path

            except Exception as fallback_error:
                logger.error(f"❌ Error en feedback de respaldo: {fallback_error}")
                raise

    def cleanup_all_resources(self):
        """
        Cleanup completo de todos los recursos al final del procesamiento.
        """
        logger.info("🧹 Iniciando cleanup final...")

        try:
            # Cleanup de singletons
            self.cleanup_singletons()

            # Limpiar referencias de datos grandes
            self.user_data = None
            self.expert_data = None
            self.synchronized_user_data = None
            self.synchronized_expert_data = None
            self.normalized_expert_data = None
            self.aligned_expert_data = None

            # Forzar garbage collection
            import gc

            gc.collect()

            logger.info("✅ Cleanup final completado")

        except Exception as e:
            logger.warning(f"⚠️ Error en cleanup final: {e}")

    def __del__(self):
        """Destructor para asegurar cleanup."""
        try:
            self.cleanup_all_resources()
        except:
            pass  # Ignorar errores en destructor
