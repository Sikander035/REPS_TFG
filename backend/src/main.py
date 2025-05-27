import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging
import time
import json
from pathlib import Path
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ruta absoluta base del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEOS_DIR = os.path.join(BASE_DIR, "media", "videos")
CONFIG_PATH = os.path.join(BASE_DIR, "src", "config", "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

# Leer el nombre del ejercicio desde config.json
nombre_ejercicio = "press_militar"
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output", f"resultados_{nombre_ejercicio}")

# Añadir la ruta al directorio raíz del proyecto (backend)
sys.path.append(BASE_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.core.data_extraction.data_extraction import extract_landmarks_from_video
    from src.core.data_segmentation.detect_repetitions import detect_repetitions
    from src.core.data_synchronization.synchronize_data import (
        synchronize_data,
    )
    from src.core.data_transformation.min_square_normalization import (
        normalize_skeletons_with_affine_method,
    )
    from src.core.data_transformation.align_data import align_skeletons_dataframe
    from src.feedback.dual_body_visualization import (
        generate_dual_skeleton_video,
        visualize_frame_dual_skeletons,
    )

    # CORREGIDO: Usar solo el singleton, no la función duplicada
    from src.config.config_manager import config_manager
    from src.feedback.analysis_report import (
        run_exercise_analysis,
        generate_analysis_report,
    )
    from src.feedback.analysis_graphics import visualize_analysis_results

    # CORREGIDO: Cargar configuración usando el singleton una sola vez al inicio
    config_manager.load_config_file(CONFIG_PATH)
    logger.info("Configuración cargada usando config_manager singleton")

except ImportError as e:
    logger.warning(f"No se encontraron todos los módulos necesarios: {e}")
    logger.warning("El procesamiento puede fallar")


def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def process_exercise(
    exercise_name,
    videos_dir,
    output_dir,
    config_path,
    user_video=None,
    expert_video=None,
    skip_extraction=False,
    skip_normalization=False,
    skip_visualization=False,
    skip_analysis=False,
    diagnostics=False,
    model_path=None,
):
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    diagnostics_dir = (
        os.path.join(output_dir, "diagnostics", exercise_name, run_id)
        if diagnostics
        else None
    )
    if diagnostics_dir:
        os.makedirs(diagnostics_dir, exist_ok=True)
        logger.info(f"Generando diagnósticos en: {diagnostics_dir}")

    try:
        # CORREGIDO: Usar el singleton correctamente
        exercise_config = config_manager.get_exercise_config(exercise_name, config_path)
        logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
    except Exception as e:
        logger.error(f"Error al cargar configuración: {e}")
        exercise_config = {"sync_config": {}}

    if diagnostics_dir:
        with open(os.path.join(diagnostics_dir, "config.json"), "w") as f:
            json.dump(exercise_config, f, indent=2)

    if user_video is None:
        user_video = os.path.join(videos_dir, f"{exercise_name}_Usuario.mp4")
    if expert_video is None:
        expert_video = os.path.join(videos_dir, f"{exercise_name}_Experto.mp4")
    else:
        expert_video = (
            os.path.join(videos_dir, expert_video)
            if not os.path.isabs(expert_video)
            else expert_video
        )

    video_usuario = user_video
    video_experto = expert_video
    csv_usuario = os.path.join(output_dir, f"{exercise_name}_Usuario.csv")
    csv_experto = os.path.join(output_dir, f"{exercise_name}_Experto.csv")

    logger.info(f"USANDO VIDEO USUARIO: {video_usuario}")
    logger.info(f"USANDO VIDEO EXPERTO: {video_experto}")
    logger.info(f"USANDO MODELO: {model_path}")
    if not os.path.exists(video_usuario):
        logger.error(f"No se encontró el video del usuario: {video_usuario}")
        return None
    if not os.path.exists(video_experto):
        logger.error(f"No se encontró el video del experto: {video_experto}")
        return None

    results = {
        "exercise_name": exercise_name,
        "input": {
            "user_video": video_usuario,
            "expert_video": video_experto,
            "config": config_path,
        },
        "output": {
            "landmarks": {},
            "processed": {},
            "visualizations": {},
            "analysis": {},
        },
        "config_info": {
            "singleton_used": True,
            "config_loaded_at_start": True,
        },
    }

    # 1. EXTRACCIÓN DE LANDMARKS
    if not skip_extraction:
        logger.info("1. FASE DE EXTRACCIÓN DE LANDMARKS")
        if model_path is None:
            model_path = MODEL_PATH

        if os.path.exists(csv_usuario):
            os.remove(csv_usuario)
        if os.path.exists(csv_experto):
            os.remove(csv_experto)

        try:
            logger.info(f"Extrayendo landmarks del video del usuario: {video_usuario}")
            user_data = extract_landmarks_from_video(
                video_usuario,
                csv_usuario,
                exercise_name,
                config_path=config_path,
                model_path=model_path,
                force_model_reload=False,
            )

            logger.info(f"Extrayendo landmarks del video del experto: {video_experto}")
            expert_data = extract_landmarks_from_video(
                video_experto,
                csv_experto,
                exercise_name,
                config_path=config_path,
                model_path=model_path,
                force_model_reload=True,
            )

            results["output"]["landmarks"]["user"] = csv_usuario
            results["output"]["landmarks"]["expert"] = csv_experto

            logger.info(
                f"Extracción completada: {len(user_data)} frames de usuario, {len(expert_data)} frames de experto"
            )

        except Exception as e:
            logger.error(f"Error en extracción de landmarks: {e}")
            if skip_normalization:
                logger.error("No se puede continuar sin datos de landmarks.")
                return results
    else:
        logger.info("Cargando landmarks previamente extraídos...")
        try:
            user_data = pd.read_csv(csv_usuario)
            expert_data = pd.read_csv(csv_experto)

            results["output"]["landmarks"]["user"] = csv_usuario
            results["output"]["landmarks"]["expert"] = csv_experto

            logger.info(
                f"Datos cargados: {len(user_data)} frames de usuario, {len(expert_data)} frames de experto"
            )
        except Exception as e:
            logger.error(f"Error al cargar landmarks: {e}")
            return results

    # Guardar datos originales para referencias posteriores
    user_data_original = user_data.copy()
    expert_data_original = expert_data.copy()

    # 2. DETECCIÓN DE REPETICIONES (UNA SOLA VEZ)
    logger.info("2. FASE DE DETECCIÓN DE REPETICIONES")
    try:
        # Detectar repeticiones del usuario
        user_repetitions = detect_repetitions(user_data)
        num_user_reps = len(user_repetitions) if user_repetitions else 0
        logger.info(f"Repeticiones detectadas en usuario: {num_user_reps}")

        # Detectar repeticiones del experto
        expert_repetitions = detect_repetitions(expert_data)
        num_expert_reps = len(expert_repetitions) if expert_repetitions else 0
        logger.info(f"Repeticiones detectadas en experto: {num_expert_reps}")

        # Calcular rango de ejercicio para visualización (del usuario)
        exercise_frame_range = None
        if user_repetitions:
            min_frame = min(rep["start_frame"] for rep in user_repetitions)
            max_frame = max(rep["end_frame"] for rep in user_repetitions)
            exercise_frame_range = (int(min_frame), int(max_frame))
            logger.info(f"Rango de ejercicio: frames {min_frame} a {max_frame}")
        else:
            logger.warning("No se detectaron repeticiones - usando video completo")

        if diagnostics_dir and user_repetitions:
            rep_info = []
            for i, rep in enumerate(user_repetitions):
                rep_info.append(
                    {
                        "repetition": i + 1,
                        "start_frame": int(rep["start_frame"]),
                        "mid_frame": (
                            int(rep["mid_frame"])
                            if not np.isnan(rep["mid_frame"])
                            else None
                        ),
                        "end_frame": int(rep["end_frame"]),
                        "duration": int(rep["end_frame"] - rep["start_frame"]),
                    }
                )
            with open(os.path.join(diagnostics_dir, "repetitions.json"), "w") as f:
                json.dump(rep_info, f, indent=2)
    except Exception as e:
        logger.error(f"Error en detección de repeticiones: {e}")
        user_repetitions = None
        expert_repetitions = None
        exercise_frame_range = None

    # 3. SINCRONIZACIÓN DE DATOS (CON REPETICIONES PRE-DETECTADAS)
    logger.info("3. FASE DE SINCRONIZACIÓN DE DATOS")
    try:
        user_processed_data, expert_processed_data = synchronize_data(
            user_data,
            expert_data,
            exercise_name=exercise_name,
            config_path=config_path,
            user_repetitions=user_repetitions,  # PASAR REPETICIONES
            expert_repetitions=expert_repetitions,  # PASAR REPETICIONES
        )
        user_sync_path = os.path.join(
            output_dir, f"{exercise_name}_user_synchronized.csv"
        )
        expert_sync_path = os.path.join(
            output_dir, f"{exercise_name}_expert_synchronized.csv"
        )
        user_processed_data.to_csv(user_sync_path, index=False)
        expert_processed_data.to_csv(expert_sync_path, index=False)
        results["output"]["processed"]["user_sync"] = user_sync_path
        results["output"]["processed"]["expert_sync"] = expert_sync_path

        logger.info(f"Sincronización completada: {len(user_processed_data)} frames")

        if diagnostics_dir:
            try:
                sync_stats = {
                    "original_user_frames": len(user_data),
                    "original_expert_frames": len(expert_data),
                    "synchronized_frames": len(user_processed_data),
                    "compression_ratio": (
                        len(user_processed_data) / len(user_data)
                        if len(user_data) > 0
                        else 0
                    ),
                }
                with open(os.path.join(diagnostics_dir, "sync_stats.json"), "w") as f:
                    json.dump(sync_stats, f, indent=2)
            except Exception as e:
                logger.error(f"Error al generar diagnósticos de sincronización: {e}")
    except Exception as e:
        logger.error(f"Error en sincronización: {e}")
        return results

    if skip_normalization:
        logger.info("Omitiendo normalización y alineación según parámetros.")
        return results

    # 4. NORMALIZACIÓN POR CUADRADOS MÍNIMOS
    logger.info("4. FASE DE NORMALIZACIÓN POR CUADRADOS MÍNIMOS")
    try:
        logger.info("Normalizando datos del experto usando transformación afín...")
        normalized_expert_data = normalize_skeletons_with_affine_method(
            user_processed_data, expert_processed_data
        )
        norm_expert_path = os.path.join(
            output_dir, f"{exercise_name}_expert_normalized.csv"
        )
        normalized_expert_data.to_csv(norm_expert_path, index=False)
        results["output"]["processed"]["expert_norm"] = norm_expert_path
        logger.info(f"Normalización completada: {len(normalized_expert_data)} frames")
    except Exception as e:
        logger.error(f"Error en normalización: {e}")
        return results

    # 5. ALINEACIÓN DE ESQUELETOS
    logger.info("5. FASE DE ALINEACIÓN DE ESQUELETOS")
    try:
        logger.info("Alineando esqueletos...")
        aligned_expert_data = align_skeletons_dataframe(
            user_processed_data, normalized_expert_data
        )
        aligned_expert_path = os.path.join(
            output_dir, f"{exercise_name}_expert_aligned.csv"
        )
        aligned_expert_data.to_csv(aligned_expert_path, index=False)
        results["output"]["processed"]["expert_aligned"] = aligned_expert_path
        logger.info(f"Alineación completada: {len(aligned_expert_data)} frames")
    except Exception as e:
        logger.error(f"Error en alineación: {e}")
        aligned_expert_data = normalized_expert_data
        logger.warning("Usando datos normalizados sin alineación final")

    # # 6. GENERACIÓN DE VISUALIZACIONES (CON RANGO DE EJERCICIO)
    # if not skip_visualization:
    #     logger.info("6. FASE DE GENERACIÓN DE VISUALIZACIONES")
    #     try:
    #         output_video_path = os.path.join(
    #             output_dir, f"{exercise_name}_comparison_video.mp4"
    #         )
    #         ensure_dir_exists(output_video_path)
    #         generate_dual_skeleton_video(
    #             original_video_path=video_usuario,
    #             user_data=user_processed_data,
    #             expert_data=aligned_expert_data,
    #             output_video_path=output_video_path,
    #             config_path=CONFIG_PATH,
    #             original_user_data=user_data_original,
    #             exercise_frame_range=exercise_frame_range,  # PASAR RANGO DE EJERCICIO
    #         )
    #         results["output"]["visualizations"]["video"] = output_video_path
    #         logger.info(f"Video comparativo generado: {output_video_path}")

    #         try:
    #             mid_frame = len(user_processed_data) // 2
    #             frame_image_path = os.path.join(
    #                 output_dir, f"{exercise_name}_frame_comparison.png"
    #             )
    #             visualize_frame_dual_skeletons(
    #                 original_image=np.zeros((480, 640, 3), dtype=np.uint8),
    #                 user_frame_data=user_processed_data.iloc[mid_frame],
    #                 expert_frame_data=aligned_expert_data.iloc[mid_frame],
    #                 config_path=CONFIG_PATH,
    #                 save_path=frame_image_path,
    #                 show_image=False,
    #             )
    #             results["output"]["visualizations"]["frame"] = frame_image_path
    #             logger.info(f"Imagen de comparación generada: {frame_image_path}")
    #         except Exception as frame_error:
    #             logger.error(f"Error al generar imagen de comparación: {frame_error}")
    #     except Exception as e:
    #         logger.error(f"Error al generar visualizaciones: {e}")

    # 7. ANÁLISIS DETALLADO DEL EJERCICIO
    if not skip_analysis:
        logger.info("7. FASE DE ANÁLISIS DETALLADO DEL EJERCICIO")
        try:
            analysis_dir = os.path.join(output_dir, f"{exercise_name}_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            # CORREGIDO: Análisis usando singleton - pasamos config_path para usar singleton
            analysis_results = run_exercise_analysis(
                user_data=user_processed_data,
                expert_data=aligned_expert_data,
                exercise_name=exercise_name,
                config_path=config_path,  # Asegurar que usa singleton
            )

            # Generar reporte
            report_path = os.path.join(analysis_dir, f"{exercise_name}_informe.json")
            report = generate_analysis_report(
                analysis_results=analysis_results,
                exercise_name=exercise_name,
                output_path=report_path,
            )

            # Generar visualizaciones
            viz_paths = visualize_analysis_results(
                analysis_results=analysis_results,
                user_data=user_processed_data,
                expert_data=aligned_expert_data,
                exercise_name=exercise_name,
                output_dir=analysis_dir,
            )

            # Almacenar resultados
            results["output"]["analysis"] = {
                "report": report_path,
                "visualizations": viz_paths,
                "score": analysis_results["score"],
                "level": analysis_results["level"],
                "repetitions_used": {
                    "user_reps": len(user_repetitions) if user_repetitions else 0,
                    "expert_reps": len(expert_repetitions) if expert_repetitions else 0,
                    "abduction_analysis": (
                        "bajada_only" if user_repetitions else "completo"
                    ),
                },
                "config_source": "singleton_config_manager",
            }

            logger.info(
                f"Análisis completado - Puntuación: {report['puntuacion_global']:.1f}/100 - Nivel: {report['nivel']}"
            )

            if report["areas_mejora"]:
                logger.info("Áreas de mejora:")
                for area in report["areas_mejora"]:
                    logger.info(f"  - {area}")
            if report["puntos_fuertes"]:
                logger.info("Puntos fuertes:")
                for punto in report["puntos_fuertes"]:
                    logger.info(f"  + {punto}")

        except Exception as e:
            logger.error(f"Error en análisis detallado del ejercicio: {e}")
            import traceback

            logger.error(traceback.format_exc())

    # Información final sobre configuración
    total_time = time.time() - start_time
    results["processing_info"] = {
        "total_time_seconds": total_time,
        "config_manager_singleton": True,
        "config_loaded_once": True,
        "analysis_config_source": "singleton",
    }

    logger.info(
        f"Procesamiento completado en {total_time:.2f} segundos usando singleton correctamente"
    )

    config_manager.clear_cache()

    return results


def main():
    print("DEBUG: Verificando rutas y archivos...")
    print("DEBUG: Directorio de trabajo actual:", os.getcwd())
    print(f"DEBUG: BASE_DIR: {BASE_DIR}")
    print(f"DEBUG: VIDEOS_DIR: {VIDEOS_DIR}")
    print(f"DEBUG: OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"DEBUG: CONFIG_PATH: {CONFIG_PATH}")
    print(f"DEBUG: MODEL_PATH: {MODEL_PATH}")

    exercise_name = nombre_ejercicio
    user_video = os.path.join(VIDEOS_DIR, "Press_Militar_Cerca_Usuario.mp4")
    expert_video = os.path.join(VIDEOS_DIR, "Press_Militar_Cerca_Monitor.mp4")

    print(f"DEBUG: Ruta completa al video de usuario: {user_video}")
    print(f"DEBUG: ¿Existe el video de usuario?: {os.path.exists(user_video)}")
    print(f"DEBUG: Ruta completa al video de experto: {expert_video}")
    print(f"DEBUG: ¿Existe el video de experto?: {os.path.exists(expert_video)}")
    print(f"DEBUG: Ruta al modelo: {MODEL_PATH}")
    print(f"DEBUG: ¿Existe el modelo?: {os.path.exists(MODEL_PATH)}")
    print(f"DEBUG: Ruta a config.json: {CONFIG_PATH}")
    print(f"DEBUG: ¿Existe config.json?: {os.path.exists(CONFIG_PATH)}")

    # INFORMACIÓN ADICIONAL: Verificar que el singleton funciona
    try:
        print(
            f"DEBUG: Configuración cargada en singleton: {len(config_manager._loaded_files)} archivos"
        )
        print(
            f"DEBUG: Ejercicios disponibles: {config_manager.get_available_exercises(CONFIG_PATH)}"
        )
    except Exception as e:
        print(f"DEBUG: Error verificando singleton: {e}")

    results = process_exercise(
        exercise_name=exercise_name,
        videos_dir=VIDEOS_DIR,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
        user_video=user_video,
        expert_video=expert_video,
        diagnostics=True,
        skip_extraction=False,
        skip_normalization=False,
        skip_visualization=True,  # Cambiar a False si quieres generar videos
        skip_analysis=False,
        model_path=MODEL_PATH,
    )

    if results:
        logger.info(
            f"Procesamiento completado con éxito usando singleton correctamente."
        )
        logger.info(f"Información de configuración: {results.get('config_info', {})}")
        logger.info(
            f"Información de procesamiento: {results.get('processing_info', {})}"
        )
        logger.info(f"Todos los resultados están disponibles en: {OUTPUT_DIR}")
    else:
        logger.error("Error en el procesamiento del ejercicio.")


if __name__ == "__main__":
    main()
