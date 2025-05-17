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
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_heavy.task")

# Leer el nombre del ejercicio desde config.json
nombre_ejercicio = "press_militar"
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output", f"resultados_{nombre_ejercicio}")

# Añadir la ruta al directorio raíz del proyecto (backend)
sys.path.append(BASE_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from core.data_extraction import extract_landmarks_from_video
    from core.detect_repetitions import detect_repetitions
    from core.synchronise_by_interpolation import (
        synchronize_data_by_height as synchronize_data,
    )
    from core.min_square_normalization import normalize_skeletons_with_affine_method
    from core.align_data import align_skeletons_dataframe
    from core.visualization import (
        generate_dual_skeleton_video,
        visualize_frame_dual_skeletons,
    )
    from config.config_manager import load_exercise_config, config_manager

    config_manager.load_config_file(CONFIG_PATH)
    from core.exercise_analyzer import ExerciseAnalyzer

    logger.info("Usando módulos existentes")

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
        exercise_config = load_exercise_config(exercise_name, config_path)
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
            )

            logger.info(f"Extrayendo landmarks del video del experto: {video_experto}")
            expert_data = extract_landmarks_from_video(
                video_experto,
                csv_experto,
                exercise_name,
                config_path=config_path,
                model_path=model_path,
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

    # 2. DETECCIÓN DE REPETICIONES
    logger.info("2. FASE DE DETECCIÓN DE REPETICIONES")
    try:
        user_repetitions = detect_repetitions(user_data)
        num_reps = len(user_repetitions) if user_repetitions else 0
        logger.info(f"Repeticiones detectadas: {num_reps}")

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

    # 3. SINCRONIZACIÓN DE DATOS
    logger.info("3. FASE DE SINCRONIZACIÓN DE DATOS")
    try:
        user_processed_data, expert_processed_data = synchronize_data(
            user_data,
            expert_data,
            num_divisions=7,
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

    # 6. GENERACIÓN DE VISUALIZACIONES
    if not skip_visualization:
        logger.info("6. FASE DE GENERACIÓN DE VISUALIZACIONES")
        try:
            output_video_path = os.path.join(
                output_dir, f"{exercise_name}_comparison_video.mp4"
            )
            ensure_dir_exists(output_video_path)
            generate_dual_skeleton_video(
                original_video_path=video_usuario,
                user_data=user_processed_data,
                expert_data=aligned_expert_data,
                output_video_path=output_video_path,
                original_user_data=user_data_original,
                user_color=(0, 255, 0),
                expert_color=(0, 0, 255),
                user_alpha=0.7,
                expert_alpha=0.9,
                user_thickness=2,
                expert_thickness=3,
                resize_factor=1.0,
                show_progress=True,
                text_info=True,
            )
            results["output"]["visualizations"]["video"] = output_video_path
            logger.info(f"Video comparativo generado: {output_video_path}")

            try:
                mid_frame = len(user_processed_data) // 2
                frame_image_path = os.path.join(
                    output_dir, f"{exercise_name}_frame_comparison.png"
                )
                visualize_frame_dual_skeletons(
                    original_image=np.zeros((480, 640, 3), dtype=np.uint8),
                    user_frame_data=user_processed_data.iloc[mid_frame],
                    expert_frame_data=aligned_expert_data.iloc[mid_frame],
                    save_path=frame_image_path,
                    show_image=False,
                )
                results["output"]["visualizations"]["frame"] = frame_image_path
                logger.info(f"Imagen de comparación generada: {frame_image_path}")
            except Exception as frame_error:
                logger.error(f"Error al generar imagen de comparación: {frame_error}")
        except Exception as e:
            logger.error(f"Error al generar visualizaciones: {e}")

    # 7. ANÁLISIS DETALLADO DEL EJERCICIO
    if not skip_analysis:
        logger.info("7. FASE DE ANÁLISIS DETALLADO DEL EJERCICIO")
        try:
            analysis_dir = os.path.join(output_dir, f"{exercise_name}_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            analyzer = ExerciseAnalyzer(
                user_data=user_processed_data,
                expert_data=aligned_expert_data,
                exercise_name=exercise_name,
            )
            analysis_results = analyzer.run_full_analysis()
            report_path = os.path.join(analysis_dir, f"{exercise_name}_informe.json")
            report = analyzer.generate_report(output_path=report_path)
            viz_paths = analyzer.visualize_analysis(output_dir=analysis_dir)
            results["output"]["analysis"] = {
                "report": report_path,
                "visualizations": viz_paths,
                "score": analysis_results["score"],
                "level": report["nivel"],
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

    elapsed_time = time.time() - start_time
    logger.info(
        f"Procesamiento completo de {exercise_name} en {elapsed_time:.2f} segundos"
    )
    summary = {
        "exercise": exercise_name,
        "timestamp": run_id,
        "processing_time": elapsed_time,
        "frames": {
            "original_user": len(user_data_original),
            "original_expert": len(expert_data_original),
            "processed": len(user_processed_data),
            "normalized": len(normalized_expert_data),
            "aligned": (
                len(aligned_expert_data)
                if "aligned_expert_data" in locals()
                else len(normalized_expert_data)
            ),
        },
        "videos": {"user": video_usuario, "expert": video_experto},
        "files": results["output"],
    }
    summary_path = os.path.join(output_dir, f"{exercise_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

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
        skip_visualization=False,
        skip_analysis=False,
        model_path=MODEL_PATH,
    )

    if results:
        logger.info(f"Procesamiento completado con éxito.")
        logger.info(f"Todos los resultados están disponibles en: {OUTPUT_DIR}")
    else:
        logger.error("Error en el procesamiento del ejercicio.")


if __name__ == "__main__":
    main()
