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

# Ruta absoluta base del proyecto (ajusta si es necesario)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEOS_DIR = os.path.join(BASE_DIR, "media", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output", "resultados")
CONFIG_PATH = os.path.join(BASE_DIR, "src", "config", "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_heavy.task")

# Añadir la ruta al directorio raíz del proyecto (backend)
sys.path.append(BASE_DIR)
# Añadir también el directorio src para que las importaciones relativas funcionen
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

    return results


def main():
    print("DEBUG: Verificando rutas y archivos...")
    print("DEBUG: Directorio de trabajo actual:", os.getcwd())
    print(f"DEBUG: BASE_DIR: {BASE_DIR}")
    print(f"DEBUG: VIDEOS_DIR: {VIDEOS_DIR}")
    print(f"DEBUG: OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"DEBUG: CONFIG_PATH: {CONFIG_PATH}")
    print(f"DEBUG: MODEL_PATH: {MODEL_PATH}")

    exercise_name = "press_militar"
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
