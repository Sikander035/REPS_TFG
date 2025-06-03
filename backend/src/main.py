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
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Absolute base path of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEOS_DIR = os.path.join(BASE_DIR, "media", "videos")
CONFIG_PATH = os.path.join(BASE_DIR, "src", "config", "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
PROMPT_PATH = os.path.join(BASE_DIR, "src", "config", "trainer_prompt.txt")

# Read exercise name from config.json
exercise_name = "military_press"
OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output", f"results_{exercise_name}")

# Add path to project root directory (backend)
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

    # CORRECTED: Use only singleton, not duplicated function
    from src.config.config_manager import config_manager
    from src.feedback.analysis_report import (
        run_exercise_analysis,
        generate_analysis_report,
    )
    from src.feedback.analysis_graphics import visualize_analysis_results

    # NEW: Import custom feedback module
    from src.feedback.analysis_llm import generate_trainer_feedback

    TRAINER_FEEDBACK_AVAILABLE = True
    logger.info("Feedback module with DeepSeek V3 loaded correctly")

    # CORRECTED: Load configuration using singleton once at startup
    config_manager.load_config_file(CONFIG_PATH)
    logger.info("Configuration loaded using config_manager singleton")

except ImportError as e:
    logger.warning(f"Not all necessary modules found: {e}")
    logger.warning("Processing may fail")
    TRAINER_FEEDBACK_AVAILABLE = False


def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def get_deepseek_api_key():
    """
    Obtiene la API key de DeepSeek desde variables de entorno.

    Returns:
        str: API key de DeepSeek

    Raises:
        ValueError: Si no se encuentra la API key
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        logger.error("❌ DEEPSEEK_API_KEY no encontrada en variables de entorno")
        logger.error(
            "📝 Asegúrate de que el archivo .env contenga: DEEPSEEK_API_KEY=tu_clave_real"
        )
        raise ValueError(
            "DEEPSEEK_API_KEY no configurada. "
            "Crea un archivo .env en la carpeta backend con: DEEPSEEK_API_KEY=tu_clave_real"
        )

    if api_key == "your_deepseek_api_key":
        logger.error(
            "❌ Debes cambiar 'your_deepseek_api_key' por tu clave real en el archivo .env"
        )
        raise ValueError(
            "Debes cambiar el valor por defecto en .env por tu API key real de DeepSeek"
        )

    logger.info("✅ DEEPSEEK_API_KEY cargada correctamente desde variables de entorno")
    return api_key


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
    skip_trainer_feedback=False,
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
        logger.info(f"Generating diagnostics at: {diagnostics_dir}")

    try:
        # CORRECTED: Use singleton correctly
        exercise_config = config_manager.get_exercise_config(exercise_name, config_path)
        logger.info(f"Configuration loaded for exercise: {exercise_name}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        exercise_config = {"sync_config": {}}

    if diagnostics_dir:
        with open(os.path.join(diagnostics_dir, "config.json"), "w") as f:
            json.dump(exercise_config, f, indent=2)

    if user_video is None:
        user_video = os.path.join(videos_dir, f"{exercise_name}_User.mp4")
    if expert_video is None:
        expert_video = os.path.join(videos_dir, f"{exercise_name}_Expert.mp4")
    else:
        expert_video = (
            os.path.join(videos_dir, expert_video)
            if not os.path.isabs(expert_video)
            else expert_video
        )

    video_user = user_video
    video_expert = expert_video
    csv_user = os.path.join(output_dir, f"{exercise_name}_User.csv")
    csv_expert = os.path.join(output_dir, f"{exercise_name}_Expert.csv")

    logger.info(f"USING USER VIDEO: {video_user}")
    logger.info(f"USING EXPERT VIDEO: {video_expert}")
    logger.info(f"USING MODEL: {model_path}")
    if not os.path.exists(video_user):
        logger.error(f"User video not found: {video_user}")
        return None
    if not os.path.exists(video_expert):
        logger.error(f"Expert video not found: {video_expert}")
        return None

    results = {
        "exercise_name": exercise_name,
        "input": {
            "user_video": video_user,
            "expert_video": video_expert,
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

    # 1. LANDMARK EXTRACTION
    if not skip_extraction:
        logger.info("1. LANDMARK EXTRACTION PHASE")
        if model_path is None:
            model_path = MODEL_PATH

        if os.path.exists(csv_user):
            os.remove(csv_user)
        if os.path.exists(csv_expert):
            os.remove(csv_expert)

        try:
            logger.info(f"Extracting landmarks from user video: {video_user}")
            user_data = extract_landmarks_from_video(
                video_user,
                csv_user,
                exercise_name,
                config_path=config_path,
                model_path=model_path,
                force_model_reload=False,
            )

            logger.info(f"Extracting landmarks from expert video: {video_expert}")
            expert_data = extract_landmarks_from_video(
                video_expert,
                csv_expert,
                exercise_name,
                config_path=config_path,
                model_path=model_path,
                force_model_reload=True,
            )

            results["output"]["landmarks"]["user"] = csv_user
            results["output"]["landmarks"]["expert"] = csv_expert

            logger.info(
                f"Extraction completed: {len(user_data)} user frames, {len(expert_data)} expert frames"
            )

        except Exception as e:
            logger.error(f"Error in landmark extraction: {e}")
            if skip_normalization:
                logger.error("Cannot continue without landmark data.")
                return results
    else:
        logger.info("Loading previously extracted landmarks...")
        try:
            user_data = pd.read_csv(csv_user)
            expert_data = pd.read_csv(csv_expert)

            results["output"]["landmarks"]["user"] = csv_user
            results["output"]["landmarks"]["expert"] = csv_expert

            logger.info(
                f"Data loaded: {len(user_data)} user frames, {len(expert_data)} expert frames"
            )
        except Exception as e:
            logger.error(f"Error loading landmarks: {e}")
            return results

    # Save original data for later references
    user_data_original = user_data.copy()
    expert_data_original = expert_data.copy()

    # 2. REPETITION DETECTION (ONCE ONLY)
    logger.info("2. REPETITION DETECTION PHASE")
    try:
        # Detect user repetitions
        user_repetitions = detect_repetitions(user_data)
        num_user_reps = len(user_repetitions) if user_repetitions else 0
        logger.info(f"Repetitions detected in user: {num_user_reps}")

        # Detect expert repetitions
        expert_repetitions = detect_repetitions(expert_data)
        num_expert_reps = len(expert_repetitions) if expert_repetitions else 0
        logger.info(f"Repetitions detected in expert: {num_expert_reps}")

        # Calculate exercise range for visualization (from user)
        exercise_frame_range = None
        if user_repetitions:
            min_frame = min(rep["start_frame"] for rep in user_repetitions)
            max_frame = max(rep["end_frame"] for rep in user_repetitions)
            exercise_frame_range = (int(min_frame), int(max_frame))
            logger.info(f"Exercise range: frames {min_frame} to {max_frame}")
        else:
            logger.warning("No repetitions detected - using complete video")

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
        logger.error(f"Error in repetition detection: {e}")
        user_repetitions = None
        expert_repetitions = None
        exercise_frame_range = None

    # 3. DATA SYNCHRONIZATION (WITH PRE-DETECTED REPETITIONS)
    logger.info("3. DATA SYNCHRONIZATION PHASE")
    try:
        user_processed_data, expert_processed_data = synchronize_data(
            user_data,
            expert_data,
            exercise_name=exercise_name,
            config_path=config_path,
            user_repetitions=user_repetitions,  # PASS REPETITIONS
            expert_repetitions=expert_repetitions,  # PASS REPETITIONS
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

        logger.info(f"Synchronization completed: {len(user_processed_data)} frames")

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
                logger.error(f"Error generating synchronization diagnostics: {e}")
    except Exception as e:
        logger.error(f"Error in synchronization: {e}")
        return results

    if skip_normalization:
        logger.info("Skipping normalization and alignment per parameters.")
        return results

    # 4. LEAST SQUARES NORMALIZATION
    logger.info("4. LEAST SQUARES NORMALIZATION PHASE")
    try:
        logger.info("Normalizing expert data using affine transformation...")
        normalized_expert_data = normalize_skeletons_with_affine_method(
            user_processed_data, expert_processed_data
        )
        norm_expert_path = os.path.join(
            output_dir, f"{exercise_name}_expert_normalized.csv"
        )
        normalized_expert_data.to_csv(norm_expert_path, index=False)
        results["output"]["processed"]["expert_norm"] = norm_expert_path
        logger.info(f"Normalization completed: {len(normalized_expert_data)} frames")
    except Exception as e:
        logger.error(f"Error in normalization: {e}")
        return results

    # 5. SKELETON ALIGNMENT
    logger.info("5. SKELETON ALIGNMENT PHASE")
    try:
        logger.info("Aligning skeletons...")
        aligned_expert_data = align_skeletons_dataframe(
            user_processed_data, normalized_expert_data
        )
        aligned_expert_path = os.path.join(
            output_dir, f"{exercise_name}_expert_aligned.csv"
        )
        aligned_expert_data.to_csv(aligned_expert_path, index=False)
        results["output"]["processed"]["expert_aligned"] = aligned_expert_path
        logger.info(f"Alignment completed: {len(aligned_expert_data)} frames")
    except Exception as e:
        logger.error(f"Error in alignment: {e}")
        aligned_expert_data = normalized_expert_data
        logger.warning("Using normalized data without final alignment")

    # 6. GENERACIÓN DE VISUALIZACIONES (CON RANGO DE EJERCICIO)
    if not skip_visualization:
        logger.info("6. FASE DE GENERACIÓN DE VISUALIZACIONES")
        try:
            output_video_path = os.path.join(
                output_dir, f"{exercise_name}_comparison_video.mp4"
            )
            ensure_dir_exists(output_video_path)
            generate_dual_skeleton_video(
                original_video_path=video_user,
                user_data=user_processed_data,
                expert_data=aligned_expert_data,
                output_video_path=output_video_path,
                config_path=CONFIG_PATH,
                original_user_data=user_data_original,
                exercise_frame_range=exercise_frame_range,  # PASAR RANGO DE EJERCICIO
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
                    config_path=CONFIG_PATH,
                    save_path=frame_image_path,
                    show_image=False,
                )
                results["output"]["visualizations"]["frame"] = frame_image_path
                logger.info(f"Imagen de comparación generada: {frame_image_path}")
            except Exception as frame_error:
                logger.error(f"Error al generar imagen de comparación: {frame_error}")
        except Exception as e:
            logger.error(f"Error al generar visualizaciones: {e}")

    # 7. DETAILED EXERCISE ANALYSIS
    if not skip_analysis:
        logger.info("7. DETAILED EXERCISE ANALYSIS PHASE")
        try:
            analysis_dir = os.path.join(output_dir, f"{exercise_name}_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            # CORRECTED: Analysis using singleton - pass config_path to use singleton
            analysis_results = run_exercise_analysis(
                user_data=user_processed_data,
                expert_data=aligned_expert_data,
                exercise_name=exercise_name,
                config_path=config_path,
                user_repetitions=user_repetitions,
                expert_repetitions=expert_repetitions,
            )

            # Generate report
            report_path = os.path.join(analysis_dir, f"{exercise_name}_report.json")
            report = generate_analysis_report(
                analysis_results=analysis_results,
                exercise_name=exercise_name,
                output_path=report_path,
            )

            # Generate visualizations
            viz_paths = visualize_analysis_results(
                analysis_results=analysis_results,
                user_data=user_processed_data,
                expert_data=aligned_expert_data,
                exercise_name=exercise_name,
                output_dir=analysis_dir,
                config_path=config_path,
            )

            # Store results
            results["output"]["analysis"] = {
                "report": report_path,
                "visualizations": viz_paths,
                "score": analysis_results["score"],
                "level": analysis_results["level"],
                "repetitions_used": {
                    "user_reps": len(user_repetitions) if user_repetitions else 0,
                    "expert_reps": len(expert_repetitions) if expert_repetitions else 0,
                    "analysis_type": (
                        "descent_only" if user_repetitions else "complete"
                    ),
                },
                "config_source": "singleton_config_manager",
            }

            logger.info(
                f"Analysis completed - Score: {report['overall_score']:.1f}/100 - Level: {report['level']}"
            )

            if report["improvement_areas"]:
                logger.info("Improvement areas:")
                for area in report["improvement_areas"]:
                    logger.info(f"  - {area}")
            if report["strengths"]:
                logger.info("Strengths:")
                for strength in report["strengths"]:
                    logger.info(f"  + {strength}")

        except Exception as e:
            logger.error(f"Error in detailed exercise analysis: {e}")
            import traceback

            logger.error(traceback.format_exc())

    # 8. PERSONALIZED FEEDBACK GENERATION (NEW STEP)
    if not skip_trainer_feedback and TRAINER_FEEDBACK_AVAILABLE:
        logger.info("8. PERSONALIZED FEEDBACK GENERATION PHASE")
        try:
            # Obtener API key desde variables de entorno
            deepseek_api_key = get_deepseek_api_key()

            # CORRECTED: Paths based on real project structure
            analysis_dir = os.path.join(output_dir, f"{exercise_name}_analysis")
            report_path = os.path.join(analysis_dir, f"{exercise_name}_report.json")

            logger.info(f"Looking for report at: {report_path}")

            if os.path.exists(report_path):
                # Generate feedback using DeepSeek V3
                feedback_path = os.path.join(
                    analysis_dir, f"{exercise_name}_personalized_feedback.txt"
                )

                logger.info("Generating personalized feedback with DeepSeek V3...")

                # Use API key from environment variables
                if os.path.exists(PROMPT_PATH):
                    logger.info(f"Using prompt file: {PROMPT_PATH}")
                    prompt_file_arg = PROMPT_PATH
                else:
                    logger.info(
                        "Prompt file not found at expected location, using auto-discovery"
                    )
                    prompt_file_arg = None

                # Use API key from environment variables and pass prompt path
                feedback = generate_trainer_feedback(
                    informe_path=report_path,
                    output_path=feedback_path,
                    api_key=deepseek_api_key,
                    prompt_file_path=prompt_file_arg,  # ← AÑADIR esta línea
                )

                # Add result to results dictionary
                if "analysis" not in results["output"]:
                    results["output"]["analysis"] = {}

                results["output"]["analysis"]["trainer_feedback"] = feedback_path
                results["output"]["analysis"]["feedback_preview"] = (
                    feedback[:200] + "..." if len(feedback) > 200 else feedback
                )

                logger.info(f"Personalized feedback generated: {feedback_path}")

                # Show feedback preview in logs
                logger.info("=== FEEDBACK PREVIEW ===")
                lines = feedback.split("\n")[:5]  # First 5 lines
                for line in lines:
                    if line.strip():
                        logger.info(f"  {line.strip()}")
                logger.info("=== END PREVIEW ===")

                # SHOW COMPLETE FEEDBACK AT THE END
                print("\n" + "=" * 60)
                print("🤖 PERSONAL TRAINER FEEDBACK")
                print("=" * 60)
                print(feedback)
                print("=" * 60 + "\n")

            else:
                logger.warning(
                    f"Analysis report not found at {report_path}. Skipping feedback generation."
                )
                logger.info("Directory structure:")
                if os.path.exists(output_dir):
                    for item in os.listdir(output_dir):
                        logger.info(f"  - {item}")
                else:
                    logger.warning(f"Output directory doesn't exist: {output_dir}")

        except Exception as e:
            logger.error(f"Error in personalized feedback generation: {e}")
            import traceback

            logger.error(traceback.format_exc())
    elif skip_trainer_feedback:
        logger.info("Skipping personalized feedback generation per parameters.")
    else:
        logger.warning(
            "Trainer feedback module not available. Install necessary dependencies."
        )

    # Final information about configuration
    total_time = time.time() - start_time
    results["processing_info"] = {
        "total_time_seconds": total_time,
        "config_manager_singleton": True,
        "config_loaded_once": True,
        "analysis_config_source": "singleton",
    }

    logger.info(
        f"Processing completed in {total_time:.2f} seconds using singleton correctly"
    )

    config_manager.clear_cache()

    return results


def main():
    print("DEBUG: Verifying paths and files...")
    print("DEBUG: Current working directory:", os.getcwd())
    print(f"DEBUG: BASE_DIR: {BASE_DIR}")
    print(f"DEBUG: VIDEOS_DIR: {VIDEOS_DIR}")
    print(f"DEBUG: OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"DEBUG: CONFIG_PATH: {CONFIG_PATH}")
    print(f"DEBUG: MODEL_PATH: {MODEL_PATH}")

    exercise_name_for_files = exercise_name
    user_video = os.path.join(VIDEOS_DIR, "Press_Militar_Cerca_Usuario.mp4")
    expert_video = os.path.join(VIDEOS_DIR, "Press_Militar_Cerca_Monitor.mp4")

    print(f"DEBUG: Full path to user video: {user_video}")
    print(f"DEBUG: Does user video exist?: {os.path.exists(user_video)}")
    print(f"DEBUG: Full path to expert video: {expert_video}")
    print(f"DEBUG: Does expert video exist?: {os.path.exists(expert_video)}")
    print(f"DEBUG: Path to model: {MODEL_PATH}")
    print(f"DEBUG: Does model exist?: {os.path.exists(MODEL_PATH)}")
    print(f"DEBUG: Path to config.json: {CONFIG_PATH}")
    print(f"DEBUG: Does config.json exist?: {os.path.exists(CONFIG_PATH)}")

    # ADDITIONAL INFORMATION: Verify singleton works
    try:
        print(
            f"DEBUG: Configuration loaded in singleton: {len(config_manager._loaded_files)} files"
        )
        print(
            f"DEBUG: Available exercises: {config_manager.get_available_exercises(CONFIG_PATH)}"
        )
    except Exception as e:
        print(f"DEBUG: Error verifying singleton: {e}")

    # VERIFICAR VARIABLES DE ENTORNO
    try:
        deepseek_api_key = get_deepseek_api_key()
        print("DEBUG: ✅ DEEPSEEK_API_KEY loaded successfully")
    except Exception as e:
        print(f"DEBUG: ❌ Error loading DEEPSEEK_API_KEY: {e}")
        return

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
        skip_trainer_feedback=False,  # NEW: Change to True to disable feedback
        model_path=MODEL_PATH,
    )

    if results:
        logger.info(f"Processing completed successfully using singleton correctly.")
        logger.info(f"Configuration information: {results.get('config_info', {})}")
        logger.info(f"Processing information: {results.get('processing_info', {})}")

        # NEW: Show feedback information if available
        if (
            "analysis" in results["output"]
            and "trainer_feedback" in results["output"]["analysis"]
        ):
            feedback_path = results["output"]["analysis"]["trainer_feedback"]
            logger.info(f"Personalized feedback generated at: {feedback_path}")

        logger.info(f"All results are available at: {OUTPUT_DIR}")
    else:
        logger.error("Error in exercise processing.")


if __name__ == "__main__":
    main()
