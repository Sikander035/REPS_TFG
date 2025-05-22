"""
Main synchronization function for exercise data.

This module provides the synchronize_data function which is the main entry point
for synchronizing exercise data between user and expert recordings.
"""

import numpy as np
import pandas as pd
import sys
import logging
import os
from typing import List, Dict, Tuple, Optional, Union, Any

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import utility functions from the utilities module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.synchronization_utils import (
    validate_input_data,
    preprocess_dataframe,
    match_repetitions,
    process_all_repetitions,
    combine_and_validate,
)


def synchronize_data(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    exercise_name: Optional[str] = None,
    config_path: str = "config.json",
    user_repetitions: Optional[List[Dict]] = None,  # NUEVO PARÁMETRO
    expert_repetitions: Optional[List[Dict]] = None,  # NUEVO PARÁMETRO
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronizes expert data with user data using a flexible, configurable strategy.

    This is the main entry point for the synchronization process. It orchestrates
    the entire synchronization workflow, including preprocessing, repetition detection,
    phase identification, and interpolation.

    Args:
        user_data: DataFrame with user data
        expert_data: DataFrame with expert data
        config: Dictionary with configuration (optional if exercise_name is provided)
        exercise_name: Exercise name to load configuration (optional if config is provided)
        config_path: Path to configuration file (default: "config.json")
        user_repetitions: Pre-detected user repetitions (optional)
        expert_repetitions: Pre-detected expert repetitions (optional)

    Returns:
        Tuple (user_data_sync, expert_data_sync) with synchronized data
    """
    logger.info("Starting synchronization process...")

    # Validar repeticiones si se proporcionan
    if user_repetitions is not None:
        if not isinstance(user_repetitions, list):
            raise ValueError("user_repetitions debe ser una lista")
        logger.info(f"Using {len(user_repetitions)} pre-detected user repetitions")

    if expert_repetitions is not None:
        if not isinstance(expert_repetitions, list):
            raise ValueError("expert_repetitions debe ser una lista")
        logger.info(f"Using {len(expert_repetitions)} pre-detected expert repetitions")

    # Obtener configuración - CARGA ÚNICA
    if config is None:
        if exercise_name is None:
            raise ValueError("Either config or exercise_name must be provided")

        try:
            # Importar solo una vez
            from src.config.config_manager import load_exercise_config

            exercise_config = load_exercise_config(exercise_name, config_path)

            if not exercise_config or "sync_config" not in exercise_config:
                raise ValueError(f"No sync_config found for exercise {exercise_name}")

            config = exercise_config["sync_config"]
            logger.info(f"Configuration loaded for exercise: {exercise_name}")
        except Exception as e:
            raise ValueError(f"Could not load configuration for {exercise_name}: {e}")
    else:
        logger.info("Using provided configuration")

    # 1. Validar datos de entrada
    validate_input_data(user_data, expert_data, config)

    # 2. Preprocesar datos (opcional)
    if config.get("preprocess", True):
        logger.info("Preprocessing data...")
        user_data_prep = preprocess_dataframe(user_data, config)
        expert_data_prep = preprocess_dataframe(expert_data, config)
    else:
        user_data_prep = user_data.copy()
        expert_data_prep = expert_data.copy()

    # 3. Detectar repeticiones (solo si no se proporcionaron)
    if user_repetitions is None or expert_repetitions is None:
        try:
            # Importar función de detección
            from src.core.data_segmentation.detect_repetitions import detect_repetitions

            if user_repetitions is None:
                logger.info("Detecting repetitions in user data...")
                user_repetitions = detect_repetitions(
                    user_data_prep,
                    plot_graph=False,
                    config=config,
                )

            if expert_repetitions is None:
                logger.info("Detecting repetitions in expert data...")
                expert_repetitions = detect_repetitions(
                    expert_data_prep,
                    plot_graph=False,
                    config=config,
                )

            if not user_repetitions:
                raise ValueError("No repetitions detected in user data")
            if not expert_repetitions:
                raise ValueError("No repetitions detected in expert data")

            logger.info(
                f"Detected {len(user_repetitions)} user repetitions and {len(expert_repetitions)} expert repetitions"
            )
        except Exception as e:
            logger.error(f"Error detecting repetitions: {e}")
            raise
    else:
        logger.info(
            f"Using provided repetitions: {len(user_repetitions)} user, {len(expert_repetitions)} expert"
        )

    # 4. Emparejar repeticiones
    logger.info("Matching repetitions...")
    repetition_pairs = match_repetitions(
        user_repetitions, expert_repetitions, user_data_prep, expert_data_prep, config
    )

    # 5. Procesar todas las repeticiones
    logger.info(f"Processing {len(repetition_pairs)} repetition pairs...")
    user_segments, expert_segments = process_all_repetitions(
        user_data_prep, expert_data_prep, repetition_pairs, config
    )

    # 6. Combinar y validar
    logger.info("Combining results...")
    final_user_data, final_expert_data = combine_and_validate(
        user_segments, expert_segments
    )

    logger.info("Synchronization successfully completed.")
    return final_user_data, final_expert_data
