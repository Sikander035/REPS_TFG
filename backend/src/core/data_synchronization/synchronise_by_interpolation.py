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

# Ensure path includes parent directory for imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

# Import utility functions from the utilities module
from src.utils.synchronization_utils import (
    validate_input_data,
    preprocess_dataframe,
    get_repetitions,
    match_repetitions,
    process_all_repetitions,
    combine_and_validate,
)

# Import from detect_repetitions to maintain compatibility
try:
    from src.core.data_segmentation.detect_repetitions import detect_repetitions
except ImportError:
    logging.warning(
        "Could not import detect_repetitions, some functions may be limited"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def synchronize_data(
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    exercise_name: Optional[str] = None,
    config_path: str = "config.json",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronizes expert data with user data using a flexible, configurable strategy.

    This is the main entry point for the synchronization process. It orchestrates
    the entire synchronization workflow, including preprocessing, repetition
    detection, phase identification, and interpolation.

    Args:
        user_data: DataFrame with user data
        expert_data: DataFrame with expert data
        config: Dictionary with configuration (optional if exercise_name is provided)
        exercise_name: Exercise name to load configuration (optional if config is provided)
        config_path: Path to configuration file (default: "config_expanded.json")

    Returns:
        Tuple (user_data_sync, expert_data_sync) with synchronized data

    Raises:
        ValueError: If configuration cannot be obtained

    Example:
        # Option 1: Pass configuration directly
        >>> user_df = pd.read_csv("user_landmarks.csv")
        >>> expert_df = pd.read_csv("expert_landmarks.csv")
        >>> config = {"landmarks": ["landmark_right_wrist", "landmark_left_wrist"],
                      "division_strategy": "height",
                      "num_divisions": 7}
        >>> user_sync, expert_sync = synchronize_data(
        ...     user_df, expert_df, config=config
        ... )

        # Option 2: Load configuration from exercise_name
        >>> user_sync, expert_sync = synchronize_data(
        ...     user_df, expert_df, exercise_name="press_militar"
        ... )
    """
    logger.info("Starting synchronization process...")

    # Get configuration - either provided directly or from exercise name
    if config is None:
        if exercise_name is None:
            raise ValueError("Either config or exercise_name must be provided")

        try:
            # Try to import from config_manager
            try:
                from src.config.config_manager import load_exercise_config

                exercise_config = load_exercise_config(exercise_name, config_path)
            except ImportError as e:
                raise ImportError(
                    "No se pudo importar load_exercise_config desde src.config.config_manager"
                ) from e

            if not exercise_config or "sync_config" not in exercise_config:
                raise ValueError(f"No sync_config found for exercise {exercise_name}")

            config = exercise_config["sync_config"]
            logger.info(f"Configuration loaded for exercise: {exercise_name}")
        except Exception as e:
            raise ValueError(f"Could not load configuration for {exercise_name}: {e}")
    else:
        logger.info("Using provided configuration")

    # 1. Validate input data
    validate_input_data(user_data, expert_data, config)

    # 2. Preprocess data (optional)
    if config.get("preprocess", True):
        logger.info("Preprocessing data...")
        user_data_prep = preprocess_dataframe(user_data, config)
        expert_data_prep = preprocess_dataframe(expert_data, config)
    else:
        user_data_prep = user_data.copy()
        expert_data_prep = expert_data.copy()

    # 3. Detect and match repetitions
    logger.info("Detecting repetitions...")
    user_repetitions = get_repetitions(user_data_prep, config, is_user=True)
    expert_repetitions = get_repetitions(expert_data_prep, config, is_user=False)

    logger.info("Matching repetitions...")
    repetition_pairs = match_repetitions(
        user_repetitions, expert_repetitions, user_data_prep, expert_data_prep, config
    )

    # 4. Process all repetitions
    logger.info(f"Processing {len(repetition_pairs)} repetition pairs...")
    user_segments, expert_segments = process_all_repetitions(
        user_data_prep, expert_data_prep, repetition_pairs, config
    )

    # 5. Combine and validate
    logger.info("Combining results...")
    final_user_data, final_expert_data = combine_and_validate(
        user_segments, expert_segments
    )

    logger.info("Synchronization successfully completed.")
    return final_user_data, final_expert_data
