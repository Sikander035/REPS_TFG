# backend/src/core/align_data.py
import pandas as pd
import numpy as np
import logging
import sys
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import load_exercise_config
from src.utils.landmark_utils import align_skeleton_frame, extract_landmarks_as_matrix

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def align_skeletons_dataframe(
    user_data,
    expert_data,
    config=None,
    exercise_name=None,
    config_path="config_expanded.json",
    alignment_method=None,
    reference_landmarks=None,
    use_parallel=False,
    max_workers=4,
    compute_error=False,
    show_progress=True,
    diagnostics_path=None,
):
    """
    Alinea todos los frames del esqueleto del experto con los del usuario.
    """
    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los DataFrames deben tener la misma longitud. Usuario: {len(user_data)}, Experto: {len(expert_data)}"
        )

    # Cargar configuración
    if exercise_name and not config:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Configuraciones
    if config is None:
        config = {}

    alignment_method = alignment_method or config.get("alignment_method", "centroid")
    reference_landmarks = reference_landmarks or (
        config.get("alignment_landmarks") if alignment_method == "custom" else None
    )
    use_parallel = use_parallel or config.get("use_parallel", False)
    max_workers = config.get("max_workers", max_workers)

    # Diagnósticos
    if diagnostics_path:
        os.makedirs(diagnostics_path, exist_ok=True)

    # Crear DataFrames
    aligned_expert_data = pd.DataFrame(columns=expert_data.columns)
    error_metrics_df = (
        pd.DataFrame(
            index=range(len(user_data)),
            columns=["error_before", "error_after", "improvement", "valid_points"],
        )
        if compute_error
        else None
    )

    # Procesamiento
    if use_parallel and len(user_data) > 10:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Enviar trabajos
            for i in range(len(user_data)):
                future = executor.submit(
                    align_skeleton_frame,
                    expert_data.iloc[i].to_dict(),
                    user_data.iloc[i].to_dict(),
                    alignment_method,
                    reference_landmarks,
                    compute_error,
                )
                future.frame_idx = i
                futures.append(future)

            # Procesar resultados con barra de progreso
            from concurrent.futures import as_completed

            iterator = (
                tqdm(as_completed(futures), total=len(futures), desc="Alineando frames")
                if show_progress
                else as_completed(futures)
            )

            for future in iterator:
                i = future.frame_idx
                try:
                    if compute_error:
                        aligned_frame, metrics = future.result()
                        row_dict = {
                            col: val
                            for col, val in aligned_frame.items()
                            if col in aligned_expert_data.columns
                        }
                        aligned_expert_data.loc[i] = pd.Series(row_dict)

                        if metrics:
                            error_metrics_df.loc[i, "error_before"] = metrics.get(
                                "error_before", np.nan
                            )
                            error_metrics_df.loc[i, "error_after"] = metrics.get(
                                "error_after", np.nan
                            )
                            error_metrics_df.loc[i, "improvement"] = metrics.get(
                                "improvement", np.nan
                            )
                            error_metrics_df.loc[i, "valid_points"] = metrics.get(
                                "valid_points", 0
                            )
                    else:
                        row_dict = {
                            col: val
                            for col, val in future.result().items()
                            if col in aligned_expert_data.columns
                        }
                        aligned_expert_data.loc[i] = pd.Series(row_dict)
                except Exception as e:
                    logger.error(f"Error al procesar frame {i}: {e}")
    else:
        # Procesamiento secuencial
        iterator = (
            tqdm(range(len(user_data)), desc="Alineando frames")
            if show_progress
            else range(len(user_data))
        )

        for i in iterator:
            try:
                if compute_error:
                    aligned_frame, metrics = align_skeleton_frame(
                        expert_data.iloc[i],
                        user_data.iloc[i],
                        alignment_method,
                        reference_landmarks,
                        compute_error,
                    )

                    aligned_expert_data.loc[i] = aligned_frame

                    if metrics:
                        error_metrics_df.loc[i, "error_before"] = metrics.get(
                            "error_before", np.nan
                        )
                        error_metrics_df.loc[i, "error_after"] = metrics.get(
                            "error_after", np.nan
                        )
                        error_metrics_df.loc[i, "improvement"] = metrics.get(
                            "improvement", np.nan
                        )
                        error_metrics_df.loc[i, "valid_points"] = metrics.get(
                            "valid_points", 0
                        )
                else:
                    aligned_expert_data.loc[i] = align_skeleton_frame(
                        expert_data.iloc[i],
                        user_data.iloc[i],
                        alignment_method,
                        reference_landmarks,
                    )
            except Exception as e:
                logger.error(f"Error al procesar frame {i}: {e}")
                aligned_expert_data.loc[i] = expert_data.iloc[i]

    # Conservar columna 'frame'
    if "frame" in expert_data.columns:
        aligned_expert_data["frame"] = expert_data["frame"].values

    # Generar diagnósticos
    if compute_error and diagnostics_path:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(error_metrics_df["improvement"], "g-", alpha=0.7)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.title(
                f"Mejora en alineación por frame - {exercise_name or 'Ejercicio'}"
            )
            plt.xlabel("Frame")
            plt.ylabel("Mejora (%)")
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(
                    diagnostics_path,
                    f"{exercise_name or 'exercise'}_alignment_improvement.png",
                ),
                dpi=100,
            )

            # Guardar métricas
            error_metrics_df.to_csv(
                os.path.join(
                    diagnostics_path,
                    f"{exercise_name or 'exercise'}_alignment_metrics.csv",
                )
            )

            # Resumen
            avg_improvement = error_metrics_df["improvement"].mean()
            logger.info(f"Mejora promedio en alineación: {avg_improvement:.2f}%")

            plt.close()
        except Exception as e:
            logger.error(f"Error al generar diagnósticos: {e}")

    logger.info(f"Alineación completada para {len(aligned_expert_data)} frames")
    return (
        (aligned_expert_data, error_metrics_df)
        if compute_error
        else aligned_expert_data
    )


def visualize_alignment(
    user_landmarks,
    expert_landmarks_before,
    expert_landmarks_after,
    output_path=None,
    exercise_name=None,
    show_plot=True,
    figure_size=(12, 8),
    dpi=100,
):
    """
    Visualiza la alineación de un frame de esqueleto.
    """
    # Extraer coordenadas usando nuestra utilidad
    user_points, _ = extract_landmarks_as_matrix(user_landmarks)
    expert_before_points, _ = extract_landmarks_as_matrix(expert_landmarks_before)
    expert_after_points, _ = extract_landmarks_as_matrix(expert_landmarks_after)

    # Crear figura
    fig = plt.figure(figsize=figure_size, dpi=dpi)

    # Vista 3D
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        user_points[:, 0],
        user_points[:, 1],
        user_points[:, 2],
        c="blue",
        marker="o",
        label="Usuario",
    )
    ax1.scatter(
        expert_before_points[:, 0],
        expert_before_points[:, 1],
        expert_before_points[:, 2],
        c="red",
        marker="x",
        label="Experto (antes)",
    )
    ax1.scatter(
        expert_after_points[:, 0],
        expert_after_points[:, 1],
        expert_after_points[:, 2],
        c="green",
        marker="+",
        label="Experto (después)",
    )
    ax1.set_title("Vista 3D")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # Vista frontal
    ax2 = fig.add_subplot(122)
    ax2.scatter(
        user_points[:, 0], user_points[:, 1], c="blue", marker="o", label="Usuario"
    )
    ax2.scatter(
        expert_before_points[:, 0],
        expert_before_points[:, 1],
        c="red",
        marker="x",
        label="Experto (antes)",
    )
    ax2.scatter(
        expert_after_points[:, 0],
        expert_after_points[:, 1],
        c="green",
        marker="+",
        label="Experto (después)",
    )
    ax2.set_title("Vista Frontal (XY)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Título y ajustes
    fig.suptitle(f"Alineación de Esqueleto - {exercise_name or 'Ejercicio'}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Guardar y mostrar
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Visualización guardada en: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig
