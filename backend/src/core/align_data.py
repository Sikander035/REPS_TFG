import pandas as pd
import numpy as np
import logging
import sys
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.config_manager import load_exercise_config

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def align_skeleton_frame(
    expert_landmarks,
    user_landmarks,
    alignment_method="centroid",
    reference_landmarks=None,
    compute_error=False,
):
    """
    Alinea un frame del esqueleto del experto con el del usuario.

    Args:
        expert_landmarks: Serie/diccionario con landmarks del experto
        user_landmarks: Serie/diccionario con landmarks del usuario
        alignment_method: Método de alineación ('centroid', 'shoulders', 'hips', 'custom')
        reference_landmarks: Lista de landmarks a usar para alineación con método 'custom'
        compute_error: Si es True, calcula y devuelve el error de alineación

    Returns:
        Serie/diccionario con landmarks del experto alineados, y opcionalmente error
    """
    # Crear una copia para no modificar los originales
    aligned_expert = (
        expert_landmarks.copy()
        if isinstance(expert_landmarks, pd.Series)
        else expert_landmarks.copy()
    )

    # Determinar landmarks de referencia según el método
    if alignment_method == "shoulders":
        reference_points = ["landmark_left_shoulder", "landmark_right_shoulder"]
    elif alignment_method == "hips":
        reference_points = ["landmark_left_hip", "landmark_right_hip"]
    elif alignment_method == "torso":
        reference_points = [
            "landmark_left_shoulder",
            "landmark_right_shoulder",
            "landmark_left_hip",
            "landmark_right_hip",
        ]
    elif alignment_method == "custom" and reference_landmarks:
        reference_points = reference_landmarks
    else:  # 'centroid' (default) o fallback
        # Usar todos los landmarks disponibles
        reference_points = []
        for col in (
            expert_landmarks.index
            if isinstance(expert_landmarks, pd.Series)
            else expert_landmarks.keys()
        ):
            if col.endswith("_x"):
                landmark = col[:-2]
                if f"{landmark}_y" in (
                    expert_landmarks.index
                    if isinstance(expert_landmarks, pd.Series)
                    else expert_landmarks.keys()
                ) and f"{landmark}_z" in (
                    expert_landmarks.index
                    if isinstance(expert_landmarks, pd.Series)
                    else expert_landmarks.keys()
                ):
                    reference_points.append(landmark)

    # Calcular centroides para usuario y experto
    user_centroid = np.array([0.0, 0.0, 0.0])
    expert_centroid = np.array([0.0, 0.0, 0.0])
    valid_points = 0

    for point in reference_points:
        # Verificar que el landmark existe en ambos datasets
        if all(
            f"{point}_{axis}"
            in (
                expert_landmarks.index
                if isinstance(expert_landmarks, pd.Series)
                else expert_landmarks.keys()
            )
            and f"{point}_{axis}"
            in (
                user_landmarks.index
                if isinstance(user_landmarks, pd.Series)
                else user_landmarks.keys()
            )
            for axis in ["x", "y", "z"]
        ):

            # Extraer coordenadas
            if isinstance(expert_landmarks, pd.Series):
                expert_point = np.array(
                    [
                        expert_landmarks[f"{point}_x"],
                        expert_landmarks[f"{point}_y"],
                        expert_landmarks[f"{point}_z"],
                    ]
                )
                user_point = np.array(
                    [
                        user_landmarks[f"{point}_x"],
                        user_landmarks[f"{point}_y"],
                        user_landmarks[f"{point}_z"],
                    ]
                )
            else:
                expert_point = np.array(
                    [
                        expert_landmarks[f"{point}_x"],
                        expert_landmarks[f"{point}_y"],
                        expert_landmarks[f"{point}_z"],
                    ]
                )
                user_point = np.array(
                    [
                        user_landmarks[f"{point}_x"],
                        user_landmarks[f"{point}_y"],
                        user_landmarks[f"{point}_z"],
                    ]
                )

            # Verificar que no hay NaN
            if not np.isnan(expert_point).any() and not np.isnan(user_point).any():
                expert_centroid += expert_point
                user_centroid += user_point
                valid_points += 1

    # Si no hay puntos válidos, devolver sin cambios
    if valid_points == 0:
        logger.warning("No se encontraron landmarks válidos para alineación")
        return (
            (aligned_expert, {"error": np.nan, "valid_points": 0})
            if compute_error
            else aligned_expert
        )

    # Calcular centroides promedio
    expert_centroid /= valid_points
    user_centroid /= valid_points

    # Calcular vector de desplazamiento
    displacement = user_centroid - expert_centroid

    # Aplicar desplazamiento a todos los landmarks del experto
    error_before = 0
    error_after = 0

    for key in (
        expert_landmarks.index
        if isinstance(expert_landmarks, pd.Series)
        else expert_landmarks.keys()
    ):
        if key.endswith("_x"):
            landmark = key[:-2]
            if f"{landmark}_y" in (
                expert_landmarks.index
                if isinstance(expert_landmarks, pd.Series)
                else expert_landmarks.keys()
            ) and f"{landmark}_z" in (
                expert_landmarks.index
                if isinstance(expert_landmarks, pd.Series)
                else expert_landmarks.keys()
            ):

                # Calcular error antes de alineación (solo para landmarks de referencia)
                if compute_error and landmark in reference_points:
                    if isinstance(expert_landmarks, pd.Series):
                        expert_point = np.array(
                            [
                                expert_landmarks[f"{landmark}_x"],
                                expert_landmarks[f"{landmark}_y"],
                                expert_landmarks[f"{landmark}_z"],
                            ]
                        )
                        user_point = np.array(
                            [
                                user_landmarks[f"{landmark}_x"],
                                user_landmarks[f"{landmark}_y"],
                                user_landmarks[f"{landmark}_z"],
                            ]
                        )
                    else:
                        expert_point = np.array(
                            [
                                expert_landmarks[f"{landmark}_x"],
                                expert_landmarks[f"{landmark}_y"],
                                expert_landmarks[f"{landmark}_z"],
                            ]
                        )
                        user_point = np.array(
                            [
                                user_landmarks[f"{landmark}_x"],
                                user_landmarks[f"{landmark}_y"],
                                user_landmarks[f"{landmark}_z"],
                            ]
                        )

                    error_before += np.sum((expert_point - user_point) ** 2)

                # Aplicar desplazamiento
                if isinstance(aligned_expert, pd.Series):
                    aligned_expert[f"{landmark}_x"] += displacement[0]
                    aligned_expert[f"{landmark}_y"] += displacement[1]
                    aligned_expert[f"{landmark}_z"] += displacement[2]
                else:
                    aligned_expert[f"{landmark}_x"] += displacement[0]
                    aligned_expert[f"{landmark}_y"] += displacement[1]
                    aligned_expert[f"{landmark}_z"] += displacement[2]

                # Calcular error después de alineación (solo para landmarks de referencia)
                if compute_error and landmark in reference_points:
                    if isinstance(aligned_expert, pd.Series):
                        aligned_point = np.array(
                            [
                                aligned_expert[f"{landmark}_x"],
                                aligned_expert[f"{landmark}_y"],
                                aligned_expert[f"{landmark}_z"],
                            ]
                        )
                    else:
                        aligned_point = np.array(
                            [
                                aligned_expert[f"{landmark}_x"],
                                aligned_expert[f"{landmark}_y"],
                                aligned_expert[f"{landmark}_z"],
                            ]
                        )

                    error_after += np.sum((aligned_point - user_point) ** 2)

    # Calcular error promedio (raíz del error cuadrático medio)
    if compute_error and valid_points > 0:
        error_before = np.sqrt(error_before / valid_points)
        error_after = np.sqrt(error_after / valid_points)
        improvement = (
            ((error_before - error_after) / error_before * 100)
            if error_before > 0
            else 0
        )

        error_metrics = {
            "error_before": error_before,
            "error_after": error_after,
            "improvement": improvement,
            "valid_points": valid_points,
            "displacement": displacement,
        }

        return aligned_expert, error_metrics

    return aligned_expert


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

    Args:
        user_data: DataFrame con datos del usuario
        expert_data: DataFrame con datos del experto (normalizados)
        config: Configuración personalizada (opcional)
        exercise_name: Nombre del ejercicio para cargar configuración (opcional)
        config_path: Ruta al archivo de configuración
        alignment_method: Método de alineación ('centroid', 'shoulders', 'hips', 'custom')
        reference_landmarks: Lista de landmarks a usar para alineación con método 'custom'
        use_parallel: Si es True, procesa frames en paralelo
        max_workers: Número máximo de workers para procesamiento paralelo
        compute_error: Si es True, calcula y devuelve métricas de error
        show_progress: Mostrar barra de progreso
        diagnostics_path: Directorio para guardar diagnósticos

    Returns:
        DataFrame con los datos del experto alineados, y opcionalmente métricas de error
    """
    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los DataFrames deben tener la misma longitud. Usuario: {len(user_data)}, Experto: {len(expert_data)}"
        )

    # Cargar configuración si se proporciona un nombre de ejercicio
    if exercise_name and not config:
        try:
            exercise_config = load_exercise_config(exercise_name, config_path)
            config = exercise_config.get("sync_config", {})
            logger.info(f"Configuración cargada para ejercicio: {exercise_name}")
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {e}")
            config = {}

    # Si config sigue siendo None, inicializar como diccionario vacío
    if config is None:
        config = {}

    # Determinar método de alineación
    if alignment_method is None:
        alignment_method = config.get("alignment_method", "centroid")

    # Determinar landmarks de referencia
    if reference_landmarks is None and alignment_method == "custom":
        reference_landmarks = config.get("alignment_landmarks", None)

    logger.info(f"Alineando esqueletos usando método: {alignment_method}")

    # Decidir si usar paralelismo
    use_parallel = use_parallel or config.get("use_parallel", False)
    max_workers = config.get("max_workers", max_workers)

    # Para diagnósticos
    if diagnostics_path:
        os.makedirs(diagnostics_path, exist_ok=True)

    # Crear DataFrame para resultado
    aligned_expert_data = pd.DataFrame(columns=expert_data.columns)

    # DataFrame para métricas de error si se solicita
    error_metrics_df = None
    if compute_error:
        error_metrics_df = pd.DataFrame(
            index=range(len(user_data)),
            columns=["error_before", "error_after", "improvement", "valid_points"],
        )

    # Tiempo de inicio para estimación
    start_time = time.time()

    # Procesar frames
    if use_parallel and len(user_data) > 10:
        logger.info(
            f"Procesando {len(user_data)} frames en paralelo con {max_workers} workers"
        )

        # Lista para almacenar resultados
        results = [None] * len(user_data)
        error_results = [None] * len(user_data) if compute_error else None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Enviar trabajos al pool
            for i in range(len(user_data)):
                futures.append(
                    executor.submit(
                        align_skeleton_frame,
                        expert_data.iloc[
                            i
                        ].to_dict(),  # Convertir a dict para serialización
                        user_data.iloc[i].to_dict(),
                        alignment_method,
                        reference_landmarks,
                        compute_error,
                    )
                )

            # Usar tqdm para mostrar progreso
            iterator = (
                tqdm(as_completed(futures), total=len(futures), desc="Alineando frames")
                if show_progress
                else as_completed(futures)
            )

            # Procesar resultados a medida que se completan
            for i, future in enumerate(iterator):
                try:
                    if compute_error:
                        aligned_frame, error_metrics = future.result()
                        results[future.frame_idx] = aligned_frame
                        error_results[future.frame_idx] = error_metrics
                    else:
                        results[future.frame_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error al procesar frame: {e}")

        # Convertir resultados a DataFrame
        aligned_expert_data = pd.DataFrame(results, columns=expert_data.columns)

        # Procesar métricas de error si se solicitaron
        if compute_error and error_results:
            for i, metrics in enumerate(error_results):
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
        # Procesamiento secuencial
        logger.info(f"Procesando {len(user_data)} frames secuencialmente")

        # Crear una lista vacía para almacenar los frames procesados
        processed_frames = []

        # Usar tqdm para mostrar progreso
        iterator = (
            tqdm(range(len(user_data)), desc="Alineando frames")
            if show_progress
            else range(len(user_data))
        )

        # Estimación de tiempo
        frames_processed = 0

        for i in iterator:
            try:
                if compute_error:
                    aligned_frame, error_metrics = align_skeleton_frame(
                        expert_data.iloc[i],
                        user_data.iloc[i],
                        alignment_method,
                        reference_landmarks,
                        compute_error,
                    )
                    processed_frames.append(aligned_frame)

                    # Guardar métricas de error
                    error_metrics_df.loc[i, "error_before"] = error_metrics.get(
                        "error_before", np.nan
                    )
                    error_metrics_df.loc[i, "error_after"] = error_metrics.get(
                        "error_after", np.nan
                    )
                    error_metrics_df.loc[i, "improvement"] = error_metrics.get(
                        "improvement", np.nan
                    )
                    error_metrics_df.loc[i, "valid_points"] = error_metrics.get(
                        "valid_points", 0
                    )
                else:
                    aligned_frame = align_skeleton_frame(
                        expert_data.iloc[i],
                        user_data.iloc[i],
                        alignment_method,
                        reference_landmarks,
                    )
                    processed_frames.append(aligned_frame)

                # Actualizar contador para estimación de tiempo
                frames_processed += 1

                # Mostrar estimación de tiempo cada 100 frames
                if frames_processed % 100 == 0 and show_progress:
                    elapsed = time.time() - start_time
                    frames_per_second = frames_processed / elapsed
                    remaining_frames = len(user_data) - frames_processed
                    eta_seconds = remaining_frames / frames_per_second

                    logger.info(
                        f"Procesados {frames_processed}/{len(user_data)} frames. "
                        f"Velocidad: {frames_per_second:.2f} fps. "
                        f"ETA: {eta_seconds/60:.1f} minutos"
                    )

            except Exception as e:
                logger.error(f"Error al procesar frame {i}: {e}")
                # Usar frame original en caso de error
                processed_frames.append(expert_data.iloc[i])

        # Convertir la lista de frames procesados a DataFrame
        aligned_expert_data = pd.DataFrame(processed_frames)

    # Conservar columna 'frame' si existe
    if "frame" in expert_data.columns:
        aligned_expert_data["frame"] = expert_data["frame"].values

    # Generar diagnósticos si se solicitan
    if compute_error and diagnostics_path:
        try:
            # Gráfico de mejora por frame
            plt.figure(figsize=(12, 6))
            plt.plot(error_metrics_df["improvement"], "g-", alpha=0.7)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.title(
                f"Mejora en alineación por frame - {exercise_name or 'Ejercicio'}"
            )
            plt.xlabel("Frame")
            plt.ylabel("Mejora (%)")
            plt.grid(True, alpha=0.3)

            # Guardar gráfico
            plt.savefig(
                os.path.join(
                    diagnostics_path,
                    f"{exercise_name or 'exercise'}_alignment_improvement.png",
                ),
                dpi=100,
            )

            # Guardar métricas en CSV
            error_metrics_df.to_csv(
                os.path.join(
                    diagnostics_path,
                    f"{exercise_name or 'exercise'}_alignment_metrics.csv",
                )
            )

            # Mostrar resumen
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

    Args:
        user_landmarks: Serie/diccionario con landmarks del usuario
        expert_landmarks_before: Serie/diccionario con landmarks del experto antes de alinear
        expert_landmarks_after: Serie/diccionario con landmarks del experto después de alinear
        output_path: Ruta para guardar la visualización
        exercise_name: Nombre del ejercicio para el título
        show_plot: Si es True, muestra el gráfico
        figure_size: Tamaño de la figura en pulgadas
        dpi: Resolución de la imagen

    Returns:
        Figure de matplotlib
    """

    # Extraer coordenadas
    def extract_coordinates(landmarks):
        x, y, z = [], [], []

        if isinstance(landmarks, pd.Series):
            for col in landmarks.index:
                if col.endswith("_x"):
                    joint = col.rsplit("_", 1)[0]
                    if (
                        f"{joint}_y" in landmarks.index
                        and f"{joint}_z" in landmarks.index
                    ):
                        x.append(landmarks[f"{joint}_x"])
                        y.append(landmarks[f"{joint}_y"])
                        z.append(landmarks[f"{joint}_z"])
        else:
            for col in landmarks.keys():
                if col.endswith("_x"):
                    joint = col.rsplit("_", 1)[0]
                    if f"{joint}_y" in landmarks and f"{joint}_z" in landmarks:
                        x.append(landmarks[f"{joint}_x"])
                        y.append(landmarks[f"{joint}_y"])
                        z.append(landmarks[f"{joint}_z"])

        return np.array(x), np.array(y), np.array(z)

    # Extraer coordenadas para cada conjunto
    user_x, user_y, user_z = extract_coordinates(user_landmarks)
    expert_before_x, expert_before_y, expert_before_z = extract_coordinates(
        expert_landmarks_before
    )
    expert_after_x, expert_after_y, expert_after_z = extract_coordinates(
        expert_landmarks_after
    )

    # Crear figura con dos subplots (vistas diferentes)
    fig = plt.figure(figsize=figure_size, dpi=dpi)

    # Vista superior (XY)
    ax1 = fig.add_subplot(121)
    ax1.scatter(user_x, user_y, c="blue", marker="o", s=50, label="Usuario")
    ax1.scatter(
        expert_before_x,
        expert_before_y,
        c="red",
        marker="x",
        s=50,
        label="Experto (antes)",
    )
    ax1.scatter(
        expert_after_x,
        expert_after_y,
        c="green",
        marker="+",
        s=50,
        label="Experto (después)",
    )
    ax1.set_title("Vista Superior (XY)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Vista lateral (XZ)
    ax2 = fig.add_subplot(122)
    ax2.scatter(user_x, user_z, c="blue", marker="o", s=50, label="Usuario")
    ax2.scatter(
        expert_before_x,
        expert_before_z,
        c="red",
        marker="x",
        s=50,
        label="Experto (antes)",
    )
    ax2.scatter(
        expert_after_x,
        expert_after_z,
        c="green",
        marker="+",
        s=50,
        label="Experto (después)",
    )
    ax2.set_title("Vista Lateral (XZ)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Título general
    fig.suptitle(f"Alineación de Esqueleto - {exercise_name or 'Ejercicio'}")

    # Añadir ajuste automático
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Dejar espacio para el título

    # Guardar si se proporciona una ruta
    if output_path:
        # Crear directorio si no existe
        directory = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(directory, exist_ok=True)

        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Visualización guardada en: {output_path}")

    # Mostrar si se solicita
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


# Función legacy para mantener compatibilidad
def align_skeletons_dataframe_legacy(user_data, expert_data):
    """
    Implementación original para compatibilidad.
    """
    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los DataFrames deben tener la misma longitud. Usuario: {len(user_data)}, Experto: {len(expert_data)}"
        )

    # Crear DataFrame para resultado
    aligned_expert_data = pd.DataFrame(columns=expert_data.columns)

    # Procesar cada frame
    for i in tqdm(range(len(user_data)), desc="Alineando frames"):
        user_frame = user_data.iloc[i].to_dict()
        expert_frame = expert_data.iloc[i].to_dict()

        aligned_frame = align_skeleton_frame(expert_frame, user_frame)
        aligned_expert_data = pd.concat(
            [aligned_expert_data, pd.DataFrame([aligned_frame])], ignore_index=True
        )

    # Conservar columna 'frame' si existe
    if "frame" in expert_data.columns:
        aligned_expert_data["frame"] = expert_data["frame"].values

    return aligned_expert_data
