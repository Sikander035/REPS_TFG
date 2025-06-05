"""
Visualizaciones duales de movimiento corporal - WEB-COMPATIBLE VERSION.
FIX para reproducción en navegadores HTML5.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import subprocess
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import logging
import time

# Detectar Docker (solo para info, sin cambiar lógica)
try:
    import platform
except ImportError:
    platform = None

# Configurar matplotlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.matplotlib_config import (
    ThreadSafeMatplotlib,
    ensure_matplotlib_thread_safety,
)

import matplotlib.pyplot as plt

from src.utils.visualization_utils import (
    draw_skeleton,
    extract_frame_ranges,
    save_visualization,
)

from src.config.config_manager import config_manager

logger = logging.getLogger("visualization.dual")
logger.setLevel(logging.INFO)


def get_web_compatible_fourcc():
    """
    Retorna fourcc que genere H.264, NO FMP4.
    CRÍTICO: Evitar FMP4 que no es web-compatible.
    """
    # Detectar entorno Docker
    is_docker = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
    if is_docker:
        logger.info("🐳 Docker detectado - forzando codec H.264 compatible")

    # PRIORIDAD: Codecs que generan H.264, NO FMP4
    codecs_to_try = [
        ("avc1", "H.264 AVC1 (mejor para web)"),
        ("h264", "H.264 directo"),
        ("X264", "X264 encoder"),
        # ÚLTIMO RECURSO: mp4v genera FMP4 (necesita conversión)
        ("mp4v", "MPEG-4 (requiere conversión a H.264)"),
    ]

    for codec, desc in codecs_to_try:
        try:
            logger.info(f"🧪 Probando codec: {codec}")
            fourcc = cv2.VideoWriter_fourcc(*codec)

            # Test con escritura real
            test_file = f"test_{codec}.mp4"
            test_writer = cv2.VideoWriter(test_file, fourcc, 30.0, (320, 240), True)

            if test_writer.isOpened():
                try:
                    import numpy as np

                    test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    test_frame[:] = (100, 100, 100)
                    success = test_writer.write(test_frame)
                    test_writer.release()

                    if (
                        success
                        and os.path.exists(test_file)
                        and os.path.getsize(test_file) > 500
                    ):
                        # VERIFICAR QUE NO GENERE FMP4
                        cap_test = cv2.VideoCapture(test_file)
                        if cap_test.isOpened():
                            test_fourcc = int(cap_test.get(cv2.CAP_PROP_FOURCC))
                            test_fourcc_str = "".join(
                                [chr((test_fourcc >> 8 * i) & 0xFF) for i in range(4)]
                            )
                            cap_test.release()

                            logger.info(
                                f"   📊 Codec {codec} genera FourCC: {test_fourcc_str}"
                            )

                            # RECHAZAR si genera FMP4
                            if test_fourcc_str == "FMP4":
                                logger.warning(
                                    f"   ❌ Codec {codec} genera FMP4 (no web-compatible)"
                                )
                                try:
                                    os.remove(test_file)
                                except:
                                    pass
                                continue
                            else:
                                logger.info(
                                    f"   ✅ Codec {codec} genera {test_fourcc_str} (compatible)"
                                )
                                try:
                                    os.remove(test_file)
                                except:
                                    pass
                                return fourcc, codec

                except Exception as e:
                    logger.warning(f"❌ Codec {codec} error: {e}")
                finally:
                    test_writer.release()
                    try:
                        if os.path.exists(test_file):
                            os.remove(test_file)
                    except:
                        pass
            else:
                test_writer.release()
                logger.warning(f"❌ Codec {codec} no se puede abrir")

        except Exception as e:
            logger.warning(f"❌ Error con codec {codec}: {e}")
            continue

    # Si ningún codec H.264 funciona, usar mp4v pero ADVERTIR
    logger.warning(
        "🚨 ADVERTENCIA: Usando mp4v que genera FMP4 - REQUIERE conversión ffmpeg"
    )
    logger.warning("🔧 ffmpeg DEBE convertir FMP4 → H.264 para compatibilidad web")
    return cv2.VideoWriter_fourcc(*"mp4v"), "mp4v"


def check_ffmpeg_availability() -> bool:
    """
    Verifica si ffmpeg está disponible para post-procesamiento.
    CRÍTICO en Windows donde OpenCV H.264 puede fallar.
    """
    try:
        # Intentar con imageio-ffmpeg primero (más confiable)
        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            result = subprocess.run(
                [ffmpeg_path, "-version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info(f"✅ ffmpeg disponible (imageio): {ffmpeg_path}")
                return True
        except ImportError:
            logger.debug("imageio-ffmpeg no instalado")
        except Exception as e:
            logger.warning(f"Error con imageio-ffmpeg: {e}")

        # Intentar con ffmpeg del sistema
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("✅ ffmpeg disponible (sistema)")
            return True

    except FileNotFoundError:
        logger.error("❌ ffmpeg no encontrado en PATH del sistema")
    except subprocess.TimeoutExpired:
        logger.error("❌ ffmpeg timeout")
    except Exception as e:
        logger.error(f"❌ Error verificando ffmpeg: {e}")

    logger.error("❌ ffmpeg NO disponible")
    logger.error("💡 SOLUCIÓN: pip install imageio-ffmpeg")
    return False


def optimize_video_for_web(input_path: str, output_path: str) -> bool:
    """
    Optimiza video para reproducción web usando ffmpeg.
    CRÍTICO: Convierte FMP4 → H.264 para compatibilidad navegadores.
    """
    try:
        # Intentar usar ffmpeg del entorno virtual
        ffmpeg_cmd = "ffmpeg"
        try:
            import imageio_ffmpeg

            ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
            logger.debug(f"📦 Usando ffmpeg del entorno virtual: {ffmpeg_cmd}")
        except ImportError:
            logger.debug("🔍 imageio-ffmpeg no disponible, usando ffmpeg del sistema")

        # COMANDO CRÍTICO: Forzar conversión FMP4 → H.264
        cmd = [
            ffmpeg_cmd,
            "-i",
            input_path,
            # FORZAR H.264 (crítico para convertir FMP4)
            "-c:v",
            "libx264",  # ✅ OBLIGATORIO: Forzar H.264
            "-profile:v",
            "baseline",  # ✅ Máxima compatibilidad navegadores
            "-level",
            "3.0",  # ✅ Level compatible universal
            "-pix_fmt",
            "yuv420p",  # ✅ Formato de píxel estándar
            # Optimización web específica
            "-movflags",
            "faststart",  # ✅ Metadata al principio (streaming)
            "-preset",
            "fast",  # ✅ Encoding rápido
            "-crf",
            "23",  # ✅ Calidad constante balanceada
            # Asegurar compatibilidad máxima
            "-avoid_negative_ts",
            "make_zero",  # ✅ Evitar timestamps negativos
            "-fflags",
            "+genpts",  # ✅ Generar timestamps si faltan
            # Sin audio para simplificar (los videos de ejercicio no necesitan audio)
            "-an",  # ✅ Sin audio (eliminar complejidad)
            output_path,
            "-y",  # Sobrescribir si existe
        ]

        logger.info("🔧 CONVERSIÓN CRÍTICA: FMP4 → H.264 para compatibilidad web...")
        logger.debug(f"Comando ffmpeg: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutos máximo
        )

        # Verificar que el archivo se creó correctamente
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.info("✅ Conversión FMP4 → H.264 exitosa")

            # VERIFICACIÓN CRÍTICA: Confirmar que ahora es H.264
            try:
                cap_verify = cv2.VideoCapture(output_path)
                if cap_verify.isOpened():
                    verify_fourcc = int(cap_verify.get(cv2.CAP_PROP_FOURCC))
                    verify_fourcc_str = "".join(
                        [chr((verify_fourcc >> 8 * i) & 0xFF) for i in range(4)]
                    )
                    cap_verify.release()

                    logger.info(
                        f"🔍 Video convertido tiene FourCC: {verify_fourcc_str}"
                    )

                    if verify_fourcc_str in ["h264", "avc1", "H264", "AVC1"]:
                        logger.info("🎉 ¡ÉXITO! Video ahora es H.264 web-compatible")
                        return True
                    else:
                        logger.warning(
                            f"⚠️ Conversión parcial: FourCC es {verify_fourcc_str}, no h264"
                        )
                        # Aún puede funcionar si ffmpeg hizo la conversión correctamente
                        return True

            except Exception as verify_error:
                logger.warning(
                    f"⚠️ No se pudo verificar FourCC del video convertido: {verify_error}"
                )
                # Si ffmpeg terminó exitosamente, asumir que está bien
                return True

            return True
        else:
            logger.error("❌ ffmpeg terminó pero el archivo de salida es inválido")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ ffmpeg timeout (>5 min) - video muy grande o sistema lento")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ ffmpeg falló (código {e.returncode})")
        logger.error(f"   stderr: {e.stderr}")
        logger.error(
            "💡 Posibles causas: libx264 no disponible, video corrupto, falta espacio"
        )
        return False
    except FileNotFoundError:
        logger.error("❌ ffmpeg no encontrado")
        logger.error("💡 Solución: pip install imageio-ffmpeg")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado en conversión: {e}")
        return False


def verify_video_web_compatibility(video_path: str) -> Dict[str, Any]:
    """
    Verifica si un video es compatible con navegadores web.
    Retorna información detallada del video.
    """
    try:
        # Usar ffprobe para analizar el video
        ffprobe_cmd = "ffprobe"
        try:
            import imageio_ffmpeg

            # ffprobe suele estar en el mismo directorio que ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            ffprobe_cmd = ffmpeg_path.replace("ffmpeg", "ffprobe")
        except ImportError:
            pass

        cmd = [
            ffprobe_cmd,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json

        info = json.loads(result.stdout)

        # Extraer información relevante
        video_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if video_stream:
            codec = video_stream.get("codec_name", "unknown")
            profile = video_stream.get("profile", "unknown")
            pix_fmt = video_stream.get("pix_fmt", "unknown")

            # Verificar compatibilidad web
            is_web_compatible = (
                codec in ["h264", "avc"]
                and pix_fmt in ["yuv420p", "yuvj420p"]
                and profile in ["Baseline", "Main", "High"]
            )

            return {
                "compatible": is_web_compatible,
                "codec": codec,
                "profile": profile,
                "pix_fmt": pix_fmt,
                "duration": float(info.get("format", {}).get("duration", 0)),
                "size": int(info.get("format", {}).get("size", 0)),
                "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
            }

    except Exception as e:
        logger.warning(f"⚠️ No se pudo verificar compatibilidad: {e}")

    # Fallback: usar OpenCV para verificación básica
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            return {
                "compatible": None,  # No se puede determinar
                "codec": "unknown",
                "frame_count": frame_count,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "duration": frame_count / fps if fps > 0 else 0,
                "size": (
                    os.path.getsize(video_path) if os.path.exists(video_path) else 0
                ),
            }
    except Exception as e:
        logger.error(f"❌ Error verificando video con OpenCV: {e}")

    return {"compatible": False, "error": "No se pudo analizar el video"}


def generate_dual_skeleton_video(
    original_video_path: str,
    user_data: pd.DataFrame,
    expert_data: pd.DataFrame,
    output_video_path: str,
    config_path: str,
    original_user_data: pd.DataFrame = None,
    exercise_frame_range: tuple = None,
    frame_range: tuple = None,
    fps_factor: float = 1.0,
) -> str:
    """
    Genera video con esqueletos del usuario y experto superpuestos.
    WEB-COMPATIBLE VERSION con H.264 encoding.
    """
    # Verificaciones de entrada obligatorias
    if not original_video_path:
        raise ValueError("original_video_path es obligatorio")
    if user_data is None or user_data.empty:
        raise ValueError("user_data es obligatorio y no puede estar vacío")
    if expert_data is None or expert_data.empty:
        raise ValueError("expert_data es obligatorio y no puede estar vacío")
    if not output_video_path:
        raise ValueError("output_video_path es obligatorio")
    if not config_path:
        raise ValueError("config_path es obligatorio")

    if len(user_data) != len(expert_data):
        raise ValueError(
            f"Los datos tienen longitudes diferentes: usuario={len(user_data)}, experto={len(expert_data)}"
        )

    if not isinstance(fps_factor, (int, float)) or fps_factor <= 0:
        raise ValueError("fps_factor debe ser un número positivo")

    # Cargar configuración usando el config_manager singleton
    try:
        viz_config = config_manager.get_global_visualization_config(config_path)
        connections = config_manager.get_global_connections(config_path)
    except Exception as e:
        raise RuntimeError(f"Error cargando configuración: {e}")

    # Variables de OpenCV para cleanup
    cap = None
    out = None

    try:
        # ✅ VERIFICAR ffmpeg disponibilidad (crítico en Windows)
        ffmpeg_available = check_ffmpeg_availability()
        if not ffmpeg_available:
            logger.warning(
                "⚠️ ffmpeg no disponible - video puede no ser compatible con navegadores"
            )
            logger.warning("💡 Instala con: pip install imageio-ffmpeg")

        # Obtener parámetros de configuración
        user_color = tuple(viz_config["user_color"])
        expert_color = tuple(viz_config["expert_color"])
        user_alpha = viz_config["user_alpha"]
        expert_alpha = viz_config["expert_alpha"]
        user_thickness = viz_config["user_thickness"]
        expert_thickness = viz_config["expert_thickness"]
        show_progress = viz_config["show_progress"]
        text_info = viz_config["text_info"]
        show_labels = viz_config["show_labels"]
        resize_factor = viz_config["resize_factor"]
        highlight_landmarks = viz_config.get("highlight_landmarks")

        # Abrir video original
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {original_video_path}")

        # Obtener propiedades del video
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            raise RuntimeError("No se pudo obtener FPS del video")

        fps = original_fps * fps_factor
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)

        # Asegurar dimensiones pares (requerido por algunos encoders)
        width = width + (width % 2)
        height = height + (height % 2)

        if width <= 0 or height <= 0:
            raise RuntimeError("Dimensiones de video inválidas")

        # Crear directorio de salida
        output_dir = Path(output_video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # ✅ MODIFICADO: Crear video temporal primero
        temp_output = output_video_path.replace(".mp4", "_temp_opencv.mp4")

        # ✅ Codec selection CONSERVADOR - solo mejoras mínimas
        fourcc, codec_name = get_web_compatible_fourcc()

        # Configurar escritor de video (MANTENER LÓGICA ORIGINAL)
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height), True)

        if not out.isOpened():
            # Fallback simple si falla
            logger.warning(f"❌ Codec {codec_name} falló, probando mp4v...")
            fourcc_fallback = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                temp_output, fourcc_fallback, fps, (width, height), True
            )
            codec_name = "mp4v"

            if not out.isOpened():
                raise RuntimeError(
                    "VideoWriter no se pudo inicializar con ningún codec"
                )

        logger.info(f"🎬 VideoWriter inicializado con codec: {codec_name}")

        logger.info(f"🎥 Generando video temporal con codec: {codec_name}")

        # Mapear frames procesados a frames originales
        try:
            original_frames = extract_frame_ranges(
                original_user_data if original_user_data is not None else user_data,
                user_data,
                exercise_frame_range,
            )
        except Exception as e:
            raise RuntimeError(f"Error mapeando frames: {e}")

        # Aplicar rango si se especifica
        if frame_range is not None:
            if not isinstance(frame_range, (tuple, list)) or len(frame_range) != 2:
                raise ValueError("frame_range debe ser una tupla de 2 elementos")

            start_idx, end_idx = frame_range
            if not isinstance(start_idx, int) or not isinstance(end_idx, int):
                raise ValueError("frame_range debe contener enteros")

            start_idx = max(0, start_idx)
            end_idx = min(len(original_frames), end_idx)

            if start_idx >= end_idx:
                raise ValueError(
                    "frame_range inválido: start_idx debe ser menor que end_idx"
                )

            original_frames = original_frames[start_idx:end_idx]

        # Número total de frames a procesar
        total_frames = len(original_frames)
        if total_frames == 0:
            raise ValueError("No hay frames para procesar")

        logger.info(f"Generando video con {total_frames} frames ({fps:.1f} FPS)")

        # Iniciar tiempo para cálculos de ETA
        start_time = time.time()
        last_update = start_time
        update_interval = 2.0

        # Procesar frames
        frame_iterator = (
            tqdm(enumerate(original_frames), total=total_frames, desc="Generando video")
            if show_progress
            else enumerate(original_frames)
        )

        for i, orig_frame_idx in frame_iterator:
            # Actualizar progreso ocasionalmente
            current_time = time.time()
            if current_time - last_update >= update_interval and not show_progress:
                elapsed = current_time - start_time
                if i > 10:
                    frames_remaining = total_frames - i
                    time_per_frame = elapsed / i
                    eta = frames_remaining * time_per_frame
                    logger.info(
                        f"Progreso: {i/total_frames*100:.1f}% - ETA: {int(eta//60)}:{int(eta%60):02d}"
                    )
                last_update = current_time

            # Leer frame original
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"No se pudo leer frame {orig_frame_idx}")
                continue

            # Redimensionar si es necesario
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (width, height))

            # Obtener landmarks para este frame
            if i < len(user_data) and i < len(expert_data):
                user_landmarks = user_data.iloc[i]
                expert_landmarks = expert_data.iloc[i]

                # Dibujar esqueletos (experto primero para que esté detrás)
                frame = draw_skeleton(
                    frame,
                    expert_landmarks,
                    color=expert_color,
                    thickness=expert_thickness,
                    alpha=expert_alpha,
                    connections=connections,
                    highlight_landmarks=highlight_landmarks,
                    show_labels=show_labels,
                )

                frame = draw_skeleton(
                    frame,
                    user_landmarks,
                    color=user_color,
                    thickness=user_thickness,
                    alpha=user_alpha,
                    connections=connections,
                    highlight_landmarks=highlight_landmarks,
                    show_labels=show_labels,
                )

                # Añadir información textual
                if text_info:
                    cv2.putText(
                        frame,
                        f"Frame: {orig_frame_idx}/{i}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "Usuario",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        user_color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        "Experto",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        expert_color,
                        2,
                    )

            # Escribir frame al video
            out.write(frame)

        # CLEANUP de video writer ANTES de post-procesamiento
        out.release()
        out = None
        cap.release()
        cap = None
        logger.debug("🧹 Recursos OpenCV liberados")

        # ✅ POST-PROCESAMIENTO CRÍTICO: FMP4 → H.264 conversión
        try:
            if ffmpeg_available:
                logger.info(
                    "🎯 PASO CRÍTICO: Convertir FMP4 → H.264 para navegadores..."
                )

                # Verificar qué FourCC generó OpenCV
                cap_check = cv2.VideoCapture(temp_output)
                if cap_check.isOpened():
                    opencv_fourcc = int(cap_check.get(cv2.CAP_PROP_FOURCC))
                    opencv_fourcc_str = "".join(
                        [chr((opencv_fourcc >> 8 * i) & 0xFF) for i in range(4)]
                    )
                    cap_check.release()

                    logger.info(f"📊 OpenCV generó FourCC: {opencv_fourcc_str}")

                    if opencv_fourcc_str == "FMP4":
                        logger.warning(
                            "🚨 DETECTADO: FMP4 - NO compatible con navegadores"
                        )
                        logger.info("🔧 Convirtiendo FMP4 → H.264 con ffmpeg...")
                    elif opencv_fourcc_str in ["h264", "avc1", "H264", "AVC1"]:
                        logger.info("✅ OpenCV ya generó H.264 - optimizando para web")
                    else:
                        logger.info(
                            f"🔄 Convirtiendo {opencv_fourcc_str} → H.264 para máxima compatibilidad"
                        )

                optimization_success = optimize_video_for_web(
                    temp_output, output_video_path
                )

                if optimization_success:
                    # Limpiar archivo temporal
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                        logger.debug(f"🧹 Archivo temporal eliminado: {temp_output}")

                    logger.info("🎉 ÉXITO: Video optimizado como H.264 web-compatible")
                else:
                    # Fallback crítico
                    logger.error("❌ FALLO CRÍTICO: Conversión H.264 falló")
                    logger.warning("🚨 Video puede NO reproducirse en navegadores")

                    if os.path.exists(temp_output):
                        if os.path.exists(output_video_path):
                            os.remove(output_video_path)
                        os.rename(temp_output, output_video_path)
                        logger.debug(
                            f"📁 Usando video OpenCV como último recurso: {temp_output} → {output_video_path}"
                        )
            else:
                # Sin ffmpeg - PROBLEMA CRÍTICO
                logger.error("🚨 PROBLEMA CRÍTICO: ffmpeg no disponible")
                logger.error("🚨 Video será FMP4 - NO se reproducirá en navegadores")
                logger.error("💡 SOLUCIÓN: pip install imageio-ffmpeg")

                if os.path.exists(temp_output):
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)
                    os.rename(temp_output, output_video_path)
                    logger.debug(
                        f"📁 Video FMP4 copiado (sin conversión): {temp_output} → {output_video_path}"
                    )

        except Exception as e:
            logger.error(f"❌ Error en conversión FMP4 → H.264: {e}")
            # Fallback: usar video original
            if os.path.exists(temp_output):
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)
                os.rename(temp_output, output_video_path)
                logger.debug(f"📁 Fallback: {temp_output} → {output_video_path}")

        # ✅ VERIFICAR compatibilidad final
        compatibility_info = verify_video_web_compatibility(output_video_path)
        logger.info(f"📊 Compatibilidad web: {compatibility_info}")

        if compatibility_info.get("compatible") is False:
            logger.warning(
                "⚠️ El video generado puede no ser compatible con todos los navegadores"
            )

        # Calcular tiempo total
        total_time = time.time() - start_time
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
        logger.info(
            f"Video generado exitosamente: {output_video_path} "
            f"({total_frames} frames, {file_size:.1f}MB en {total_time:.1f}s)"
        )

        return output_video_path

    except Exception as e:
        logger.error(f"Error generando video: {e}")
        raise RuntimeError(f"Error generando video: {e}")

    finally:
        # CLEANUP CRÍTICO: Liberar recursos OpenCV
        if cap is not None:
            cap.release()
            logger.debug("🧹 VideoCapture liberado")

        if out is not None:
            out.release()
            logger.debug("🧹 VideoWriter liberado")

        # Forzar liberación de recursos de OpenCV
        cv2.destroyAllWindows()


def visualize_frame_dual_skeletons(
    original_image: Union[np.ndarray, str],
    user_frame_data: pd.Series,
    expert_frame_data: pd.Series,
    config_path: str,
    save_path: str = None,
    show_image: bool = False,
    title: str = "Comparación de Esqueletos",
) -> np.ndarray:
    """
    Visualiza un frame con esqueletos superpuestos.
    THREAD-SAFE VERSION con cleanup automático.
    """
    # Verificaciones de entrada obligatorias
    if original_image is None:
        raise ValueError("original_image es obligatorio")
    if user_frame_data is None or user_frame_data.empty:
        raise ValueError("user_frame_data es obligatorio y no puede estar vacío")
    if expert_frame_data is None or expert_frame_data.empty:
        raise ValueError("expert_frame_data es obligatorio y no puede estar vacío")
    if not config_path:
        raise ValueError("config_path es obligatorio")

    # Cargar configuración usando el config_manager singleton
    try:
        viz_config = config_manager.get_global_visualization_config(config_path)
        connections = config_manager.get_global_connections(config_path)
    except Exception as e:
        raise RuntimeError(f"Error cargando configuración: {e}")

    # Obtener parámetros de configuración
    user_color = tuple(viz_config["user_color"])
    expert_color = tuple(viz_config["expert_color"])
    user_alpha = viz_config["user_alpha"]
    expert_alpha = viz_config["expert_alpha"]
    user_thickness = viz_config["user_thickness"]
    expert_thickness = viz_config["expert_thickness"]
    show_labels = viz_config["show_labels"]
    highlight_landmarks = viz_config.get("highlight_landmarks")

    # Cargar la imagen si es una ruta
    if isinstance(original_image, str):
        image = cv2.imread(original_image)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {original_image}")
    else:
        image = original_image.copy()

    try:
        # Dibujar esqueletos (experto primero para que esté detrás)
        image = draw_skeleton(
            image,
            expert_frame_data,
            color=expert_color,
            thickness=expert_thickness,
            alpha=expert_alpha,
            connections=connections,
            highlight_landmarks=highlight_landmarks,
            show_labels=show_labels,
        )

        image = draw_skeleton(
            image,
            user_frame_data,
            color=user_color,
            thickness=user_thickness,
            alpha=user_alpha,
            connections=connections,
            highlight_landmarks=highlight_landmarks,
            show_labels=show_labels,
        )

        # Añadir leyenda
        cv2.putText(
            image, "Usuario", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, user_color, 2
        )
        cv2.putText(
            image, "Experto", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, expert_color, 2
        )

        # Mostrar la imagen (thread-safe)
        if show_image:
            with ThreadSafeMatplotlib():
                plt.figure(figsize=(12, 10))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title(title)
                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=100, bbox_inches="tight")

                plt.show()

        # Guardar la imagen si se especifica ruta (sin mostrar)
        elif save_path:
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, image)
                logger.info(f"Imagen guardada: {save_path}")
            except Exception as e:
                raise RuntimeError(f"Error guardando imagen: {e}")

        return image

    except Exception as e:
        raise RuntimeError(f"Error dibujando esqueletos: {e}")
