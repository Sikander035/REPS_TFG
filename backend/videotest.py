#!/usr/bin/env python3
"""
Comparador rápido de videos - versión simplificada.
Uso: python quick_video_compare.py video1.mp4 video2.mp4
"""

import os
import sys
import subprocess
import json
import cv2


def get_video_info_quick(video_path):
    """Obtiene información rápida de un video usando ffprobe y OpenCV."""
    print(f"\n🔍 {os.path.basename(video_path)}")

    info = {
        "file": os.path.basename(video_path),
        "size_mb": round(os.path.getsize(video_path) / 1024 / 1024, 1),
        "ffprobe": {},
        "opencv": {},
        "web_compatible": "unknown",
    }

    # Análisis con ffprobe
    try:
        # Intentar ffprobe del sistema
        ffprobe_cmd = "ffprobe"
        try:
            import imageio_ffmpeg

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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            data = json.loads(result.stdout)

            # Buscar stream de video
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if video_stream:
                info["ffprobe"] = {
                    "codec": video_stream.get("codec_name", "unknown"),
                    "profile": video_stream.get("profile", "unknown"),
                    "pixel_format": video_stream.get("pix_fmt", "unknown"),
                    "width": video_stream.get("width", 0),
                    "height": video_stream.get("height", 0),
                    "fps": video_stream.get("avg_frame_rate", "unknown"),
                    "duration": float(video_stream.get("duration", 0)),
                    "bitrate": (
                        int(video_stream.get("bit_rate", 0))
                        if video_stream.get("bit_rate")
                        else 0
                    ),
                }

                # Formato/contenedor
                fmt = data.get("format", {})
                info["ffprobe"]["container"] = fmt.get("format_name", "unknown")

                # Evaluar compatibilidad web básica
                codec = info["ffprobe"]["codec"].lower()
                profile = info["ffprobe"]["profile"].lower()
                pix_fmt = info["ffprobe"]["pixel_format"].lower()

                is_web_compatible = (
                    codec in ["h264", "avc"]
                    and any(p in profile for p in ["baseline", "main", "high"])
                    and pix_fmt in ["yuv420p", "yuvj420p"]
                )

                info["web_compatible"] = "✅" if is_web_compatible else "❌"

        else:
            info["ffprobe"]["error"] = "ffprobe failed"

    except Exception as e:
        info["ffprobe"]["error"] = str(e)

    # Análisis con OpenCV
    try:
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened():
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            info["opencv"] = {
                "fourcc": fourcc_str,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "backend": cap.getBackendName(),
            }

            # Test de lectura
            ret, frame = cap.read()
            info["opencv"]["can_read_frames"] = "✅" if ret else "❌"

            cap.release()
        else:
            info["opencv"]["error"] = "Cannot open with OpenCV"

    except Exception as e:
        info["opencv"]["error"] = str(e)

    return info


def print_comparison_table(videos_info):
    """Imprime tabla comparativa."""
    print(f"\n📊 COMPARACIÓN RÁPIDA")
    print("=" * 100)

    # Headers
    headers = [
        "Archivo",
        "Tamaño",
        "Codec",
        "Profile",
        "Pixel Fmt",
        "Resolución",
        "Web",
        "OpenCV FourCC",
        "Can Read",
    ]

    # Calcular anchos
    widths = [15, 8, 8, 12, 10, 10, 5, 8, 8]

    # Imprimir header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Imprimir datos
    for info in videos_info:
        ffprobe = info.get("ffprobe", {})
        opencv = info.get("opencv", {})

        row_data = [
            info["file"][:15],
            f"{info['size_mb']}MB",
            ffprobe.get("codec", "?")[:8],
            ffprobe.get("profile", "?")[:12],
            ffprobe.get("pixel_format", "?")[:10],
            f"{ffprobe.get('width', 0)}x{ffprobe.get('height', 0)}",
            info.get("web_compatible", "?"),
            opencv.get("fourcc", "?")[:8],
            opencv.get("can_read_frames", "?"),
        ]

        row_line = " | ".join(str(data).ljust(w) for data, w in zip(row_data, widths))
        print(row_line)


def print_detailed_comparison(videos_info):
    """Imprime comparación detallada."""
    print(f"\n🔍 ANÁLISIS DETALLADO")
    print("=" * 60)

    # Encontrar diferencias
    if len(videos_info) >= 2:
        fields_to_compare = [
            ("codec", "ffprobe", "codec"),
            ("profile", "ffprobe", "profile"),
            ("pixel_format", "ffprobe", "pixel_format"),
            ("container", "ffprobe", "container"),
            ("opencv_fourcc", "opencv", "fourcc"),
            ("web_compatible", None, "web_compatible"),
        ]

        print("🔄 DIFERENCIAS ENCONTRADAS:")

        for field_name, section, key in fields_to_compare:
            values = []
            for info in videos_info:
                if section:
                    value = info.get(section, {}).get(key, "unknown")
                else:
                    value = info.get(key, "unknown")
                values.append(value)

            unique_values = list(set(values))

            if len(unique_values) > 1:
                print(
                    f"   📌 {field_name.upper()}: {' vs '.join(map(str, unique_values))} ⚠️ DIFERENTE"
                )
            else:
                print(f"   ✓ {field_name}: {unique_values[0]} (igual)")

    # Mostrar recomendaciones si hay incompatibles
    incompatible_videos = [
        info for info in videos_info if info.get("web_compatible") == "❌"
    ]

    if incompatible_videos:
        print(f"\n💡 RECOMENDACIONES PARA VIDEOS INCOMPATIBLES:")

        for info in incompatible_videos:
            print(f"\n   📁 {info['file']}:")
            ffprobe = info.get("ffprobe", {})

            codec = ffprobe.get("codec", "").lower()
            if codec and codec not in ["h264", "avc"]:
                print(f"      • Cambiar codec de '{codec}' a 'H.264'")

            profile = ffprobe.get("profile", "").lower()
            if profile and not any(p in profile for p in ["baseline", "main", "high"]):
                print(f"      • Cambiar profile de '{profile}' a 'baseline'")

            pix_fmt = ffprobe.get("pixel_format", "").lower()
            if pix_fmt and pix_fmt not in ["yuv420p", "yuvj420p"]:
                print(f"      • Cambiar pixel format de '{pix_fmt}' a 'yuv420p'")

            # Comando ffmpeg sugerido
            print(f"      🔧 Comando sugerido:")
            print(
                f"         ffmpeg -i '{info['file']}' -c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags faststart '{info['file'].replace('.mp4', '_web.mp4')}'"
            )


def main():
    """Función principal."""
    if len(sys.argv) < 2:
        print("Uso: python quick_video_compare.py video1.mp4 [video2.mp4] [...]")
        print("\nEjemplo:")
        print(
            "  python quick_video_compare.py video_que_funciona.mp4 video_que_no_funciona.mp4"
        )
        sys.exit(1)

    video_paths = sys.argv[1:]

    print("🎬 COMPARADOR RÁPIDO DE VIDEOS")
    print(f"📁 Analizando {len(video_paths)} video(s)...")

    # Verificar que los archivos existen
    for path in video_paths:
        if not os.path.exists(path):
            print(f"❌ Archivo no encontrado: {path}")
            sys.exit(1)

    # Analizar videos
    videos_info = []
    for path in video_paths:
        info = get_video_info_quick(path)
        videos_info.append(info)

    # Mostrar resultados
    print_comparison_table(videos_info)
    print_detailed_comparison(videos_info)

    print(f"\n🎉 Análisis completado para {len(video_paths)} video(s)")


if __name__ == "__main__":
    main()
