#!/usr/bin/env python3
"""
Script de prueba para la API de análisis de ejercicios.
Simula el flujo completo: subida → polling → descarga de assets.
"""

import requests
import time
import os
from pathlib import Path
import json

# Configuración
API_BASE_URL = "http://localhost:8000"  # Cambia si tu API está en otro puerto
POLL_INTERVAL = 3  # segundos entre polling
MAX_WAIT_TIME = 300  # máximo 5 minutos esperando


def test_exercise_analysis():
    """Prueba completa del flujo de análisis de ejercicios."""

    print("🚀 INICIANDO PRUEBA DE API DE ANÁLISIS DE EJERCICIOS")
    print("=" * 60)

    # 1. VERIFICAR ARCHIVOS NECESARIOS
    print("\n1. Verificando archivos necesarios...")

    # Archivo de video de prueba (puedes usar cualquier MP4)
    test_video_path = input("📁 Ruta al video de prueba (MP4): ").strip()
    if not os.path.exists(test_video_path):
        print(f"❌ Video no encontrado: {test_video_path}")
        return False

    # Parámetros del ejercicio
    exercise_name = (
        input("🏋️ Nombre del ejercicio [military_press]: ").strip() or "military_press"
    )

    print(f"✅ Video: {test_video_path}")
    print(f"✅ Ejercicio: {exercise_name}")
    print(f"📊 Buscará CSV del experto en: media/data/{exercise_name}_Expert.csv")

    # 2. SUBIR VIDEO Y INICIAR ANÁLISIS
    print(f"\n2. Subiendo video y iniciando análisis...")

    try:
        with open(test_video_path, "rb") as f:
            files = {"file": f}
            params = {"exercise_name": exercise_name}

            response = requests.post(
                f"{API_BASE_URL}/analyze-exercise",
                files=files,
                params=params,
                timeout=30,
            )

        if response.status_code != 200:
            print(f"❌ Error en upload: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        result = response.json()
        job_id = result["job_id"]

        print(f"✅ Análisis iniciado!")
        print(f"📋 Job ID: {job_id}")

    except Exception as e:
        print(f"❌ Error subiendo video: {e}")
        return False

    # 3. POLLING Y MONITOREO
    print(f"\n3. Monitoreando progreso...")

    start_time = time.time()
    assets_downloaded = set()

    while True:
        try:
            # Consultar estado
            response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
            if response.status_code != 200:
                print(f"❌ Error consultando estado: {response.status_code}")
                break

            status = response.json()

            # Mostrar progreso
            print(f"\n⏳ Estado: {status['status']}")
            print(f"📝 Paso actual: {status['current_step']}")
            print(f"✅ Pasos completados: {', '.join(status['completed_steps'])}")

            # Verificar assets listos
            assets_ready = status["assets_ready"]
            urls = status.get("urls", {})

            print("📦 Assets disponibles:")
            for asset, ready in assets_ready.items():
                status_emoji = "✅" if ready else "⏳"
                print(f"   {status_emoji} {asset}: {ready}")

            # Descargar assets nuevos
            for asset, url in urls.items():
                if asset not in assets_downloaded:
                    success = download_asset(job_id, asset, url)
                    if success:
                        assets_downloaded.add(asset)

            # Verificar si terminó
            if status["status"] in ["completed", "error"]:
                print(f"\n🎉 Análisis terminado con estado: {status['status']}")
                if status["status"] == "error":
                    print(f"❌ Error: {status.get('error', 'Desconocido')}")
                break

            # Verificar timeout
            if time.time() - start_time > MAX_WAIT_TIME:
                print(f"\n⏰ Timeout alcanzado ({MAX_WAIT_TIME}s)")
                break

            # Esperar antes del siguiente poll
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"❌ Error en polling: {e}")
            break

    # 4. RESUMEN FINAL
    print(f"\n4. Resumen final...")
    print(f"📁 Assets descargados: {len(assets_downloaded)}")
    for asset in assets_downloaded:
        print(f"   ✅ {asset}")

    # 5. LIMPIAR (OPCIONAL)
    cleanup = input("\n🗑️ ¿Limpiar archivos del servidor? [y/N]: ").strip().lower()
    if cleanup == "y":
        try:
            response = requests.delete(f"{API_BASE_URL}/jobs/{job_id}")
            if response.status_code == 200:
                print("✅ Archivos del servidor limpiados")
            else:
                print("⚠️ Error limpiando archivos del servidor")
        except Exception as e:
            print(f"⚠️ Error en cleanup: {e}")

    print("\n🏁 PRUEBA COMPLETADA")
    return True


def download_asset(job_id, asset_name, asset_url):
    """Descarga un asset específico."""
    try:
        print(f"⬇️ Descargando {asset_name}...")

        response = requests.get(f"{API_BASE_URL}{asset_url}", stream=True)
        if response.status_code != 200:
            print(f"❌ Error descargando {asset_name}: {response.status_code}")
            return False

        # Crear directorio de descarga
        download_dir = Path(f"downloads/{job_id}")
        download_dir.mkdir(parents=True, exist_ok=True)

        # Determinar extensión
        extensions = {
            "radar": ".png",
            "video": ".mp4",
            "feedback": ".txt",
            "report": ".json",
        }
        ext = extensions.get(asset_name, "")

        # Guardar archivo
        file_path = download_dir / f"{asset_name}{ext}"
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(file_path)
        print(f"✅ {asset_name} descargado: {file_path} ({file_size:,} bytes)")

        # Preview especial para algunos assets
        if asset_name == "feedback":
            show_feedback_preview(file_path)
        elif asset_name == "report":
            show_report_preview(file_path)

        return True

    except Exception as e:
        print(f"❌ Error descargando {asset_name}: {e}")
        return False


def show_feedback_preview(file_path):
    """Muestra preview del feedback."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        print("\n" + "=" * 50)
        print("🤖 PREVIEW DEL FEEDBACK PERSONALIZADO")
        print("=" * 50)

        # Mostrar primeras 5 líneas
        lines = content.split("\n")[:10]
        for line in lines:
            if line.strip():
                print(line.strip())

        if len(content.split("\n")) > 10:
            print("...")
            print(f"(Ver archivo completo en: {file_path})")

        print("=" * 50)

    except Exception as e:
        print(f"Error mostrando preview de feedback: {e}")


def show_report_preview(file_path):
    """Muestra preview del reporte."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        print(f"\n📊 RESUMEN DEL ANÁLISIS:")
        print(f"   🏋️ Ejercicio: {report.get('exercise', 'N/A')}")
        print(f"   🎯 Puntuación: {report.get('overall_score', 'N/A')}/100")
        print(f"   📈 Nivel: {report.get('level', 'N/A')}")

        if "individual_scores" in report:
            print("   📋 Puntuaciones por categoría:")
            for category, score in report["individual_scores"].items():
                print(f"      • {category}: {score:.1f}")

    except Exception as e:
        print(f"Error mostrando preview de reporte: {e}")


def test_simple_curl_example():
    """Muestra ejemplos de curl para pruebas rápidas."""
    print("\n" + "=" * 60)
    print("📋 EJEMPLOS DE CURL PARA PRUEBAS RÁPIDAS")
    print("=" * 60)

    print("\n1. Subir video:")
    print(
        'curl -X POST "http://localhost:8000/analyze-exercise?exercise_name=military_press" \\'
    )
    print('  -F "file=@tu_video.mp4"')
    print("   (Buscará automáticamente: media/data/military_press_Expert.csv)")

    print("\n2. Consultar estado (reemplaza JOB_ID):")
    print('curl "http://localhost:8000/jobs/TU_JOB_ID"')

    print("\n3. Descargar radar:")
    print('curl "http://localhost:8000/assets/TU_JOB_ID/radar.png" -o radar.png')

    print("\n4. Descargar video:")
    print('curl "http://localhost:8000/assets/TU_JOB_ID/video.mp4" -o comparison.mp4')

    print("\n5. Limpiar:")
    print('curl -X DELETE "http://localhost:8000/jobs/TU_JOB_ID"')


def check_server_status():
    """Verifica que el servidor esté corriendo."""
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor FastAPI está corriendo")
            return True
        else:
            print(f"⚠️ Servidor responde pero con código: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ No se puede conectar al servidor")
        print(f"   Verifica que FastAPI esté corriendo en {API_BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error verificando servidor: {e}")
        return False


if __name__ == "__main__":
    print("🧪 SCRIPT DE PRUEBA - API DE ANÁLISIS DE EJERCICIOS")
    print("=" * 60)

    # Verificar servidor
    if not check_server_status():
        print("\n💡 Para iniciar el servidor:")
        print("   uvicorn main:app --reload --port 8000")
        exit(1)

    # Menú de opciones
    print("\n📋 Opciones disponibles:")
    print("1. 🧪 Prueba completa (recomendado)")
    print("2. 📋 Ver ejemplos de curl")
    print("3. 🔍 Solo verificar servidor")

    option = input("\nSelecciona opción [1]: ").strip() or "1"

    if option == "1":
        test_exercise_analysis()
    elif option == "2":
        test_simple_curl_example()
    elif option == "3":
        print("✅ Servidor verificado correctamente")
    else:
        print("❌ Opción inválida")
