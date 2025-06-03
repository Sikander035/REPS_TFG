"""Module for the main API."""

import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Fix de importación para backend/api/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar estructura
SRC_DIR = os.path.join(BASE_DIR, "src")
if not os.path.exists(SRC_DIR):
    raise FileNotFoundError(f"Directorio 'src' no encontrado en: {SRC_DIR}")

try:
    from routes import router

    logger.info("✅ Rutas importadas correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando rutas: {e}")
    raise

# Crear la aplicación FastAPI
app = FastAPI(
    title="Exercise Analysis API",
    description="API para análisis de técnica de ejercicios físicos usando MediaPipe + IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción usar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(router)


@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Exercise Analysis API",
        "version": "1.0.0",
        "status": "running",
        "structure": "backend/api/",
        "exercises_available": ["military_press", "bench_press", "squat", "pull_up"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "exercises": "/exercises",
            "analyze_exercise": "/analyze-exercise",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    config_path = os.path.join(SRC_DIR, "config", "config.json")
    processor_path = os.path.join(SRC_DIR, "exercise_processor.py")

    return {
        "status": "healthy",
        "base_dir": BASE_DIR,
        "checks": {
            "src_directory": os.path.exists(SRC_DIR),
            "config_file": os.path.exists(config_path),
            "exercise_processor": os.path.exists(processor_path),
            "deepseek_api_key": bool(os.getenv("DEEPSEEK_API_KEY")),
        },
    }


@app.on_event("startup")
async def startup_event():
    """Eventos de inicio."""
    logger.info("🚀 Iniciando Exercise Analysis API...")

    # Crear directorios necesarios
    required_dirs = [
        os.path.join(BASE_DIR, "media", "videos"),
        os.path.join(BASE_DIR, "media", "data"),
        os.path.join(BASE_DIR, "temp_jobs"),
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"📁 Directorio creado: {dir_path}")

    # Verificar configuración
    try:
        from src.config.config_manager import config_manager

        config_path = os.path.join(SRC_DIR, "config", "config.json")
        config_manager.load_config_file(config_path)
        exercises = config_manager.get_available_exercises(config_path)
        logger.info(f"⚙️ Configuración cargada - Ejercicios: {exercises}")
    except Exception as e:
        logger.warning(f"⚠️ Error cargando configuración: {e}")

    # Verificar API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key":
        logger.warning(
            "⚠️ DEEPSEEK_API_KEY no configurada - feedback personalizado deshabilitado"
        )
    else:
        logger.info("✅ DEEPSEEK_API_KEY configurada")

    logger.info("✅ API iniciada correctamente")


# Para compatibilidad con python main.py (pero mejor usar uvicorn)
def init():
    """Función de compatibilidad - mejor usar uvicorn directamente."""
    import uvicorn

    logger.warning(
        "⚠️ Ejecutando con uvicorn interno - mejor usar: uvicorn main:app --reload"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    init()
