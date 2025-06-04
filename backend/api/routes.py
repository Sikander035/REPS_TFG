"""Module for API routes - THREAD-SAFE VERSION"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Header
from fastapi.responses import StreamingResponse, FileResponse, Response
import os, sys
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib, ssl
import uuid
import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from jinja2 import Template
import pandas as pd
from typing import Tuple, Dict, Any
import shutil
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# CRÍTICO: Configurar matplotlib ANTES de cualquier import
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

# Configurar matplotlib para threading
from src.config.matplotlib_config import configure_matplotlib_for_threading

configure_matplotlib_for_threading()

# Ahora importar el resto
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "services", "database")
    )
)
from db_service import (
    get_all_exercises,
    get_exercise_by_name,
    get_exercises_by_muscle_group,
    check_credentials,
    register_user,
    get_all_users,
    user_exists,
    change_password,
)

from request_schemas import (
    RegisterRequest,
    LoginRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)

# IMPORTACIONES para análisis de ejercicios
from src.exercise_processor import ExerciseProcessor

load_dotenv()
PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")

router = APIRouter()

# Estado global de trabajos con thread safety
jobs_state: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

# ThreadPoolExecutor configurado para thread safety
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ExerciseProcessor")

# Configuración de rutas
TEMP_DIR = os.path.join(BASE_DIR, "temp_jobs")
VIDEOS_DIR = os.path.join(BASE_DIR, "media", "videos")
DATA_DIR = os.path.join(BASE_DIR, "media", "data")
CONFIG_PATH = os.path.join(BASE_DIR, "src", "config", "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

# Asegurar que existen los directorios necesarios
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# RUTAS ORIGINALES (MANTENER IGUAL)
# ===============================

token_dict = {}


@router.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest):
    """Forgot password"""
    if not user_exists(email=request.email):
        raise HTTPException(status_code=404, detail="Email not found")

    token = generate_token(email=request.email)
    content = load_recovery_template(token)

    em = EmailMessage()
    em["From"] = EMAIL_SENDER
    em["To"] = request.email
    em["Subject"] = "Recuperación de contraseña"
    em.add_alternative(content, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_SENDER, PASSWORD)
        server.sendmail(EMAIL_SENDER, request.email, em.as_string())
    return {"success": True}


@router.post("/reset-password")
def reset_password(request: ResetPasswordRequest):
    """Reset password"""
    if request.token not in token_dict:
        raise HTTPException(status_code=404, detail="Token not found")
    if is_token_expired(request.token):
        raise HTTPException(status_code=400, detail="Token expired")
    email = token_dict[request.token]["email"]
    if not change_password(email=email, password=request.new_password):
        raise HTTPException(status_code=400, detail="Bad request")
    del token_dict[request.token]
    return {"success": True}


@router.get("/exercises")
def get_exercises(muscle_group: str = Query(None), exercise_name: str = Query(None)):
    """Get exercises filtered by muscle group and exercise name"""
    if muscle_group is None and exercise_name is None:
        return get_all_exercises()
    elif exercise_name is not None:
        return get_exercise_by_name(exercise_name=exercise_name)
    elif muscle_group is not None:
        return get_exercises_by_muscle_group(muscle_group=muscle_group)


@router.post("/register/")
async def register(request: RegisterRequest):
    """Register"""
    if register_user(email=request.email, name=request.name, password=request.password):
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="Bad request")


@router.post("/login/")
async def login(request: LoginRequest):
    """Login"""
    if check_credentials(email=request.email, password=request.password):
        return {"success": True}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/users")
def get_users():
    """Get all users"""
    return get_all_users()


@router.get("/video")
def get_video(video_name: str = Query(...)):
    """Get exercise video"""
    video_path = os.path.join(VIDEOS_DIR, video_name)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    if not video_path.endswith(".mp4"):
        return

    def iterfile():
        with open(video_path, mode="rb") as video:
            yield from video

    return StreamingResponse(iterfile(), media_type="video/mp4")


@router.get("/image")
async def get_image(image_name: str = Query(...)):
    """Get exercise image"""
    image_path = os.path.join(BASE_DIR, "media", "images", image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


# ===============================
# NUEVAS RUTAS DE ANÁLISIS (THREAD-SAFE)
# ===============================


@router.post("/analyze-exercise")
async def analyze_exercise(
    file: UploadFile = File(...), exercise_name: str = Query(default="military_press")
):
    """
    Inicia análisis completo de ejercicio.
    VERSION THREAD-SAFE con cleanup automático.
    """
    job_id = str(uuid.uuid4())

    # MAPEAR EL EJERCICIO PRIMERO (MOVER AQUÍ)
    exercise_mapping = {
        "press_militar_con_mancuernas": "military_press",
        "military_press": "military_press",
        "bench_press": "bench_press",
        "squat": "squat",
        "pull_up": "pull_up",
    }
    mapped_exercise_name = exercise_mapping.get(exercise_name, "military_press")

    # Thread-safe job creation
    with jobs_lock:
        # Crear directorio para este trabajo
        job_dir = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Guardar video subido
        user_video_path = os.path.join(job_dir, f"user_video_{file.filename}")

        # Inicializar estado del trabajo
        jobs_state[job_id] = {
            "status": "processing",
            "completed_steps": [],
            "current_step": "saving_video",
            "assets_ready": {
                "radar": False,
                "video": False,
                "feedback": False,
                "report": False,
            },
            "urls": {},
            "created_at": datetime.now(),
            "job_dir": job_dir,
            "user_video": user_video_path,
            "exercise_name": mapped_exercise_name,  # AHORA YA EXISTE
            "error": None,
        }

    try:
        # Guardar video
        with open(user_video_path, "wb") as buffer:
            buffer.write(await file.read())

        # Construir ruta al CSV del experto (YA NO REPETIR EL MAPEO)
        expert_csv_name = f"{mapped_exercise_name}_Expert.csv"
        expert_csv_path = os.path.join(DATA_DIR, expert_csv_name)

        if not os.path.exists(expert_csv_path):
            with jobs_lock:
                jobs_state[job_id]["status"] = "error"
                jobs_state[job_id]["error"] = f"Expert CSV not found: {expert_csv_name}"
            raise HTTPException(
                status_code=404,
                detail=f"Expert CSV not found: media/data/{expert_csv_name}",
            )

        # Añadir ruta del experto al estado
        with jobs_lock:
            jobs_state[job_id]["expert_csv"] = expert_csv_path
            jobs_state[job_id]["current_step"] = "initializing"

        # Iniciar procesamiento asíncrono
        asyncio.create_task(process_exercise_async(job_id))

        return {"job_id": job_id, "status": "processing"}

    except Exception as e:
        # Cleanup en caso de error
        with jobs_lock:
            jobs_state[job_id]["status"] = "error"
            jobs_state[job_id]["error"] = str(e)

        logger.error(f"Error inicializando job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Obtiene estado actual del trabajo (thread-safe)."""
    with jobs_lock:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs_state[job_id].copy()  # Crear copia para evitar race conditions

    # Construir URLs para assets listos
    urls = {}
    if job["assets_ready"]["radar"]:
        urls["radar"] = f"/assets/{job_id}/radar.png"
    if job["assets_ready"]["video"]:
        urls["video"] = f"/assets/{job_id}/video.mp4"
    if job["assets_ready"]["feedback"]:
        urls["feedback"] = f"/assets/{job_id}/feedback.txt"
    if job["assets_ready"]["report"]:
        urls["report"] = f"/assets/{job_id}/report.json"

    return {
        "status": job["status"],
        "completed_steps": job["completed_steps"],
        "current_step": job["current_step"],
        "assets_ready": job["assets_ready"],
        "urls": urls,
        "error": job.get("error"),
    }


@router.get("/assets/{job_id}/video.mp4")
async def get_video_asset(job_id: str, range: str = Header(None)):
    """Video endpoint con CORS obligatorio para cross-origin."""

    with jobs_lock:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Job not found")
        job_dir = jobs_state[job_id]["job_dir"]

    video_path = os.path.join(job_dir, "comparison_video.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not ready")

    file_size = os.path.getsize(video_path)

    # CORS headers OBLIGATORIOS para video cross-origin
    cors_headers = {
        "Access-Control-Allow-Origin": "*",  # Permitir todos los orígenes
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Type",
        "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
    }

    # Sin Range header - archivo completo
    if not range:
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={
                **cors_headers,
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600",
            },
        )

    # Con Range header - partial content
    try:
        range_str = range.replace("bytes=", "").strip()

        if range_str.endswith("-"):
            start = int(range_str[:-1])
            end = file_size - 1
        else:
            parts = range_str.split("-")
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else file_size - 1

        start = max(0, start)
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def generate_range():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            generate_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                **cors_headers,
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(content_length),
                "Cache-Control": "public, max-age=3600",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid range")


@router.get("/assets/{job_id}/feedback.txt")
async def get_feedback_asset(job_id: str):
    """Devuelve el feedback personalizado."""
    with jobs_lock:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Job not found")
        job_dir = jobs_state[job_id]["job_dir"]

    feedback_path = os.path.join(job_dir, "analysis", "personalized_feedback.txt")
    if not os.path.exists(feedback_path):
        raise HTTPException(status_code=404, detail="Feedback not ready")
    return FileResponse(feedback_path, media_type="text/plain")


@router.get("/assets/{job_id}/report.json")
async def get_report_asset(job_id: str):
    """Devuelve el reporte de análisis."""
    with jobs_lock:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Job not found")
        job_dir = jobs_state[job_id]["job_dir"]
        exercise_name = jobs_state[job_id]["exercise_name"]

    report_path = os.path.join(job_dir, "analysis", f"{exercise_name}_report.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not ready")
    return FileResponse(report_path, media_type="application/json")


@router.delete("/jobs/{job_id}")
async def cleanup_job(job_id: str):
    """Limpia archivos temporales de un trabajo."""
    with jobs_lock:
        if job_id not in jobs_state:
            raise HTTPException(status_code=404, detail="Job not found")

        job_dir = jobs_state[job_id]["job_dir"]

        # Eliminar directorio y archivos
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)

        # Remover del estado
        del jobs_state[job_id]

    return {"message": "Job cleaned up successfully"}


# ===============================
# LÓGICA DE PROCESAMIENTO THREAD-SAFE
# ===============================


async def process_exercise_async(job_id: str):
    """
    Procesa el ejercicio de forma asíncrona con thread safety completo.
    """
    processor = None

    try:
        # Obtener datos del job de forma thread-safe
        with jobs_lock:
            if job_id not in jobs_state:
                return
            job = jobs_state[job_id].copy()

        # Crear processor con cleanup automático
        processor = ExerciseProcessor(
            job["user_video"],
            job["expert_csv"],
            job["exercise_name"],
            job["job_dir"],
            CONFIG_PATH,
            MODEL_PATH,
        )

        # Ejecutar pasos secuenciales
        await run_sequential_steps(job_id, processor)

        # Ejecutar análisis
        await run_analysis_step(job_id, processor)

        # Ejecutar pasos paralelos
        await run_parallel_steps(job_id, processor)

        # Marcar como completado
        with jobs_lock:
            if job_id in jobs_state:
                jobs_state[job_id]["status"] = "completed"
                jobs_state[job_id]["current_step"] = "finished"

        logger.info(f"✅ Job {job_id} completado exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en job {job_id}: {e}")
        with jobs_lock:
            if job_id in jobs_state:
                jobs_state[job_id]["status"] = "error"
                jobs_state[job_id]["error"] = str(e)

    finally:
        # CRÍTICO: Cleanup final del processor
        if processor:
            try:
                processor.cleanup_all_resources()
                logger.debug(f"🧹 Cleanup completado para job {job_id}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Error en cleanup final: {cleanup_error}")


def update_job_step(job_id: str, step_name: str, completed: bool = False):
    """Actualiza el estado del job de forma thread-safe."""
    with jobs_lock:
        if job_id in jobs_state:
            jobs_state[job_id]["current_step"] = step_name
            if completed:
                jobs_state[job_id]["completed_steps"].append(step_name)


async def run_sequential_steps(job_id: str, processor: ExerciseProcessor):
    """Ejecuta pasos 1-6 secuencialmente con thread safety."""
    steps = [
        ("extraction", processor.extract_landmarks_user_only),
        ("load_expert", processor.load_expert_data),
        ("repetition_detection", processor.detect_repetitions),
        ("synchronization", processor.synchronize_data),
        ("normalization", processor.normalize_skeletons),
        ("alignment", processor.align_skeletons),
    ]

    for step_name, step_func in steps:
        update_job_step(job_id, step_name)

        # Ejecutar en thread con manejo de errores
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(executor, step_func)
            update_job_step(job_id, step_name, completed=True)
            logger.debug(f"✅ Paso {step_name} completado para job {job_id}")
        except Exception as e:
            logger.error(f"❌ Error en paso {step_name} para job {job_id}: {e}")
            raise


async def run_analysis_step(job_id: str, processor: ExerciseProcessor):
    """Ejecuta paso 7: análisis detallado con thread safety."""
    update_job_step(job_id, "analysis")

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(executor, processor.run_analysis)

        # Marcar assets como listos
        with jobs_lock:
            if job_id in jobs_state:
                jobs_state[job_id]["assets_ready"]["radar"] = True
                jobs_state[job_id]["assets_ready"]["report"] = True

        update_job_step(job_id, "analysis", completed=True)
        logger.debug(f"✅ Análisis completado para job {job_id}")

    except Exception as e:
        logger.error(f"❌ Error en análisis para job {job_id}: {e}")
        raise


async def run_parallel_steps(job_id: str, processor: ExerciseProcessor):
    """Ejecuta pasos 6 y 8 en paralelo con thread safety."""
    update_job_step(job_id, "generating_assets")

    loop = asyncio.get_event_loop()

    # Tasks paralelos
    video_task = loop.run_in_executor(executor, processor.generate_video)
    feedback_task = loop.run_in_executor(executor, processor.generate_feedback)

    # Esperar resultados
    video_result, feedback_result = await asyncio.gather(
        video_task, feedback_task, return_exceptions=True
    )

    # Actualizar estado según resultados
    with jobs_lock:
        if job_id in jobs_state:
            if not isinstance(video_result, Exception):
                jobs_state[job_id]["assets_ready"]["video"] = True
                jobs_state[job_id]["completed_steps"].append("video_generation")
                logger.debug(f"✅ Video generado para job {job_id}")
            else:
                logger.error(
                    f"❌ Error generando video para job {job_id}: {video_result}"
                )

            if not isinstance(feedback_result, Exception):
                jobs_state[job_id]["assets_ready"]["feedback"] = True
                jobs_state[job_id]["completed_steps"].append("feedback_generation")
                logger.debug(f"✅ Feedback generado para job {job_id}")
            else:
                logger.error(
                    f"❌ Error generando feedback para job {job_id}: {feedback_result}"
                )


# ===============================
# FUNCIONES AUXILIARES ORIGINALES
# ===============================


def load_recovery_template(token: str) -> str:
    """Carga plantilla de recuperación de contraseña."""
    with open(
        os.path.join(os.path.dirname(__file__), "..", "templates", "recovery.html"), "r"
    ) as f:
        template = f.read()
    t = Template(template)
    return t.render(token=token)


def generate_token(email):
    """Genera token con expiración."""
    token = str(uuid.uuid4())
    expiration_time = datetime.now() + timedelta(minutes=5)
    token_dict[token] = {"email": email, "expires_at": expiration_time}
    return token


def is_token_expired(token):
    """Verifica si el token está expirado."""
    if token in token_dict:
        expiration_time = token_dict[token]["expires_at"]
        return datetime.now() > expiration_time
    return False


# ===============================
# CLEANUP AUTOMÁTICO
# ===============================


async def periodic_cleanup():
    """Limpia trabajos antiguos periódicamente."""
    while True:
        try:
            current_time = datetime.now()
            jobs_to_remove = []

            with jobs_lock:
                for job_id, job in jobs_state.items():
                    # Eliminar trabajos de más de 1 hora
                    if current_time - job["created_at"] > timedelta(hours=1):
                        jobs_to_remove.append(job_id)

                for job_id in jobs_to_remove:
                    try:
                        job_dir = jobs_state[job_id]["job_dir"]
                        if os.path.exists(job_dir):
                            shutil.rmtree(job_dir)
                        del jobs_state[job_id]
                        logger.info(f"🧹 Job {job_id} limpiado automáticamente")
                    except Exception as e:
                        logger.warning(f"⚠️ Error limpiando job {job_id}: {e}")

        except Exception as e:
            logger.error(f"❌ Error en cleanup periódico: {e}")

        # Ejecutar cada 30 minutos
        await asyncio.sleep(1800)


# Iniciar cleanup al cargar el módulo
@router.on_event("startup")
async def startup_cleanup():
    """Inicia cleanup automático."""
    asyncio.create_task(periodic_cleanup())
    logger.info("🧹 Cleanup automático iniciado")
