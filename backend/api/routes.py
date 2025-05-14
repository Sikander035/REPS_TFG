""" Module for API routes"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
import os, sys
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib, ssl
import uuid
from datetime import datetime, timedelta
from jinja2 import Template
import pandas as pd
from typing import Tuple


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from data_synchronisation.synchronise_by_interpolation import synchronize_data_by_height
from data_extraction.mediapipe_data_extractor import extract_landmarks_from_video
from data_normalisation.normalise_skeleton import normalize_skeleton
from data_visualisation.dual_body_visualisation import generate_dual_skeleton_video

# Importar la función get_all_exercises del módulo db_service
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

load_dotenv()
PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")


router = APIRouter()

token_dict = {}


@router.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest):
    """Forgot password"""
    # Verificar si el email está registrado
    if not user_exists(email=request.email):
        raise HTTPException(status_code=404, detail="Email not found")

    # Generar token
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
    # Cambiar la contraseña
    if not change_password(email=email, password=request.new_password):
        raise HTTPException(status_code=400, detail="Bad request")
    # Eliminar el token
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
    video_path = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "media", "videos")
        ),
        video_name,
    )
    # Check if the file exists
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    # Check if the file is a .mp4 file
    if not video_path.endswith(".mp4"):
        return

    def iterfile():
        with open(video_path, mode="rb") as video:
            yield from video

    return StreamingResponse(iterfile(), media_type="video/mp4")


@router.get("/image")
async def get_image(image_name: str = Query(...)):
    """Get exercise image"""
    image_path = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "media", "images")
        ),
        image_name,
    )
    # Check if the file exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


import time


@router.post("/inference")
async def inference(file: UploadFile = File(...)):
    print("[INFO] Iniciando procesamiento de solicitud de inferencia.")
    videos_folder, data_folder = get_directories()

    try:
        input_video_path = save_uploaded_file(videos_folder, file)
        file_name = os.path.splitext(os.path.basename(input_video_path))[0]

        output_video_path = process_landmarks_and_generate_video(
            input_video_path, data_folder, videos_folder, file_name
        )

        def iterfile():
            with open(output_video_path, mode="rb") as video:
                while chunk := video.read(1024 * 1024):
                    yield chunk

        print("[INFO] Procesamiento completado exitosamente.")
        response = StreamingResponse(iterfile(), media_type="video/mp4")
        response.headers["Content-Disposition"] = (
            f"attachment; filename={os.path.basename(output_video_path)}"
        )

        return response

    except HTTPException as e:
        print(f"[ERROR] HTTPException: {e.detail}")
        raise e
    except Exception as e:
        print(f"[ERROR] Error inesperado: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")


# FUNCIONES AUXILIARES


# Constantes de directorios
def get_directories() -> Tuple[str, str]:
    base_path = os.path.dirname(__file__)
    videos_folder = os.path.join(base_path, "..", "media", "videos")
    data_folder = os.path.join(base_path, "..", "media", "data")
    return videos_folder, data_folder


# Función para guardar archivo subido
def save_uploaded_file(upload_folder: str, file: UploadFile) -> str:
    os.makedirs(upload_folder, exist_ok=True)

    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_folder, unique_filename)

    try:
        print(f"[INFO] Guardando archivo en: {file_path}")
        file.file.seek(0)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise HTTPException(status_code=500, detail="El archivo subido está vacío.")

        print(
            f"[INFO] Archivo guardado correctamente: {unique_filename} ({file_size} bytes)"
        )
    except Exception as e:
        print(f"[ERROR] Error al guardar el archivo: {e}")
        raise HTTPException(
            status_code=500, detail="Error al procesar el archivo subido."
        )

    return file_path


# Función para procesar landmarks y generar video
def process_landmarks_and_generate_video(
    input_video_path: str, data_folder: str, videos_folder: str, file_name: str
):
    landmarks_csv_path = os.path.join(data_folder, f"LANDMARKS_{file_name}.csv")
    print(f"[INFO] Iniciando extracción de landmarks a: {landmarks_csv_path}")

    try:
        extract_landmarks_from_video(input_video_path, landmarks_csv_path)
        print("[INFO] Landmarks extraídos correctamente.")
    except Exception as e:
        print(f"[ERROR] Error durante la extracción de landmarks: {e}")
        raise HTTPException(
            status_code=500, detail="Error durante la extracción de landmarks."
        )

    try:
        user_data = pd.read_csv(landmarks_csv_path)
        example_data_path = os.path.join(data_folder, "press_example.csv")
        if not os.path.exists(example_data_path):
            raise FileNotFoundError("Archivo de ejemplo no encontrado.")

        example_data = pd.read_csv(example_data_path)
        print("[INFO] Landmarks cargados correctamente.")

        reference_lengths = [
            (("landmark_right_wrist", "landmark_right_elbow"), 0.26),
            (("landmark_right_elbow", "landmark_right_shoulder"), 0.34),
            (("landmark_right_shoulder", "landmark_left_shoulder"), 0.44),
            (("landmark_left_shoulder", "landmark_left_elbow"), 0.34),
            (("landmark_left_elbow", "landmark_left_wrist"), 0.26),
            (("landmark_right_shoulder", "landmark_right_hip"), 0.48),
            (("landmark_left_shoulder", "landmark_left_hip"), 0.48),
        ]

        user_normalized_data = normalize_skeleton(user_data, reference_lengths)
        example_normalized_data = normalize_skeleton(example_data, reference_lengths)
        print("[INFO] Landmarks normalizados correctamente.")

        user_processed_data, example_processed_data = synchronize_data_by_height(
            user_normalized_data, example_normalized_data
        )
        print("[INFO] Landmarks sincronizados correctamente.")

        output_video_path = os.path.join(videos_folder, f"OUTPUT_{file_name}.mp4")
        generate_dual_skeleton_video(
            user_processed_data, example_processed_data, output_video_path
        )
        print(f"[INFO] Video generado en: {output_video_path}")

        return output_video_path
    except Exception as e:
        print(
            f"[ERROR] Error durante el procesamiento de landmarks o generación de video: {e}"
        )
        raise HTTPException(
            status_code=500, detail="Error al procesar landmarks o generar video."
        )
    finally:
        if os.path.exists(landmarks_csv_path):
            os.remove(landmarks_csv_path)
        if os.path.exists(input_video_path):
            os.remove(input_video_path)


def load_recovery_template(token: str) -> str:
    # Aquí cargamos y preparamos la plantilla HTML con el token.
    with open(
        os.path.join(os.path.dirname(__file__), "..", "templates", "recovery.html"), "r"
    ) as f:
        template = f.read()

    # Usar Jinja2 para reemplazar el token en la plantilla
    t = Template(template)
    return t.render(token=token)


# Función para generar un token con expiración
def generate_token(email):
    token = str(uuid.uuid4())
    expiration_time = datetime.now() + timedelta(minutes=5)
    token_dict[token] = {"email": email, "expires_at": expiration_time}
    return token


# Función para verificar si el token está expirado
def is_token_expired(token):
    if token in token_dict:
        expiration_time = token_dict[token]["expires_at"]
        return datetime.now() > expiration_time
    return False
