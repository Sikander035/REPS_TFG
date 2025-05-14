""" Module for the main API. """

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes import router


def init():
    """Initialize the API"""
    app = FastAPI()
    # Configuración de CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Permite todos los orígenes. Cambiar esto en producción.
        allow_credentials=True,
        allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
        allow_headers=["*"],  # Permite todos los encabezados
    )
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    init()
