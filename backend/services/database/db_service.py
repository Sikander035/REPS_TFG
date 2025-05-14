""" Module for database service """

from pymongo import MongoClient
from bson.json_util import dumps
import json
from dotenv import load_dotenv
import os
from hashlib import sha256

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener las variables de entorno
USER = os.getenv("MONGO_INITDB_ROOT_USERNAME")
PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
CONTAINER_NAME = os.getenv("MONGO_CONTAINER_NAME")
DATABASE_NAME = os.getenv("MONGO_DB_NAME")

# Configuración de conexión
MONGO_URI = f"mongodb://{USER}:{PASSWORD}@{CONTAINER_NAME}:27017"  # URL de conexión


# MongoClient decorator
def mongo_client_decorator(func):
    def wrapper(*args, **kwargs):
        client = MongoClient(MONGO_URI)
        kwargs["client"] = client
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("Error al conectar o consultar la base de datos:", e)
        finally:
            client.close()

    return wrapper


@mongo_client_decorator
def get_all_exercises(client):
    """Get all exercises"""
    db = client[DATABASE_NAME]
    items = db.exercises.find()
    items = list(items)
    return json.loads(dumps(items))


@mongo_client_decorator
def get_exercise_by_name(client, exercise_name):
    """Get exercise by name"""
    db = client[DATABASE_NAME]
    query = {"name": {"$regex": f"^{exercise_name}$", "$options": "i"}}
    item = db.exercises.find_one(query)
    return json.loads(dumps(item))


@mongo_client_decorator
def get_exercises_by_muscle_group(client, muscle_group):
    """Get exercises by muscle group"""
    db = client[DATABASE_NAME]
    query = {"muscle_group": {"$regex": f"^{muscle_group}$", "$options": "i"}}
    items = db.exercises.find(query)
    items = list(items)
    return json.loads(dumps(items))


@mongo_client_decorator
def user_exists(client, email):
    """Check if user exists"""
    db = client[DATABASE_NAME]
    user = db.users.find_one({"email": email})
    return user is not None


@mongo_client_decorator
def check_credentials(client, email, password):
    """Check user credentials"""
    db = client[DATABASE_NAME]
    user = db.users.find_one(
        {"email": email, "password": sha256(password.encode()).hexdigest()}
    )
    return user is not None


@mongo_client_decorator
def register_user(client, email, name, password):
    """Register user"""
    db = client[DATABASE_NAME]
    user = db.users.find_one({"email": email})
    if user is not None:
        return False
    db.users.insert_one(
        {
            "email": email,
            "name": name,
            "password": sha256(password.encode()).hexdigest(),
        }
    )
    return True


@mongo_client_decorator
def change_password(client, email, password):
    """Change password"""
    db = client[DATABASE_NAME]
    db.users.update_one(
        {"email": email}, {"$set": {"password": sha256(password.encode()).hexdigest()}}
    )
    return True


@mongo_client_decorator
def get_all_users(client):
    """Get all users"""
    db = client[DATABASE_NAME]
    items = db.users.find()
    items = list(items)
    return json.loads(dumps(items))
