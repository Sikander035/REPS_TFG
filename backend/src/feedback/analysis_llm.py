# backend/src/feedback/analysis_llm.py
import os
import json
import logging
import traceback
from openai import OpenAI
from pathlib import Path

# Configurar logging
logger = logging.getLogger(__name__)


def load_prompt_template(prompt_file_path=None):
    """
    Carga el template de prompt desde archivo.

    Args:
        prompt_file_path (str): Ruta específica al archivo de prompt (opcional)

    Returns:
        str: Contenido del prompt template

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de prompt
        Exception: Para otros errores de lectura
    """
    # Determinar la ruta del archivo de prompt
    if prompt_file_path and Path(prompt_file_path).exists():
        prompt_path = Path(prompt_file_path)
    else:
        # Buscar trainer_prompt.txt automáticamente
        current_dir = Path(__file__).parent  # .../backend/src/feedback/
        possible_paths = [
            current_dir.parent
            / "config"
            / "trainer_prompt.txt",  # backend/src/config/trainer_prompt.txt ← UBICACIÓN REAL
            current_dir
            / "trainer_prompt.txt",  # backend/src/feedback/trainer_prompt.txt
            current_dir.parent / "trainer_prompt.txt",  # backend/src/trainer_prompt.txt
            current_dir.parent.parent
            / "trainer_prompt.txt",  # backend/trainer_prompt.txt
            current_dir.parent.parent.parent
            / "trainer_prompt.txt",  # REPS_TFG/trainer_prompt.txt
            Path("src/config/trainer_prompt.txt"),  # desde backend/
            Path("trainer_prompt.txt"),  # directorio actual
            Path("backend/src/config/trainer_prompt.txt"),  # desde REPS_TFG/
        ]

        prompt_path = None
        for path in possible_paths:
            if path.exists():
                prompt_path = path
                break

    # Verificar que se encontró el archivo
    if not prompt_path or not prompt_path.exists():
        error_msg = (
            f"❌ No se pudo encontrar el archivo trainer_prompt.txt\n"
            f"Rutas buscadas:\n"
            + "\n".join([f"  - {p}" for p in possible_paths])
            + (
                f"\nRuta específica buscada: {prompt_file_path}"
                if prompt_file_path
                else ""
            )
        )
        logger.error(error_msg)
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        raise FileNotFoundError(error_msg)

    # Cargar el contenido del archivo
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()

        if not prompt_content.strip():
            raise ValueError(f"❌ El archivo de prompt está vacío: {prompt_path}")

        logger.info(f"✅ Prompt cargado exitosamente desde: {prompt_path}")
        logger.debug(f"Prompt length: {len(prompt_content)} caracteres")
        logger.debug(f"Ubicación detectada: {prompt_path.absolute()}")

        return prompt_content

    except UnicodeDecodeError as e:
        error_msg = f"❌ Error de codificación al leer {prompt_path}: {e}"
        logger.error(error_msg)
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        raise Exception(error_msg)

    except Exception as e:
        error_msg = f"❌ Error inesperado al cargar prompt desde {prompt_path}: {e}"
        logger.error(error_msg)
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        raise Exception(error_msg)


def generate_trainer_feedback(
    informe_path,
    output_path=None,
    api_key=None,
    prompt_file_path=None,
    model="deepseek-chat",
    temperature=0.8,
    max_tokens=1500,
    base_url="https://api.deepseek.com",
):
    """
    Genera feedback personalizado usando DeepSeek V3.

    Args:
        informe_path (str): Ruta al archivo de informe JSON
        output_path (str): Ruta donde guardar el feedback (opcional)
        api_key (str): API key de DeepSeek (obligatoria)
        prompt_file_path (str): Ruta al archivo de prompt (opcional)
        model (str): Modelo de DeepSeek a usar
        temperature (float): Temperatura para la generación
        max_tokens (int): Máximo número de tokens
        base_url (str): URL base de la API

    Returns:
        str: Feedback generado

    Raises:
        ValueError: Si faltan parámetros obligatorios
        Exception: Para otros errores
    """
    try:
        # Validar API key
        if not api_key:
            error_msg = (
                "❌ API key es obligatoria para generate_trainer_feedback. "
                "Pasa el parámetro api_key o carga DEEPSEEK_API_KEY desde variables de entorno."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if api_key in ["your_deepseek_api_key", "CLAVE", "tu_clave_real"]:
            error_msg = (
                "❌ Debes configurar tu API key real de DeepSeek. "
                "El valor actual parece ser un placeholder."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Cargar datos del informe
        if isinstance(informe_path, (str, Path)):
            if not os.path.exists(informe_path):
                raise FileNotFoundError(
                    f"❌ Archivo de informe no encontrado: {informe_path}"
                )

            with open(informe_path, "r", encoding="utf-8") as f:
                informe_data = json.load(f)
        else:
            informe_data = informe_path  # Asumimos que ya es un diccionario

        # Validar que informe_data tenga contenido
        if not informe_data:
            raise ValueError("❌ Los datos del informe están vacíos")

        # Extraer nombre del ejercicio
        exercise_name = informe_data.get(
            "exercise", informe_data.get("ejercicio", "ejercicio")
        )

        # Cargar template de prompt
        prompt_template = load_prompt_template(prompt_file_path)

        # Preparar el prompt con los datos del informe
        informe_json = json.dumps(informe_data, indent=2, ensure_ascii=False)
        prompt = prompt_template.replace("{INFORME_JSON}", informe_json)

        logger.info(f"🚀 Generando feedback para '{exercise_name}' usando DeepSeek V3")
        logger.debug(f"Prompt length: {len(prompt)} caracteres")
        logger.debug(
            f"Model: {model}, Temperature: {temperature}, Max tokens: {max_tokens}"
        )

        # Inicializar cliente OpenAI con configuración de DeepSeek
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Llamar a DeepSeek V3
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        feedback = response.choices[0].message.content

        if not feedback or not feedback.strip():
            raise ValueError("❌ DeepSeek V3 devolvió una respuesta vacía")

        logger.info(f"✅ Feedback generado exitosamente ({len(feedback)} caracteres)")

        # Guardar si se especifica ruta
        if output_path:
            save_feedback_to_file(feedback, output_path)

        return feedback

    except FileNotFoundError as e:
        logger.error(f"❌ Error de archivo: {e}")
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        return generate_error_feedback(f"Archivo no encontrado: {e}", exercise_name)

    except json.JSONDecodeError as e:
        logger.error(f"❌ Error de formato JSON en informe: {e}")
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        return generate_error_feedback(f"Error de formato JSON: {e}", exercise_name)

    except Exception as e:
        logger.error(f"❌ Error generando feedback con DeepSeek V3: {e}")
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())

        # Intentar generar feedback de respaldo
        try:
            exercise_name = informe_data.get(
                "exercise", informe_data.get("ejercicio", "ejercicio")
            )
            return generate_fallback_feedback(informe_data, exercise_name, str(e))
        except:
            return generate_error_feedback(f"Error crítico: {e}", "ejercicio")


def generate_fallback_feedback(informe_data, exercise_name, error_msg=""):
    """
    Genera feedback básico si falla la llamada a la API.

    Args:
        informe_data: Datos del informe
        exercise_name: Nombre del ejercicio
        error_msg: Mensaje de error (opcional)

    Returns:
        str: Feedback básico generado localmente
    """
    try:
        puntuacion = informe_data.get("overall_score", 0)
        nivel = informe_data.get("level", "No determinado")
        puntos_fuertes = informe_data.get("strengths", [])
        areas_mejora = informe_data.get("improvement_areas", [])

        feedback = f"""🤖 **ANÁLISIS AUTOMÁTICO DE TU {exercise_name.upper()}**

**📊 Evaluación General:** {puntuacion}/100 - Nivel: {nivel}

**✅ Aspectos Positivos:**
"""
        for punto in puntos_fuertes[:3]:
            feedback += f"• {punto}\n"

        feedback += "\n**🔧 Áreas de Mejora:**\n"
        for area in areas_mejora[:3]:
            feedback += f"• {area}\n"

        feedback += f"""
**🚀 Motivación:**
Con una puntuación de {puntuacion}/100, estás en el nivel "{nivel}". 
Sigue trabajando en los aspectos mencionados y verás mejoras pronto.

**⚠️  Nota Técnica:**
Este es un análisis básico generado localmente porque ocurrió un error 
con DeepSeek V3. Para feedback más detallado, verifica:
• Conexión a internet
• Configuración de la API key
• Que el archivo trainer_prompt.txt existe

Error técnico: {error_msg}"""

        logger.info("🔄 Feedback generado usando sistema de respaldo local")
        return feedback

    except Exception as fallback_error:
        logger.error(f"❌ Error crítico en sistema de respaldo: {fallback_error}")
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        return generate_error_feedback(
            f"Error crítico: {fallback_error}", exercise_name
        )


def generate_error_feedback(error_msg, exercise_name):
    """
    Genera un mensaje de error amigable para el usuario.

    Args:
        error_msg: Mensaje de error técnico
        exercise_name: Nombre del ejercicio

    Returns:
        str: Mensaje de error formateado
    """
    return f"""❌ **ERROR EN ANÁLISIS DE {exercise_name.upper()}**

No se pudo generar el feedback personalizado debido a un error técnico.

**Qué hacer:**
1. ✅ Verifica que tu archivo .env contenga: DEEPSEEK_API_KEY=tu_clave_real
2. ✅ Asegúrate de que tienes conexión a internet
3. ✅ Confirma que el archivo trainer_prompt.txt existe en src/config/
4. ✅ Revisa los logs para más detalles técnicos

**Error técnico:** {error_msg}

**Soluciones comunes:**
• Si es error de API key: actualiza tu .env con la clave correcta
• Si es error de archivo: verifica que trainer_prompt.txt existe en src/config/
• Si es error de conexión: verifica tu internet

**Contacto:** Si el problema persiste, revisa la documentación del proyecto."""


def save_feedback_to_file(feedback, output_path):
    """
    Guarda el feedback en un archivo.

    Args:
        feedback (str): Contenido del feedback
        output_path (str): Ruta donde guardar el archivo

    Returns:
        bool: True si se guardó exitosamente, False en caso contrario
    """
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Guardar el archivo
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(feedback)

        logger.info(f"✅ Feedback guardado exitosamente en: {output_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Error guardando feedback en {output_path}: {e}")
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        return False


# Funciones de utilidad para compatibilidad (si se necesitan)
def validate_api_key(api_key):
    """
    Valida que la API key sea válida.

    Args:
        api_key (str): API key a validar

    Returns:
        bool: True si es válida, False en caso contrario
    """
    if not api_key:
        return False
    if api_key in ["your_deepseek_api_key", "CLAVE", "tu_clave_real"]:
        return False
    if len(api_key) < 10:
        return False
    return True


def test_deepseek_connection(api_key, base_url="https://api.deepseek.com"):
    """
    Prueba la conexión con DeepSeek.

    Args:
        api_key (str): API key de DeepSeek
        base_url (str): URL base de la API

    Returns:
        bool: True si la conexión es exitosa, False en caso contrario
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
            temperature=0,
        )

        return bool(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error probando conexión DeepSeek: {e}")
        return False
