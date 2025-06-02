# backend/src/feedback/analysis_llm.py
import os
import json
import logging
from openai import OpenAI
from pathlib import Path
import sys

# Configurar logging
logger = logging.getLogger(__name__)

# HARDCODED API KEY - CAMBIAR POR TU CLAVE REAL
DEEPSEEK_API_KEY = "CLAVE"


class TrainerFeedbackGenerator:
    """
    Generador de feedback personalizado usando DeepSeek V3 como entrenador personal virtual.
    """

    def __init__(self, api_key=None, base_url=None):
        """
        Inicializa el generador de feedback.
        """
        # Usar API key hardcodeada o la proporcionada
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or "https://api.deepseek.com"

        if not self.api_key or self.api_key == "TU_API_KEY_AQUI":
            logger.error(
                "⚠️  DEBES CAMBIAR LA API KEY EN EL ARCHIVO trainer_feedback.py"
            )
            raise ValueError(
                "Debes configurar tu API key real en el archivo trainer_feedback.py línea 15"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Prompt hardcodeado para simplificar
        self.prompt_template = """# Prompt para Feedback de Entrenador Personal

## Rol
Eres un entrenador personal experto con años de experiencia ayudando a personas a mejorar su técnica en el gimnasio. Has estado analizando la ejecución de tu cliente usando tecnología avanzada de análisis de movimiento.

## Objetivo
Proporciona feedback profesional, motivacional y útil basándote en el informe de análisis técnico. Comunícate como lo haría un entrenador experimentado: con confianza, cercanía y conocimiento práctico.

## Directrices Clave

**Estructura básica requerida:**
1. **Aspectos positivos primero** - Siempre destaca lo que está haciendo bien
2. **Áreas de mejora** - Presenta las correcciones de forma constructiva
3. **Consejos prácticos** - Da recomendaciones específicas y accionables

**Tono y estilo:**
- Profesional pero cercano y motivacional
- Evita jerga demasiado técnica
- Sé específico en tus consejos
- Usa analogías o trucos mentales cuando sea útil
- Mantén un equilibrio entre honestidad y motivación

**Información del informe a interpretar:**
- `puntuacion_global`: Puntuación general del 0-100
- `nivel`: Clasificación del rendimiento
- `puntos_fuertes`: Lista de aspectos bien ejecutados
- `areas_mejora`: Lista de aspectos a corregir
- `feedback_detallado`: Análisis específico por categorías
- `recomendaciones`: Sugerencias técnicas del sistema

## Informe a Analizar

```json
{INFORME_JSON}
```

---

**Instrucción:** Analiza este informe y proporciona tu feedback como entrenador personal. Sé creativo en tu comunicación pero mantén la estructura de destacar lo positivo, identificar mejoras y dar consejos prácticos. Haz que el usuario se sienta motivado y con herramientas claras para mejorar."""

        logger.info("TrainerFeedbackGenerator inicializado con DeepSeek V3")

    def generate_feedback(
        self,
        informe_data,
        exercise_name=None,
        model="deepseek-chat",
        temperature=0.8,
        max_tokens=1500,
    ):
        """
        Genera feedback personalizado basado en el informe de análisis.
        """
        try:
            # Cargar datos del informe si es una ruta
            if isinstance(informe_data, (str, Path)):
                with open(informe_data, "r", encoding="utf-8") as f:
                    informe_data = json.load(f)

            # Extraer nombre del ejercicio si no se proporciona
            if not exercise_name:
                exercise_name = informe_data.get("ejercicio", "ejercicio")

            # Preparar el prompt con los datos del informe
            prompt = self.prompt_template.replace(
                "{INFORME_JSON}", json.dumps(informe_data, indent=2, ensure_ascii=False)
            )

            logger.info(f"Generando feedback para {exercise_name} usando DeepSeek V3")

            # Llamar a DeepSeek V3
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            feedback = response.choices[0].message.content

            logger.info(f"Feedback generado exitosamente ({len(feedback)} caracteres)")
            return feedback

        except Exception as e:
            logger.error(f"Error generando feedback con DeepSeek V3: {e}")
            return self._generate_fallback_feedback(informe_data, exercise_name)

    def _generate_fallback_feedback(self, informe_data, exercise_name):
        """Genera feedback básico si falla la llamada a la API."""
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

*Nota: Este es un análisis básico. Para feedback más detallado, verifica la configuración de la API de DeepSeek.*"""

            return feedback

        except Exception as e:
            logger.error(f"Error en fallback feedback: {e}")
            return f"""❌ **Error generando feedback**

Hubo un problema al analizar tu ejercicio. 
Por favor, verifica:
1. Configuración de la API key en trainer_feedback.py
2. Conexión a internet
3. Que el archivo de informe exista

Error técnico: {e}"""

    def save_feedback(self, feedback, output_path):
        """
        Guarda el feedback en un archivo.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(feedback)

            logger.info(f"Feedback guardado en: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error guardando feedback: {e}")
            return False


def generate_trainer_feedback(informe_path, output_path=None, api_key=None):
    """
    Función de conveniencia para generar feedback rápidamente.
    """
    try:
        generator = TrainerFeedbackGenerator(api_key=api_key)
        feedback = generator.generate_feedback(informe_path)

        if output_path:
            generator.save_feedback(feedback, output_path)

        return feedback

    except Exception as e:
        logger.error(f"Error en generate_trainer_feedback: {e}")
        return f"❌ Error generando feedback: {e}"
