import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExerciseAnalyzer:
    """Analizador de ejercicios para proporcionar feedback basado en la comparación de esqueletos."""

    def __init__(self, user_data, expert_data, exercise_name="press_militar"):
        """
        Inicializa el analizador con datos de usuario y experto.

        Args:
            user_data: DataFrame con datos del usuario
            expert_data: DataFrame con datos del experto
            exercise_name: Nombre del ejercicio para cargar configuraciones específicas
        """
        self.user_data = user_data
        self.expert_data = expert_data
        self.exercise_name = exercise_name
        self.feedback = {}
        self.metrics = {}
        self.overall_score = 0
        self.level = ""

        # Configuración específica para press militar
        if exercise_name == "press_militar":
            self.specific_config = {
                # Ángulos ideales para el press militar (en grados)
                "min_elbow_angle": 45,  # Ángulo mínimo de flexión de codo
                "max_elbow_angle": 175,  # Ángulo máximo de extensión de codo
                # Umbrales para análisis
                "rom_threshold": 0.85,  # Porcentaje mínimo del rango de movimiento
                "bottom_diff_threshold": 0.2,  # Diferencia máxima en posición baja
                "angle_diff_threshold": 15,  # Diferencia máxima en ángulos
                "symmetry_threshold": 0.15,  # Asimetría máxima permitida
                "lateral_dev_threshold": 0.2,  # Desviación lateral máxima
                "velocity_ratio_threshold": 0.3,  # Diferencia máxima en velocidad
            }
        else:
            # Configuración por defecto para otros ejercicios
            self.specific_config = {
                "min_elbow_angle": 45,
                "max_elbow_angle": 175,
                "rom_threshold": 0.85,
                "bottom_diff_threshold": 0.2,
                "angle_diff_threshold": 15,
                "symmetry_threshold": 0.15,
                "lateral_dev_threshold": 0.2,
                "velocity_ratio_threshold": 0.3,
            }

    def calculate_angle(self, p1, p2, p3):
        """
        Calcula el ángulo entre tres puntos en el espacio 3D.

        Args:
            p1, p2, p3: Puntos 3D (listas o arrays con coordenadas x, y, z)

        Returns:
            Ángulo en grados
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        # Evitar errores de cálculo con vectores nulos
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)

        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return 0

        cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def analyze_movement_amplitude(self):
        """
        Analiza la amplitud del movimiento, comparando puntos máximos y mínimos.

        Returns:
            Diccionario con métricas y feedback sobre amplitud
        """
        # Extraer posición vertical (y) de las muñecas
        user_r_wrist_y = self.user_data["landmark_right_wrist_y"].values
        expert_r_wrist_y = self.expert_data["landmark_right_wrist_y"].values

        # Calcular altura mínima y máxima (invertir valores porque y crece hacia abajo)
        user_min_height = -np.min(user_r_wrist_y)
        user_max_height = -np.max(user_r_wrist_y)
        expert_min_height = -np.min(expert_r_wrist_y)
        expert_max_height = -np.max(expert_r_wrist_y)

        # Calcular rango de movimiento
        user_rom = user_max_height - user_min_height
        expert_rom = expert_max_height - expert_min_height
        rom_ratio = user_rom / expert_rom if expert_rom > 0 else 0

        # Analizar diferencia en el punto más bajo
        bottom_diff = (
            (user_min_height - expert_min_height) / expert_rom if expert_rom > 0 else 0
        )

        # Generar feedback
        feedback = {}
        if rom_ratio < self.specific_config["rom_threshold"]:
            feedback["amplitud"] = (
                "Tu rango de movimiento es insuficiente. Intenta bajar más las pesas y subirlas más arriba."
            )
        elif bottom_diff > self.specific_config["bottom_diff_threshold"]:
            feedback["posicion_baja"] = (
                "No estás bajando las pesas lo suficiente. Asegúrate de llegar hasta la altura de los hombros."
            )
        elif rom_ratio > 1.15:
            feedback["amplitud"] = (
                "Tu rango de movimiento es excesivo, lo que puede causar estrés en los hombros."
            )
        else:
            feedback["amplitud"] = "Buen rango de movimiento."

        metrics = {
            "rom_usuario": user_rom,
            "rom_experto": expert_rom,
            "rom_ratio": rom_ratio,
            "diferencia_posicion_baja": bottom_diff,
        }

        return {"metrics": metrics, "feedback": feedback}

    def analyze_elbow_angles(self):
        """
        Analiza los ángulos de los codos durante el movimiento.

        Returns:
            Diccionario con métricas y feedback sobre ángulos de codos
        """
        # Analizar ángulos de codo en el punto más bajo
        user_angles = []
        expert_angles = []

        # Calcular ángulos para cada frame
        for i in range(len(self.user_data)):
            try:
                # Extraer coordenadas para el codo derecho (hombro-codo-muñeca)
                user_shoulder = [
                    self.user_data.iloc[i]["landmark_right_shoulder_x"],
                    self.user_data.iloc[i]["landmark_right_shoulder_y"],
                    self.user_data.iloc[i]["landmark_right_shoulder_z"],
                ]
                user_elbow = [
                    self.user_data.iloc[i]["landmark_right_elbow_x"],
                    self.user_data.iloc[i]["landmark_right_elbow_y"],
                    self.user_data.iloc[i]["landmark_right_elbow_z"],
                ]
                user_wrist = [
                    self.user_data.iloc[i]["landmark_right_wrist_x"],
                    self.user_data.iloc[i]["landmark_right_wrist_y"],
                    self.user_data.iloc[i]["landmark_right_wrist_z"],
                ]

                expert_shoulder = [
                    self.expert_data.iloc[i]["landmark_right_shoulder_x"],
                    self.expert_data.iloc[i]["landmark_right_shoulder_y"],
                    self.expert_data.iloc[i]["landmark_right_shoulder_z"],
                ]
                expert_elbow = [
                    self.expert_data.iloc[i]["landmark_right_elbow_x"],
                    self.expert_data.iloc[i]["landmark_right_elbow_y"],
                    self.expert_data.iloc[i]["landmark_right_elbow_z"],
                ]
                expert_wrist = [
                    self.expert_data.iloc[i]["landmark_right_wrist_x"],
                    self.expert_data.iloc[i]["landmark_right_wrist_y"],
                    self.expert_data.iloc[i]["landmark_right_wrist_z"],
                ]

                # Verificar que no hay valores NaN
                if (
                    np.isnan(user_shoulder).any()
                    or np.isnan(user_elbow).any()
                    or np.isnan(user_wrist).any()
                    or np.isnan(expert_shoulder).any()
                    or np.isnan(expert_elbow).any()
                    or np.isnan(expert_wrist).any()
                ):
                    continue

                user_angle = self.calculate_angle(user_shoulder, user_elbow, user_wrist)
                expert_angle = self.calculate_angle(
                    expert_shoulder, expert_elbow, expert_wrist
                )

                user_angles.append(user_angle)
                expert_angles.append(expert_angle)
            except Exception as e:
                logger.warning(f"Error al calcular ángulo en frame {i}: {e}")

        # Convertir a arrays
        user_angles = np.array(user_angles)
        expert_angles = np.array(expert_angles)

        # Encontrar ángulos mínimos (punto más bajo de press)
        user_min_angle = np.min(user_angles) if len(user_angles) > 0 else 0
        expert_min_angle = np.min(expert_angles) if len(expert_angles) > 0 else 0
        angle_diff = user_min_angle - expert_min_angle

        # Generar feedback
        feedback = {}
        if angle_diff > self.specific_config["angle_diff_threshold"]:
            feedback["codos"] = (
                "Tus codos están demasiado abiertos en la posición baja. Intenta flexionarlos más para lograr una mejor mecánica."
            )
        elif angle_diff < -self.specific_config["angle_diff_threshold"]:
            feedback["codos"] = (
                "Tus codos están demasiado cerrados. Asegúrate de mantener una flexión adecuada para proteger tus articulaciones."
            )
        else:
            feedback["codos"] = "Buen ángulo de codos durante el ejercicio."

        metrics = {
            "angulo_minimo_usuario": user_min_angle,
            "angulo_minimo_experto": expert_min_angle,
            "diferencia_angulo": angle_diff,
        }

        return {"metrics": metrics, "feedback": feedback}

    def analyze_symmetry(self):
        """
        Analiza la simetría entre el lado izquierdo y derecho.

        Returns:
            Diccionario con métricas y feedback sobre simetría
        """
        # Extraer posiciones verticales de muñecas
        user_r_wrist_y = self.user_data["landmark_right_wrist_y"].values
        user_l_wrist_y = self.user_data["landmark_left_wrist_y"].values

        # Calcular diferencia promedio entre lados
        height_diff = np.mean(np.abs(user_r_wrist_y - user_l_wrist_y))

        # Normalizar diferencia respecto al rango de movimiento
        user_range = np.max(user_r_wrist_y) - np.min(user_r_wrist_y)
        normalized_diff = height_diff / user_range if user_range > 0 else 0

        # Generar feedback
        feedback = {}
        if normalized_diff > self.specific_config["symmetry_threshold"]:
            feedback["simetria"] = (
                "Hay una asimetría notable entre tu lado derecho e izquierdo. Enfócate en levantar ambos brazos por igual."
            )
        else:
            feedback["simetria"] = "Buena simetría bilateral en el movimiento."

        metrics = {
            "diferencia_altura": height_diff,
            "diferencia_normalizada": normalized_diff,
        }

        return {"metrics": metrics, "feedback": feedback}

    def analyze_movement_path(self):
        """
        Analiza la trayectoria de las muñecas durante el movimiento.

        Returns:
            Diccionario con métricas y feedback sobre la trayectoria
        """
        # Extraer trayectoria de las muñecas (vista frontal: x,y)
        user_r_wrist_x = self.user_data["landmark_right_wrist_x"].values
        user_r_wrist_y = self.user_data["landmark_right_wrist_y"].values

        expert_r_wrist_x = self.expert_data["landmark_right_wrist_x"].values
        expert_r_wrist_y = self.expert_data["landmark_right_wrist_y"].values

        # Calcular desviación lateral promedio
        lateral_deviation = np.mean(np.abs(user_r_wrist_x - expert_r_wrist_x))

        # Normalizar respecto al ancho del cuerpo
        shoulder_width = np.mean(
            np.abs(
                self.user_data["landmark_right_shoulder_x"]
                - self.user_data["landmark_left_shoulder_x"]
            )
        )
        normalized_deviation = (
            lateral_deviation / shoulder_width if shoulder_width > 0 else 0
        )

        # Generar feedback
        feedback = {}
        if normalized_deviation > self.specific_config["lateral_dev_threshold"]:
            feedback["trayectoria"] = (
                "Tu trayectoria se desvía lateralmente. Intenta mantener un movimiento más vertical."
            )
        else:
            feedback["trayectoria"] = "Buena trayectoria vertical del movimiento."

        metrics = {
            "desviacion_lateral": lateral_deviation,
            "desviacion_normalizada": normalized_deviation,
        }

        return {"metrics": metrics, "feedback": feedback}

    def analyze_speed(self):
        """
        Analiza la velocidad de ejecución en las fases concéntrica y excéntrica.

        Returns:
            Diccionario con métricas y feedback sobre velocidad
        """
        # Extraer posiciones verticales
        user_wrist_y = self.user_data["landmark_right_wrist_y"].values
        expert_wrist_y = self.expert_data["landmark_right_wrist_y"].values

        # Calcular velocidades (derivada de la posición)
        user_velocity = np.gradient(user_wrist_y)
        expert_velocity = np.gradient(expert_wrist_y)

        # Separar fases concéntrica (valores negativos - subida) y excéntrica (valores positivos - bajada)
        user_concentric = user_velocity[user_velocity < 0]
        user_eccentric = user_velocity[user_velocity > 0]

        expert_concentric = expert_velocity[expert_velocity < 0]
        expert_eccentric = expert_velocity[expert_velocity > 0]

        # Calcular velocidades promedio
        user_concentric_avg = (
            np.mean(np.abs(user_concentric)) if len(user_concentric) > 0 else 0
        )
        user_eccentric_avg = (
            np.mean(np.abs(user_eccentric)) if len(user_eccentric) > 0 else 0
        )

        expert_concentric_avg = (
            np.mean(np.abs(expert_concentric)) if len(expert_concentric) > 0 else 0
        )
        expert_eccentric_avg = (
            np.mean(np.abs(expert_eccentric)) if len(expert_eccentric) > 0 else 0
        )

        # Calcular ratios
        concentric_ratio = (
            user_concentric_avg / expert_concentric_avg
            if expert_concentric_avg > 0
            else 0
        )
        eccentric_ratio = (
            user_eccentric_avg / expert_eccentric_avg if expert_eccentric_avg > 0 else 0
        )

        # Generar feedback
        feedback = {}
        velocity_threshold = self.specific_config["velocity_ratio_threshold"]

        if concentric_ratio < (1 - velocity_threshold):
            feedback["velocidad_subida"] = (
                "La fase de subida es demasiado lenta. Intenta ser más explosivo en la fase concéntrica."
            )
        elif concentric_ratio > (1 + velocity_threshold):
            feedback["velocidad_subida"] = (
                "La fase de subida es demasiado rápida. Controla más el movimiento."
            )
        else:
            feedback["velocidad_subida"] = "Buena velocidad en la fase de subida."

        if eccentric_ratio < (1 - velocity_threshold):
            feedback["velocidad_bajada"] = (
                "La fase de bajada es demasiado lenta. Controla el descenso pero no lo ralentices en exceso."
            )
        elif eccentric_ratio > (1 + velocity_threshold):
            feedback["velocidad_bajada"] = (
                "La fase de bajada es demasiado rápida. Intenta controlar más el descenso de las pesas."
            )
        else:
            feedback["velocidad_bajada"] = "Buen control en la fase de bajada."

        metrics = {
            "velocidad_subida_usuario": user_concentric_avg,
            "velocidad_subida_experto": expert_concentric_avg,
            "ratio_subida": concentric_ratio,
            "velocidad_bajada_usuario": user_eccentric_avg,
            "velocidad_bajada_experto": expert_eccentric_avg,
            "ratio_bajada": eccentric_ratio,
        }

        return {"metrics": metrics, "feedback": feedback}

    def analyze_shoulder_position(self):
        """
        Analiza la posición de los hombros para detectar si están elevados o rotados.

        Returns:
            Diccionario con métricas y feedback sobre posición de hombros
        """
        # Extraer coordenadas verticales de los hombros
        user_r_shoulder_y = self.user_data["landmark_right_shoulder_y"].values
        user_l_shoulder_y = self.user_data["landmark_left_shoulder_y"].values

        expert_r_shoulder_y = self.expert_data["landmark_right_shoulder_y"].values
        expert_l_shoulder_y = self.expert_data["landmark_left_shoulder_y"].values

        # Calcular altura media de los hombros
        user_shoulder_height = (user_r_shoulder_y + user_l_shoulder_y) / 2
        expert_shoulder_height = (expert_r_shoulder_y + expert_l_shoulder_y) / 2

        # Calcular diferencia respecto a la altura de la cadera
        user_hip_y = (
            self.user_data["landmark_right_hip_y"].values
            + self.user_data["landmark_left_hip_y"].values
        ) / 2
        expert_hip_y = (
            self.expert_data["landmark_right_hip_y"].values
            + self.expert_data["landmark_left_hip_y"].values
        ) / 2

        # Calcular ratio altura hombros/caderas (menor valor = hombros más elevados)
        user_shoulder_hip_ratio = np.mean(user_shoulder_height / user_hip_y)
        expert_shoulder_hip_ratio = np.mean(expert_shoulder_height / expert_hip_y)

        # Diferencia en ratios
        ratio_diff = user_shoulder_hip_ratio - expert_shoulder_hip_ratio

        # Generar feedback
        feedback = {}
        if ratio_diff < -0.05:  # Usuario tiene hombros más elevados
            feedback["hombros"] = (
                "Tus hombros están demasiado elevados durante el ejercicio. Intenta relajarlos y mantenerlos bajos."
            )
        elif ratio_diff > 0.05:  # Usuario tiene hombros demasiado bajos
            feedback["hombros"] = (
                "Tus hombros están demasiado bajos. Mantén una postura más erguida durante el ejercicio."
            )
        else:
            feedback["hombros"] = "Buena posición de hombros durante el ejercicio."

        metrics = {
            "ratio_hombros_caderas_usuario": user_shoulder_hip_ratio,
            "ratio_hombros_caderas_experto": expert_shoulder_hip_ratio,
            "diferencia_ratio": ratio_diff,
        }

        return {"metrics": metrics, "feedback": feedback}

    def run_full_analysis(self):
        """
        Ejecuta todos los análisis disponibles y compila los resultados.

        Returns:
            Diccionario con todas las métricas y feedback
        """
        # Ejecutar todos los análisis
        amplitude_analysis = self.analyze_movement_amplitude()
        elbow_analysis = self.analyze_elbow_angles()
        symmetry_analysis = self.analyze_symmetry()
        path_analysis = self.analyze_movement_path()
        speed_analysis = self.analyze_speed()
        shoulder_analysis = self.analyze_shoulder_position()

        # Combinar métricas
        self.metrics = {
            "amplitud": amplitude_analysis["metrics"],
            "angulos_codo": elbow_analysis["metrics"],
            "simetria": symmetry_analysis["metrics"],
            "trayectoria": path_analysis["metrics"],
            "velocidad": speed_analysis["metrics"],
            "hombros": shoulder_analysis["metrics"],
        }

        # Combinar feedback
        self.feedback = {
            **amplitude_analysis["feedback"],
            **elbow_analysis["feedback"],
            **symmetry_analysis["feedback"],
            **path_analysis["feedback"],
            **speed_analysis["feedback"],
            **shoulder_analysis["feedback"],
        }

        # Calcular puntuación global
        self.calculate_overall_score()

        return {
            "metrics": self.metrics,
            "feedback": self.feedback,
            "score": self.overall_score,
            "level": self.level,
        }

    def calculate_overall_score(self):
        """
        Calcula una puntuación global basada en las métricas individuales.
        """
        scores = []

        # 1. Amplitud (0-100)
        rom_ratio = self.metrics["amplitud"]["rom_ratio"]
        rom_score = (
            min(100, 100 * rom_ratio)
            if rom_ratio <= 1
            else max(0, 100 - 50 * (rom_ratio - 1))
        )
        scores.append(rom_score)

        # 2. Ángulos de codo (0-100)
        angle_diff = abs(self.metrics["angulos_codo"]["diferencia_angulo"])
        angle_score = max(0, 100 - 3 * angle_diff)
        scores.append(angle_score)

        # 3. Simetría (0-100)
        sym_score = max(
            0, 100 - 300 * self.metrics["simetria"]["diferencia_normalizada"]
        )
        scores.append(sym_score)

        # 4. Trayectoria (0-100)
        path_score = max(
            0, 100 - 250 * self.metrics["trayectoria"]["desviacion_normalizada"]
        )
        scores.append(path_score)

        # 5. Velocidad (0-100)
        speed_concentric = 100 - 100 * abs(
            1 - self.metrics["velocidad"]["ratio_subida"]
        )
        speed_eccentric = 100 - 100 * abs(1 - self.metrics["velocidad"]["ratio_bajada"])
        speed_score = (speed_concentric + speed_eccentric) / 2
        scores.append(speed_score)

        # 6. Posición hombros (0-100)
        shoulder_diff = abs(self.metrics["hombros"]["diferencia_ratio"])
        shoulder_score = max(0, 100 - 1000 * shoulder_diff)
        scores.append(shoulder_score)

        # Calcular promedio
        self.overall_score = np.mean(scores)

        # Determinar nivel
        if self.overall_score >= 90:
            self.level = "Excelente"
        elif self.overall_score >= 80:
            self.level = "Muy bueno"
        elif self.overall_score >= 70:
            self.level = "Bueno"
        elif self.overall_score >= 60:
            self.level = "Aceptable"
        elif self.overall_score >= 50:
            self.level = "Necesita mejorar"
        else:
            self.level = "Principiante"

    def generate_report(self, output_path=None):
        """
        Genera un informe completo con los resultados del análisis.

        Args:
            output_path: Ruta donde guardar el informe (opcional)

        Returns:
            Diccionario con el informe completo
        """
        if not self.feedback:
            self.run_full_analysis()

        # Crear informe
        report = {
            "ejercicio": self.exercise_name,
            "puntuacion_global": round(self.overall_score, 1),
            "nivel": self.level,
            "areas_mejora": [],
            "puntos_fuertes": [],
            "feedback_detallado": self.feedback,
            "metricas": self.metrics,
            "recomendaciones": self._generate_recommendations(),
        }

        # Identificar áreas de mejora y puntos fuertes
        for category, message in self.feedback.items():
            if "Buen" in message or "Buena" in message:
                report["puntos_fuertes"].append(message)
            else:
                report["areas_mejora"].append(message)

        # Guardar informe si se especificó una ruta
        if output_path:
            # Crear el directorio si no existe
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)

            logger.info(f"Informe guardado en: {output_path}")

        return report

    def _generate_recommendations(self):
        """
        Genera recomendaciones específicas basadas en las áreas de mejora.

        Returns:
            Lista de recomendaciones
        """
        recommendations = []

        # Recomendaciones basadas en el análisis
        if "amplitud" in self.feedback and "insuficiente" in self.feedback["amplitud"]:
            recommendations.append(
                "Practica el movimiento completo con menos peso para mejorar la amplitud."
            )

        if "posicion_baja" in self.feedback:
            recommendations.append(
                "Trabaja en llevar las pesas hasta la altura de los hombros o ligeramente por debajo al bajar."
            )

        if "codos" in self.feedback and "abiertos" in self.feedback["codos"]:
            recommendations.append(
                "Realiza ejercicios de conciencia corporal frente al espejo para corregir la apertura de codos."
            )

        if "codos" in self.feedback and "cerrados" in self.feedback["codos"]:
            recommendations.append(
                "Intenta mantener los codos apuntando ligeramente hacia fuera durante el ejercicio, no demasiado pegados al cuerpo."
            )

        if "simetria" in self.feedback and "asimetría" in self.feedback["simetria"]:
            recommendations.append(
                "Realiza ejercicios unilaterales (con un brazo a la vez) para equilibrar la fuerza entre ambos lados."
            )

        if "trayectoria" in self.feedback and "desvía" in self.feedback["trayectoria"]:
            recommendations.append(
                "Practica frente a un espejo con una barra ligera o sin peso para corregir la trayectoria."
            )

        if (
            "velocidad_subida" in self.feedback
            and "lenta" in self.feedback["velocidad_subida"]
        ):
            recommendations.append(
                "Incorpora alguna serie con menor peso pero mayor velocidad controlada en la fase de subida."
            )

        if (
            "velocidad_bajada" in self.feedback
            and "rápida" in self.feedback["velocidad_bajada"]
        ):
            recommendations.append(
                "Cuenta mentalmente durante la bajada para asegurar un descenso controlado (aprox. 2-3 segundos)."
            )

        if "hombros" in self.feedback and "elevados" in self.feedback["hombros"]:
            recommendations.append(
                "Antes de cada repetición, realiza una exhalación consciente mientras bajas los hombros."
            )

        # Si no hay recomendaciones específicas pero la puntuación es < 80
        if not recommendations and self.overall_score < 80:
            recommendations.append(
                "Grábate realizando el ejercicio regularmente para revisar tu técnica."
            )
            recommendations.append(
                "Considera realizar el ejercicio con menos peso para enfocarte en la técnica."
            )

        # Si hay pocas recomendaciones específicas, agregar generales
        if len(recommendations) < 2:
            if self.overall_score < 70:
                recommendations.append(
                    "Considera algunas sesiones con un entrenador personal para perfeccionar tu técnica."
                )
            if self.overall_score < 60:
                recommendations.append(
                    "Comienza con variantes más sencillas del press militar, como el press sentado con respaldo."
                )

        return recommendations

    def visualize_analysis(self, output_dir=None):
        """
        Crea visualizaciones de los resultados del análisis.

        Args:
            output_dir: Directorio donde guardar las visualizaciones

        Returns:
            Lista de rutas a las visualizaciones generadas
        """
        if not self.metrics:
            self.run_full_analysis()

        visualizations = []

        # Crear directorio si no existe
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 1. Visualizar amplitud de movimiento
        plt.figure(figsize=(10, 6))

        # Extraer valores mínimos y máximos de la muñeca derecha
        user_wrist_y = self.user_data["landmark_right_wrist_y"].values
        expert_wrist_y = self.expert_data["landmark_right_wrist_y"].values

        plt.plot(-user_wrist_y, label="Usuario", color="blue")
        plt.plot(-expert_wrist_y, label="Experto", color="red")
        plt.axhline(-np.min(user_wrist_y), linestyle="--", color="blue", alpha=0.7)
        plt.axhline(-np.max(user_wrist_y), linestyle="--", color="blue", alpha=0.7)
        plt.axhline(-np.min(expert_wrist_y), linestyle="--", color="red", alpha=0.7)
        plt.axhline(-np.max(expert_wrist_y), linestyle="--", color="red", alpha=0.7)

        plt.title("Amplitud de Movimiento - Press Militar")
        plt.xlabel("Frame")
        plt.ylabel("Altura (normalizada)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            amplitude_path = os.path.join(output_dir, "amplitud_movimiento.png")
            plt.savefig(amplitude_path, dpi=100, bbox_inches="tight")
            visualizations.append(amplitude_path)

        plt.close()

        # 2. Visualizar ángulos de codo
        plt.figure(figsize=(10, 6))

        # Calcular ángulos para cada frame
        user_angles = []
        expert_angles = []

        for i in range(len(self.user_data)):
            try:
                # Extraer coordenadas para el codo derecho
                user_shoulder = [
                    self.user_data.iloc[i]["landmark_right_shoulder_x"],
                    self.user_data.iloc[i]["landmark_right_shoulder_y"],
                    self.user_data.iloc[i]["landmark_right_shoulder_z"],
                ]
                user_elbow = [
                    self.user_data.iloc[i]["landmark_right_elbow_x"],
                    self.user_data.iloc[i]["landmark_right_elbow_y"],
                    self.user_data.iloc[i]["landmark_right_elbow_z"],
                ]
                user_wrist = [
                    self.user_data.iloc[i]["landmark_right_wrist_x"],
                    self.user_data.iloc[i]["landmark_right_wrist_y"],
                    self.user_data.iloc[i]["landmark_right_wrist_z"],
                ]

                expert_shoulder = [
                    self.expert_data.iloc[i]["landmark_right_shoulder_x"],
                    self.expert_data.iloc[i]["landmark_right_shoulder_y"],
                    self.expert_data.iloc[i]["landmark_right_shoulder_z"],
                ]
                expert_elbow = [
                    self.expert_data.iloc[i]["landmark_right_elbow_x"],
                    self.expert_data.iloc[i]["landmark_right_elbow_y"],
                    self.expert_data.iloc[i]["landmark_right_elbow_z"],
                ]
                expert_wrist = [
                    self.expert_data.iloc[i]["landmark_right_wrist_x"],
                    self.expert_data.iloc[i]["landmark_right_wrist_y"],
                    self.expert_data.iloc[i]["landmark_right_wrist_z"],
                ]

                # Verificar que no hay valores NaN
                if (
                    np.isnan(user_shoulder).any()
                    or np.isnan(user_elbow).any()
                    or np.isnan(user_wrist).any()
                    or np.isnan(expert_shoulder).any()
                    or np.isnan(expert_elbow).any()
                    or np.isnan(expert_wrist).any()
                ):
                    continue

                user_angle = self.calculate_angle(user_shoulder, user_elbow, user_wrist)
                expert_angle = self.calculate_angle(
                    expert_shoulder, expert_elbow, expert_wrist
                )

                user_angles.append(user_angle)
                expert_angles.append(expert_angle)
            except Exception as e:
                pass

        plt.plot(user_angles, label="Usuario", color="blue")
        plt.plot(expert_angles, label="Experto", color="red")

        plt.title("Ángulos de Codo durante el Press Militar")
        plt.xlabel("Frame")
        plt.ylabel("Ángulo (grados)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            angles_path = os.path.join(output_dir, "angulos_codo.png")
            plt.savefig(angles_path, dpi=100, bbox_inches="tight")
            visualizations.append(angles_path)

        plt.close()

        # 3. Visualizar trayectorias (vista frontal)
        plt.figure(figsize=(10, 8))

        # Extraer trayectorias
        user_r_wrist_x = self.user_data["landmark_right_wrist_x"].values
        user_r_wrist_y = self.user_data["landmark_right_wrist_y"].values

        expert_r_wrist_x = self.expert_data["landmark_right_wrist_x"].values
        expert_r_wrist_y = self.expert_data["landmark_right_wrist_y"].values

        plt.scatter(
            user_r_wrist_x,
            -user_r_wrist_y,
            s=10,
            alpha=0.7,
            color="blue",
            label="Usuario",
        )
        plt.plot(user_r_wrist_x, -user_r_wrist_y, color="blue", alpha=0.4)

        plt.scatter(
            expert_r_wrist_x,
            -expert_r_wrist_y,
            s=10,
            alpha=0.7,
            color="red",
            label="Experto",
        )
        plt.plot(expert_r_wrist_x, -expert_r_wrist_y, color="red", alpha=0.4)

        plt.title("Trayectoria de la Muñeca Derecha - Press Militar")
        plt.xlabel("X (lateral)")
        plt.ylabel("Y (vertical)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            trajectory_path = os.path.join(output_dir, "trayectoria_frontal.png")
            plt.savefig(trajectory_path, dpi=100, bbox_inches="tight")
            visualizations.append(trajectory_path)

        plt.close()

        # 4. Visualizar simetría bilateral
        plt.figure(figsize=(10, 6))

        diff_y = abs(
            self.user_data["landmark_right_wrist_y"].values
            - self.user_data["landmark_left_wrist_y"].values
        )

        plt.plot(diff_y, label="Diferencia entre muñecas", color="purple")
        plt.axhline(
            y=self.specific_config["symmetry_threshold"],
            linestyle="--",
            color="red",
            label=f'Umbral de asimetría ({self.specific_config["symmetry_threshold"]})',
        )

        plt.title("Simetría Bilateral - Press Militar")
        plt.xlabel("Frame")
        plt.ylabel("Diferencia de altura (valor absoluto)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            symmetry_path = os.path.join(output_dir, "simetria_bilateral.png")
            plt.savefig(symmetry_path, dpi=100, bbox_inches="tight")
            visualizations.append(symmetry_path)

        plt.close()

        # 5. Visualizar velocidad
        plt.figure(figsize=(10, 6))

        # Calcular velocidades
        user_velocity = np.gradient(self.user_data["landmark_right_wrist_y"].values)
        expert_velocity = np.gradient(self.expert_data["landmark_right_wrist_y"].values)

        plt.plot(user_velocity, label="Usuario", color="blue")
        plt.plot(expert_velocity, label="Experto", color="red")

        plt.title("Velocidad Vertical - Press Militar")
        plt.xlabel("Frame")
        plt.ylabel("Velocidad (unidades/frame)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            velocity_path = os.path.join(output_dir, "velocidad.png")
            plt.savefig(velocity_path, dpi=100, bbox_inches="tight")
            visualizations.append(velocity_path)

        plt.close()

        # 6. Visualizar puntuación de las diferentes categorías
        plt.figure(figsize=(10, 6))

        categories = [
            "Amplitud",
            "Ángulos\nCodos",
            "Simetría",
            "Trayectoria",
            "Velocidad",
            "Posición\nHombros",
            "Global",
        ]

        # Calcular puntuaciones individuales
        rom_ratio = self.metrics["amplitud"]["rom_ratio"]
        rom_score = (
            min(100, 100 * rom_ratio)
            if rom_ratio <= 1
            else max(0, 100 - 50 * (rom_ratio - 1))
        )

        angle_diff = abs(self.metrics["angulos_codo"]["diferencia_angulo"])
        angle_score = max(0, 100 - 3 * angle_diff)

        sym_score = max(
            0, 100 - 300 * self.metrics["simetria"]["diferencia_normalizada"]
        )

        path_score = max(
            0, 100 - 250 * self.metrics["trayectoria"]["desviacion_normalizada"]
        )

        speed_concentric = 100 - 100 * abs(
            1 - self.metrics["velocidad"]["ratio_subida"]
        )
        speed_eccentric = 100 - 100 * abs(1 - self.metrics["velocidad"]["ratio_bajada"])
        speed_score = (speed_concentric + speed_eccentric) / 2

        shoulder_diff = abs(self.metrics["hombros"]["diferencia_ratio"])
        shoulder_score = max(0, 100 - 1000 * shoulder_diff)

        scores = [
            rom_score,
            angle_score,
            sym_score,
            path_score,
            speed_score,
            shoulder_score,
            self.overall_score,
        ]

        # Definir colores según puntuación
        colors = []
        for score in scores:
            if score >= 90:
                colors.append("#27ae60")  # Verde fuerte
            elif score >= 70:
                colors.append("#2ecc71")  # Verde
            elif score >= 50:
                colors.append("#f39c12")  # Naranja
            else:
                colors.append("#e74c3c")  # Rojo

        bars = plt.bar(categories, scores, color=colors)

        # Añadir etiquetas con valores
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.axhline(y=50, color="r", linestyle="-", alpha=0.3, label="Deficiente")
        plt.axhline(y=70, color="y", linestyle="-", alpha=0.3, label="Aceptable")
        plt.axhline(y=90, color="g", linestyle="-", alpha=0.3, label="Excelente")

        plt.title("Puntuación por Categoría - Press Militar")
        plt.ylabel("Puntuación (0-100)")
        plt.ylim(0, 105)
        plt.legend(loc="lower right")

        if output_dir:
            score_path = os.path.join(output_dir, "puntuacion_categorias.png")
            plt.savefig(score_path, dpi=100, bbox_inches="tight")
            visualizations.append(score_path)

        plt.close()

        # 7. Visualizar gráfico de radar
        try:
            plt.figure(figsize=(10, 8))

            # Preparar datos para el gráfico de radar
            categories = [
                "Amplitud",
                "Ángulos\nCodos",
                "Simetría",
                "Trayectoria",
                "Velocidad",
                "Posición\nHombros",
            ]

            scores_normalized = [
                rom_score / 100,
                angle_score / 100,
                sym_score / 100,
                path_score / 100,
                speed_score / 100,
                shoulder_score / 100,
            ]

            # Cerrar el polígono repitiendo el primer valor
            scores_normalized = np.concatenate(
                (scores_normalized, [scores_normalized[0]])
            )
            categories = np.concatenate((categories, [categories[0]]))

            # Calcular ángulos para cada categoría
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

            # Crear el gráfico de radar
            ax = plt.subplot(111, polar=True)
            ax.fill(angles, scores_normalized, color="blue", alpha=0.25)
            ax.plot(angles, scores_normalized, color="blue", linewidth=2)

            # Añadir líneas de referencia
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories[:-1])

            # Configurar límites y etiquetas
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["25", "50", "75", "100"])

            # Añadir título
            plt.title("Análisis de Técnica - Press Militar", size=15, y=1.1)

            if output_dir:
                radar_path = os.path.join(output_dir, "analisis_radar.png")
                plt.savefig(radar_path, dpi=100, bbox_inches="tight")
                visualizations.append(radar_path)

            plt.close()

        except Exception as e:
            logger.warning(f"Error al crear gráfico de radar: {e}")

        # 8. Crear resumen visual
        try:
            plt.figure(figsize=(12, 8))

            # Configurar el área de dibujo
            plt.text(
                0.5,
                0.95,
                f"ANÁLISIS DE TÉCNICA - {self.exercise_name.upper()}",
                fontsize=18,
                ha="center",
                va="top",
                fontweight="bold",
            )

            plt.text(
                0.5,
                0.88,
                f"Puntuación Global: {self.overall_score:.1f}/100 - Nivel: {self.level}",
                fontsize=16,
                ha="center",
                va="top",
            )

            # Generar las áreas de mejora
            areas_mejora = []
            for category, message in self.feedback.items():
                if "Buen" not in message and "Buena" not in message:
                    areas_mejora.append(message)

            # Añadir áreas de mejora
            plt.text(0.05, 0.8, "ÁREAS DE MEJORA:", fontsize=14, fontweight="bold")

            y_pos = 0.75
            for i, area in enumerate(areas_mejora[:5]):  # Limitar a 5 elementos
                plt.text(0.07, y_pos - i * 0.05, f"• {area}", fontsize=12)

            # Añadir recomendaciones
            plt.text(0.05, 0.5, "RECOMENDACIONES:", fontsize=14, fontweight="bold")

            y_pos = 0.45
            for i, rec in enumerate(
                self._generate_recommendations()[:5]
            ):  # Limitar a 5 elementos
                plt.text(0.07, y_pos - i * 0.05, f"• {rec}", fontsize=12)

            # Generar puntos fuertes
            puntos_fuertes = []
            for category, message in self.feedback.items():
                if "Buen" in message or "Buena" in message:
                    puntos_fuertes.append(message)

            # Añadir puntos fuertes
            plt.text(0.05, 0.2, "PUNTOS FUERTES:", fontsize=14, fontweight="bold")

            y_pos = 0.15
            for i, punto in enumerate(puntos_fuertes[:3]):  # Limitar a 3 elementos
                plt.text(0.07, y_pos - i * 0.05, f"• {punto}", fontsize=12)

            # Configuración final
            plt.axis("off")

            if output_dir:
                summary_path = os.path.join(output_dir, "resumen_visual.png")
                plt.savefig(summary_path, dpi=100, bbox_inches="tight")
                visualizations.append(summary_path)

            plt.close()

        except Exception as e:
            logger.warning(f"Error al crear resumen visual: {e}")

        return visualizations
