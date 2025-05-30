# test_sensitivity.py - Script para probar sensibilidad de métricas
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Añadir rutas para importar módulos del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Importar funciones de análisis
from src.feedback.analysis_report import (
    analyze_movement_amplitude,
    analyze_elbow_abduction_angle,
    analyze_symmetry,
    analyze_movement_trajectory_3d,
    analyze_speed,
    analyze_scapular_stability,
)
from src.utils.analysis_utils import get_exercise_config


def load_sample_data(base_output_dir, exercise_name="press_militar"):
    """
    Carga datos sincronizados y normalizados del ejercicio.

    Args:
        base_output_dir: Directorio base de resultados
        exercise_name: Nombre del ejercicio

    Returns:
        tuple: (user_data, expert_data) o None si no se encuentran
    """
    output_dir = os.path.join(base_output_dir, f"resultados_{exercise_name}")

    # Buscar archivos sincronizados
    user_sync_file = os.path.join(output_dir, f"{exercise_name}_user_synchronized.csv")
    expert_aligned_file = os.path.join(
        output_dir, f"{exercise_name}_expert_aligned.csv"
    )

    # Alternativas si no existen los archivos principales
    alternative_files = [
        (f"{exercise_name}_expert_normalized.csv", "normalized"),
        (f"{exercise_name}_expert_synchronized.csv", "synchronized"),
    ]

    try:
        # Intentar cargar archivo del usuario
        if os.path.exists(user_sync_file):
            user_data = pd.read_csv(user_sync_file)
            print(f"✓ Usuario cargado: {user_sync_file}")
        else:
            print(f"❌ No encontrado: {user_sync_file}")
            return None, None

        # Intentar cargar archivo del experto (prioridad: aligned > normalized > synchronized)
        expert_data = None
        expert_file_used = None

        for filename, desc in [("expert_aligned.csv", "aligned")] + alternative_files:
            expert_file = os.path.join(output_dir, f"{exercise_name}_{filename}")
            if os.path.exists(expert_file):
                expert_data = pd.read_csv(expert_file)
                expert_file_used = expert_file
                print(f"✓ Experto cargado ({desc}): {expert_file}")
                break

        if expert_data is None:
            print(f"❌ No se encontró ningún archivo del experto en: {output_dir}")
            return None, None

        # Verificar que ambos DataFrames tienen la misma longitud
        if len(user_data) != len(expert_data):
            min_len = min(len(user_data), len(expert_data))
            print(f"⚠️  Longitudes diferentes. Recortando a {min_len} frames")
            user_data = user_data.iloc[:min_len]
            expert_data = expert_data.iloc[:min_len]

        print(f"📊 Datos cargados: {len(user_data)} frames")
        return user_data, expert_data

    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None, None


def create_test_configurations():
    """
    Crea diferentes configuraciones de sensibilidad para probar.

    Returns:
        dict: Diccionario con configuraciones de prueba
    """
    return {
        "baseline": {
            "name": "🟢 Baseline (Normal)",
            "sensitivity_factors": {
                "amplitud": 1.0,
                "abduccion_codos": 1.0,
                "simetria": 1.0,
                "trayectoria": 1.0,
                "velocidad": 1.0,
                "estabilidad_escapular": 1.0,
            },
        },
        "high_sensitivity": {
            "name": "🔴 Alta Sensibilidad (Estricto)",
            "sensitivity_factors": {
                "amplitud": 2.0,
                "abduccion_codos": 2.0,
                "simetria": 2.0,
                "trayectoria": 2.0,
                "velocidad": 2.0,
                "estabilidad_escapular": 2.0,
            },
        },
        "low_sensitivity": {
            "name": "🟡 Baja Sensibilidad (Permisivo)",
            "sensitivity_factors": {
                "amplitud": 0.5,
                "abduccion_codos": 0.5,
                "simetria": 0.5,
                "trayectoria": 0.5,
                "velocidad": 0.5,
                "estabilidad_escapular": 0.5,
            },
        },
        "scapular_focused": {
            "name": "🎯 Solo Escapular Estricto",
            "sensitivity_factors": {
                "amplitud": 1.0,
                "abduccion_codos": 1.0,
                "simetria": 1.0,
                "trayectoria": 1.0,
                "velocidad": 1.0,
                "estabilidad_escapular": 3.0,  # Solo escapular muy estricto
            },
        },
        "position_focused": {
            "name": "💪 Posición Estricta (Amplitud + Abducción)",
            "sensitivity_factors": {
                "amplitud": 2.5,
                "abduccion_codos": 2.5,
                "simetria": 1.0,
                "trayectoria": 1.0,
                "velocidad": 1.0,
                "estabilidad_escapular": 1.0,
            },
        },
    }


def run_analysis_with_config(
    user_data, expert_data, config_name, sensitivity_factors, base_config
):
    """
    Ejecuta todos los análisis con una configuración específica.

    Args:
        user_data: DataFrame del usuario
        expert_data: DataFrame del experto
        config_name: Nombre de la configuración
        sensitivity_factors: Factores de sensibilidad
        base_config: Configuración base del ejercicio

    Returns:
        dict: Resultados de todos los análisis
    """
    # Crear configuración modificada
    exercise_config = base_config.copy()
    exercise_config["sensitivity_factors"] = sensitivity_factors

    # Ejecutar todos los análisis
    analyses = {
        "amplitud": analyze_movement_amplitude(user_data, expert_data, exercise_config),
        "abduccion": analyze_elbow_abduction_angle(
            user_data, expert_data, exercise_config
        ),
        "simetria": analyze_symmetry(user_data, expert_data, exercise_config),
        "trayectoria": analyze_movement_trajectory_3d(
            user_data, expert_data, exercise_config
        ),
        "velocidad": analyze_speed(user_data, expert_data, exercise_config),
        "escapular": analyze_scapular_stability(
            user_data, expert_data, exercise_config
        ),
    }

    # Extraer scores y feedback principal
    results = {
        "config_name": config_name,
        "sensitivity_factors": sensitivity_factors,
        "scores": {},
        "feedback_summary": {},
    }

    for metric, analysis in analyses.items():
        results["scores"][metric] = analysis["score"]

        # Extraer feedback principal (primera entrada del feedback dict)
        feedback_dict = analysis["feedback"]
        if feedback_dict:
            first_feedback = list(feedback_dict.values())[0]
            # Truncar feedback para mostrar solo primeras palabras
            short_feedback = (
                first_feedback[:50] + "..."
                if len(first_feedback) > 50
                else first_feedback
            )
            results["feedback_summary"][metric] = short_feedback
        else:
            results["feedback_summary"][metric] = "Sin feedback"

    return results


def print_comparison_table(all_results):
    """
    Imprime una tabla comparativa de los resultados.

    Args:
        all_results: Lista de resultados de diferentes configuraciones
    """
    print("\n" + "=" * 120)
    print("📊 TABLA COMPARATIVA DE SENSIBILIDAD")
    print("=" * 120)

    # Header
    metrics = [
        "amplitud",
        "abduccion",
        "simetria",
        "trayectoria",
        "velocidad",
        "escapular",
    ]
    print(f"{'Configuración':<30} | ", end="")
    for metric in metrics:
        print(f"{metric.capitalize():<12} | ", end="")
    print()
    print("-" * 120)

    # Datos
    for result in all_results:
        config_name = result["config_name"][:28]  # Truncar nombre
        print(f"{config_name:<30} | ", end="")

        for metric in metrics:
            score = result["scores"].get(metric, 0)
            if score >= 80:
                color = "✅"
            elif score >= 60:
                color = "🟡"
            else:
                color = "❌"
            print(f"{color}{score:5.1f}    | ", end="")
        print()

    print("-" * 120)


def print_sensitivity_analysis(all_results):
    """
    Analiza y muestra cuáles métricas son más sensibles a cambios.

    Args:
        all_results: Lista de resultados de diferentes configuraciones
    """
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DE SENSIBILIDAD")
    print("=" * 80)

    # Encontrar configuración baseline
    baseline_scores = None
    for result in all_results:
        if (
            "baseline" in result["config_name"].lower()
            or "normal" in result["config_name"].lower()
        ):
            baseline_scores = result["scores"]
            break

    if not baseline_scores:
        print("❌ No se encontró configuración baseline para comparar")
        return

    metrics = [
        "amplitud",
        "abduccion",
        "simetria",
        "trayectoria",
        "velocidad",
        "escapular",
    ]
    sensitivities = []

    for metric in metrics:
        baseline_score = baseline_scores[metric]
        max_diff = 0

        # Calcular máxima diferencia con respecto al baseline
        for result in all_results:
            if result == baseline_scores:
                continue
            current_score = result["scores"][metric]
            diff = abs(current_score - baseline_score)
            max_diff = max(max_diff, diff)

        sensitivities.append((metric, max_diff))

    # Ordenar por sensibilidad (mayor diferencia = más sensible)
    sensitivities.sort(key=lambda x: x[1], reverse=True)

    print("Ranking de sensibilidad (Mayor → Menor):")
    print("-" * 50)

    for i, (metric, max_diff) in enumerate(sensitivities, 1):
        baseline_score = baseline_scores[metric]

        if max_diff > 30:
            level = "🔴 MUY ALTA"
        elif max_diff > 15:
            level = "🟡 ALTA"
        elif max_diff > 8:
            level = "🟢 MEDIA"
        else:
            level = "⚪ BAJA"

        print(
            f"{i}. {metric.capitalize():<15} | {level:<12} | "
            f"Baseline: {baseline_score:5.1f} | Max Δ: {max_diff:5.1f}"
        )


def main():
    """
    Función principal que ejecuta todas las pruebas de sensibilidad.
    """
    print("🧪 PRUEBAS DE SENSIBILIDAD DE MÉTRICAS DE ANÁLISIS")
    print("=" * 60)

    # Configuración de rutas (ajustar según tu estructura)
    BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "media", "output")
    CONFIG_PATH = os.path.join(BASE_DIR, "src", "config", "config.json")
    EXERCISE_NAME = "press_militar"

    print(f"📁 Directorio base: {BASE_OUTPUT_DIR}")
    print(f"⚙️  Configuración: {CONFIG_PATH}")
    print(f"🏋️  Ejercicio: {EXERCISE_NAME}")
    print()

    # Cargar datos
    print("📥 Cargando datos...")
    user_data, expert_data = load_sample_data(BASE_OUTPUT_DIR, EXERCISE_NAME)

    if user_data is None or expert_data is None:
        print("❌ No se pudieron cargar los datos. Verifica las rutas:")
        print(
            f"   - Buscar en: {os.path.join(BASE_OUTPUT_DIR, f'resultados_{EXERCISE_NAME}')}"
        )
        print(f"   - Archivos esperados:")
        print(f"     • {EXERCISE_NAME}_user_synchronized.csv")
        print(f"     • {EXERCISE_NAME}_expert_aligned.csv (o normalized/synchronized)")
        return

    # Cargar configuración base
    print("\n⚙️  Cargando configuración base...")
    try:
        base_config = get_exercise_config(EXERCISE_NAME, CONFIG_PATH)
        print("✓ Configuración cargada correctamente")
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return

    # Crear configuraciones de prueba
    test_configs = create_test_configurations()
    print(f"\n🧪 Ejecutando {len(test_configs)} configuraciones de prueba...")

    # Ejecutar análisis con cada configuración
    all_results = []

    for config_key, config_data in test_configs.items():
        config_name = config_data["name"]
        sensitivity_factors = config_data["sensitivity_factors"]

        print(f"\n⚡ Ejecutando: {config_name}")

        try:
            result = run_analysis_with_config(
                user_data, expert_data, config_name, sensitivity_factors, base_config
            )
            all_results.append(result)
            print(f"✓ Completado")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Mostrar resultados
    if all_results:
        print_comparison_table(all_results)
        print_sensitivity_analysis(all_results)

        # Guardar resultados detallados (opcional)
        output_file = os.path.join(BASE_OUTPUT_DIR, "sensitivity_test_results.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados detallados guardados en: {output_file}")
        except Exception as e:
            print(f"\n⚠️  No se pudieron guardar resultados: {e}")

    else:
        print("\n❌ No se pudieron ejecutar las pruebas de sensibilidad")

    print("\n✅ Pruebas de sensibilidad completadas!")


if __name__ == "__main__":
    main()
