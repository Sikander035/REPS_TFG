# backend/src/utils/matplotlib_config.py
"""
Configuración thread-safe de matplotlib para evitar conflictos en FastAPI
"""
import matplotlib
import matplotlib.pyplot as plt
import logging
import threading
import os

logger = logging.getLogger(__name__)

# Lock para thread safety
_config_lock = threading.Lock()
_configured = False


def configure_matplotlib_for_threading():
    """
    Configura matplotlib para uso thread-safe en FastAPI.
    DEBE llamarse antes de cualquier import/uso de matplotlib.
    """
    global _configured

    with _config_lock:
        if _configured:
            return

        try:
            # Forzar backend no-interactivo ANTES de cualquier plot
            matplotlib.use("Agg", force=True)

            # Configurar matplotlib para threading
            matplotlib.rcParams.update(
                {
                    "figure.max_open_warning": 0,  # Deshabilitar warnings de figuras abiertas
                    "font.size": 10,
                    "figure.figsize": [10, 6],
                    "savefig.dpi": 100,
                    "savefig.bbox": "tight",
                }
            )

            # Deshabilitar modo interactivo
            plt.ioff()

            # Configurar logs
            matplotlib.set_loglevel("WARNING")

            logger.info("✅ Matplotlib configurado para threading (backend: Agg)")
            _configured = True

        except Exception as e:
            logger.error(f"❌ Error configurando matplotlib: {e}")
            raise


def cleanup_matplotlib_resources():
    """
    Limpia recursos de matplotlib para evitar memory leaks.
    Llamar después de cada operación de plotting.
    """
    try:
        # Cerrar todas las figuras
        plt.close("all")

        # Limpiar cache de figuras
        if hasattr(plt, "_pylab_helpers"):
            plt.gcf().clear()

        # Forzar garbage collection de matplotlib
        import gc

        gc.collect()

        logger.debug("🧹 Recursos matplotlib limpiados")

    except Exception as e:
        logger.warning(f"⚠️ Error limpiando matplotlib: {e}")


def ensure_matplotlib_thread_safety():
    """
    Asegura que matplotlib esté configurado correctamente en el thread actual.
    Llamar al inicio de funciones que usen matplotlib en threads.
    """
    try:
        # Re-configurar backend si es necesario
        current_backend = matplotlib.get_backend()
        if current_backend != "Agg":
            matplotlib.use("Agg", force=True)
            logger.debug(f"🔧 Backend cambiado de {current_backend} a Agg en thread")

        # Asegurar modo no-interactivo
        if plt.isinteractive():
            plt.ioff()
            logger.debug("🔧 Modo interactivo deshabilitado en thread")

    except Exception as e:
        logger.warning(f"⚠️ Error asegurando thread safety: {e}")


class ThreadSafeMatplotlib:
    """
    Context manager para uso thread-safe de matplotlib.

    Usage:
        with ThreadSafeMatplotlib():
            plt.figure()
            # ... código de plotting ...
            plt.savefig(path)
    """

    def __enter__(self):
        ensure_matplotlib_thread_safety()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_matplotlib_resources()


# Auto-configurar al importar este módulo
configure_matplotlib_for_threading()
