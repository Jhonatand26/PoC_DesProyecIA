"""
mlflow_server.py — Servidor MLflow para tracking de experimentos
PoC Asistente Fitosanitario con IA | UAO 2026

Responsabilidad unica: arrancar el servidor MLflow con configuracion
leida desde variables de entorno (.env).

Uso local:
    uv run python mlflow_server.py

Uso en Docker Compose (Modulo 4):
    command: ["uv", "run", "python", "mlflow_server.py"]

Variables de entorno requeridas (.env):
    MLFLOW_TRACKING_URI  — URI del servidor (default: http://localhost:5000)
    MLFLOW_HOST          — Host donde escucha (default: 0.0.0.0)
    MLFLOW_PORT          — Puerto del servidor (default: 5000)
    MLFLOW_BACKEND_URI   — Backend de almacenamiento (default: ./mlruns)
    MLFLOW_ARTIFACT_ROOT — Carpeta de artefactos (default: ./mlartifacts)
"""

import os
import subprocess
import sys
import logging

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuracion de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MLFLOW-SERVER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_config() -> dict:
    """
    Lee la configuracion del servidor MLflow desde variables de entorno.

    Returns:
        dict: Configuracion con host, port, backend_uri y artifact_root.
    """
    return {
        "host":          os.getenv("MLFLOW_HOST",          "0.0.0.0"),
        "port":          os.getenv("MLFLOW_PORT",          "5000"),
        "backend_uri":   os.getenv("MLFLOW_BACKEND_URI",   "./mlruns"),
        "artifact_root": os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts"),
    }


def serve() -> None:
    """
    Arranca el servidor MLflow con la configuracion del entorno.

    Construye y ejecuta el comando mlflow server con los parametros
    leidos desde .env. El proceso reemplaza al proceso Python actual
    (os.execvp) para que Docker maneje correctamente las senales
    de parada (SIGTERM, Ctrl+C).
    """
    config = get_config()

    logger.info("Iniciando servidor MLflow...")
    logger.info("Host:           %s", config["host"])
    logger.info("Puerto:         %s", config["port"])
    logger.info("Backend URI:    %s", config["backend_uri"])
    logger.info("Artifact Root:  %s", config["artifact_root"])
    logger.info(
        "UI disponible en: http://localhost:%s", config["port"]
    )

    cmd = [
        "mlflow", "server",
        "--host",                config["host"],
        "--port",                config["port"],
        "--backend-store-uri",   config["backend_uri"],
        "--default-artifact-root", config["artifact_root"],
    ]

    logger.info("Comando: %s", " ".join(cmd))

    try:
        # subprocess.run mantiene el proceso vivo y propaga Ctrl+C
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except FileNotFoundError:
        logger.error(
            "MLflow no encontrado. Asegurate de haber corrido: uv sync"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Servidor MLflow detenido.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        logger.error("Error al iniciar MLflow: %s", e)
        sys.exit(e.returncode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    serve()