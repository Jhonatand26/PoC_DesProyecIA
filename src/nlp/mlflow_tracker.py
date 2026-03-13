"""
Modulo de tracking MLflow para el servicio NLP.
Registra experimentos, parametros y metricas de cada
llamada a GPT-5 Nano para trazabilidad del modulo NLP.
Usa almacenamiento local en carpeta mlruns/.
"""

import os
import mlflow

EXPERIMENT_NAME = "nlp-gpt5-nano-v2"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def setup_experiment() -> None:
    """
    Configura el experimento MLflow para el modulo NLP.

    Usa MLFLOW_TRACKING_URI del entorno para conectar al servidor MLflow.
    En Docker apunta a http://mlflow:5000, en local a http://localhost:5000.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        print(f"Warning: No se pudo configurar MLflow ({e})")


def log_recommendation(
    class_name: str,
    confidence: float,
    prompt: str,
    recommendation: str,
    success: bool,
    error: str = "",
    latency_ms: float = 0.0,
) -> None:
    """
    Registra una llamada a GPT-5 Nano en MLflow.

    Args:
        class_name (str): Clase detectada por el modelo CV.
        confidence (float): Confianza del diagnostico CV.
        prompt (str): Prompt enviado a GPT-5 Nano.
        recommendation (str): Respuesta recibida de GPT-5 Nano.
        success (bool): True si la llamada fue exitosa.
        error (str): Mensaje de error si success es False.
        latency_ms (float): Tiempo de respuesta en milisegundos.
    """
    try:
        setup_experiment()
        with mlflow.start_run():
            mlflow.log_param("class_name", class_name)
            mlflow.log_param("confidence", confidence)
            mlflow.log_param("model", "gpt-5-nano")
            mlflow.log_param("success", success)

            mlflow.log_metric("confidence_score", confidence)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("recommendation_length", len(recommendation))
            mlflow.log_metric("success_flag", 1 if success else 0)

            if error:
                mlflow.log_text(error, "error.txt")
            if prompt:
                mlflow.log_text(prompt, "prompt.txt")
            if recommendation:
                mlflow.log_text(recommendation, "recommendation.txt")
    except Exception as e:
        print(f"Warning: No se pudo conectar a MLflow ({e})")
