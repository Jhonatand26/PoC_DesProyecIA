"""
Módulo para el registro de experimentos y modelos en MLflow.
"""
import os
import mlflow
import mlflow.pyfunc


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "plant-disease-classification"
REGISTERED_MODEL_NAME = "mobilenet-plant-disease"


def log_inference(
    model_name: str,
    cache_dir: str,
    predicted_class: str,
    confidence: float,
    inference_time_ms: float
) -> None:
    """
    Registra parámetros, métricas y resultado de una inferencia en MLflow.

    :param model_name: Nombre del modelo usado en HuggingFace.
    :param cache_dir: Directorio de caché del modelo.
    :param predicted_class: Clase predicha por el modelo.
    :param confidence: Confianza de la predicción (0.0 - 1.0).
    :param inference_time_ms: Tiempo de inferencia en milisegundos.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": model_name,
            "cache_dir": cache_dir,
        })
        mlflow.log_metrics({
            "confidence": confidence,
            "inference_time_ms": inference_time_ms,
        })
        mlflow.set_tag("predicted_class", predicted_class)


def register_model(run_id: str) -> None:
    """
    Registra el modelo en el MLflow Model Registry.

    :param run_id: ID del run de MLflow donde se logueó el modelo.
    """
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)