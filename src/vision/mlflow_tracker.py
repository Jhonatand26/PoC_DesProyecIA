"""
Módulo para el registro de experimentos y modelos en MLflow.
"""
import os
import mlflow
import mlflow.pyfunc


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "plant-disease-cv"
REGISTERED_MODEL_NAME = "mobilenet-plant-disease"


def _init_mlflow() -> None:
    """Configura el tracking URI y el experimento activo."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_inference(
    model_name: str,
    cache_dir: str,
    predicted_class: str,
    confidence: float,
    inference_time_ms: float,
    model_version: str = "v1.0",
    model_source: str = "",
) -> None:
    """
    Registra parámetros, métricas y resultado de una inferencia en MLflow.

    :param model_name: Nombre del modelo usado en HuggingFace.
    :param cache_dir: Directorio de caché del modelo.
    :param predicted_class: Clase predicha por el modelo.
    :param confidence: Confianza de la predicción (0.0 - 1.0).
    :param inference_time_ms: Tiempo de inferencia en milisegundos.
    :param model_version: Versión del modelo.
    :param model_source: URL de origen del modelo en HuggingFace.
    """
    try:
        _init_mlflow()

        with mlflow.start_run():
            mlflow.log_params({
                "model_name": model_name,
                "model_version": model_version,
                "model_source": model_source,
                "cache_dir": cache_dir,
            })
            mlflow.log_metrics({
                "confidence": confidence,
                "inference_time_ms": inference_time_ms,
            })
            mlflow.set_tag("predicted_class", predicted_class)
    except Exception as e:
        print(f"Warning: No se pudo conectar a MLflow ({e})")


def log_model_metrics(
    accuracy: float,
    f1_score: float,
    model_name: str = "",
    model_version: str = "v1.0",
) -> None:
    """
    Registra las métricas del paper (accuracy, F1) en MLflow.

    :param accuracy: Accuracy reportada en el paper.
    :param f1_score: F1-score reportado en el paper.
    :param model_name: Nombre del modelo.
    :param model_version: Versión del modelo.
    """
    try:
        _init_mlflow()

        with mlflow.start_run():
            mlflow.log_params({
                "model_name": model_name,
                "model_version": model_version,
                "metrics_source": "paper",
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_score": f1_score,
            })
    except Exception as e:
        print(f"Warning: No se pudo conectar a MLflow ({e})")


def register_model(
    model_name: str,
    model_version: str = "v1.0",
    model_source: str = "",
    accuracy: float = 0.0,
    f1_score: float = 0.0,
) -> None:
    """
    Registra el modelo en el MLflow Model Registry junto con sus métricas.

    :param model_name: Nombre del modelo en HuggingFace.
    :param model_version: Versión del modelo.
    :param model_source: URL de origen del modelo.
    :param accuracy: Accuracy reportada.
    :param f1_score: F1-score reportado.
    """
    try:
        _init_mlflow()

        with mlflow.start_run() as run:
            mlflow.log_params({
                "model_name": model_name,
                "model_version": model_version,
                "model_source": model_source,
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_score": f1_score,
            })
            mlflow.set_tag("model_type", "huggingface")
            mlflow.set_tag("registered_model_name", REGISTERED_MODEL_NAME)
            print(f"Modelo logueado exitosamente (run_id={run.info.run_id})")
    except Exception as e:
        print(f"Error al registrar el modelo en MLflow: {e}")
