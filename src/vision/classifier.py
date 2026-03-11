"""
Módulo principal de clasificación de imágenes de enfermedades de plantas.
"""
import os
import time
from typing import Dict, Any
from .model_loader import load_model
from .preprocessor import preprocess_image
from .mlflow_tracker import log_inference

MODEL_NAME = os.getenv(
    "CV_MODEL_NAME",
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)
CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")


def classify_image(image_path: str) -> Dict[str, Any]:
    """
    Ejecuta la inferencia sobre una imagen y retorna los resultados.
    Registra parámetros, métricas y tiempo de inferencia en MLflow.

    :param image_path: Ruta de la imagen para clasificar.
    :return: Diccionario con la clase ganadora, la confianza y el top 3.
    """
    pipe = load_model(MODEL_NAME, CACHE_DIR)
    img = preprocess_image(image_path)

    start = time.time()
    results = pipe(img)
    inference_time_ms = (time.time() - start) * 1000

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    top_1 = sorted_results[0]
    top_3 = sorted_results[:3]

    log_inference(
        model_name=MODEL_NAME,
        cache_dir=CACHE_DIR,
        predicted_class=top_1["label"],
        confidence=float(top_1["score"]),
        inference_time_ms=round(inference_time_ms, 2)
    )

    return {
        "class": top_1["label"],
        "confidence": float(top_1["score"]),
        "top_3": [
            {"class": r["label"], "confidence": float(r["score"])}
            for r in top_3
        ]
    }
