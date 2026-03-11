"""
Script para registrar el modelo de clasificación de enfermedades de plantas
en MLflow Model Registry con las métricas del paper.

Ejecutar una vez para crear el experimento y registrar el modelo:
    uv run python scripts/register_model.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vision.mlflow_tracker import register_model  # noqa: E402

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
MODEL_VERSION = "v1.0"
MODEL_SOURCE = (
    "https://huggingface.co/"
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)

# Métricas reportadas en el paper del modelo
PAPER_ACCURACY = 0.9970
PAPER_F1_SCORE = 0.9932


def main():
    print("=" * 60)
    print("  Registrando modelo en MLflow Model Registry")
    print("=" * 60)
    print(f"  Modelo:   {MODEL_NAME}")
    print(f"  Versión:  {MODEL_VERSION}")
    print(f"  Fuente:   {MODEL_SOURCE}")
    print(f"  Accuracy: {PAPER_ACCURACY}")
    print(f"  F1-Score: {PAPER_F1_SCORE}")
    print("=" * 60)

    register_model(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_source=MODEL_SOURCE,
        accuracy=PAPER_ACCURACY,
        f1_score=PAPER_F1_SCORE,
    )


if __name__ == "__main__":
    main()
