"""
Módulo para la carga del modelo de clasificación de enfermedades de plantas.
"""
from transformers import pipeline, MobileNetV2ImageProcessor, AutoModelForImageClassification


def load_model(model_name: str, cache_dir: str):
    """
    Carga y retorna un pipeline de clasificación de HuggingFace.

    :param model_name: Nombre del modelo en HuggingFace Hub.
    :param cache_dir: Directorio local para guardar en caché.
    :return: Pipeline de transformers para clasificación de imágenes.
    """
    feature_extractor = MobileNetV2ImageProcessor.from_pretrained(

        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    pipe = pipeline(
        task="image-classification",
        model=model,
        feature_extractor=feature_extractor
    )
    return pipe