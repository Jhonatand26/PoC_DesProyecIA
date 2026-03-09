"""
Módulo para la carga del modelo de clasificación de enfermedades de plantas.
"""
from transformers import pipeline


def load_model(model_name: str, cache_dir: str):
    """
    Carga y retorna un pipeline de clasificación de HuggingFace.

    :param model_name: Nombre del modelo en HuggingFace Hub.
    :param cache_dir: Directorio local para guardar en caché.
    :return: Pipeline de transformers para clasificación de imágenes.
    """
    try:
        pipe = pipeline(
            task="image-classification",
            model=model_name,
            model_kwargs={"cache_dir": cache_dir},
            trust_remote_code=True
        )
    except ValueError:
        print(f"Advertencia: Problema de arquitectura nativa en "
              f"{model_name}. Usando fallback.")
        pipe = pipeline(
            task="image-classification",
            model="google/vit-base-patch16-224",
            model_kwargs={"cache_dir": cache_dir}
        )
    return pipe
