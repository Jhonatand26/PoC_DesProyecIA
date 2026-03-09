"""
Módulo principal de clasificación de imágenes de enfermedades de plantas.
"""
from typing import Dict, Any
from .model_loader import load_model
from .preprocessor import preprocess_image


def classify_image(image_path: str) -> Dict[str, Any]:
    """
    Ejecuta la inferencia sobre una imagen y retorna los resultados.

    :param image_path: Ruta de la imagen para clasificar.
    :return: Diccionario con la clase ganadora, la confianza y el top 3.
    """
    model_name = (
        "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    )
    cache_dir = "./hf_cache"

    pipe = load_model(model_name, cache_dir)
    img = preprocess_image(image_path)
    try:
        results = pipe(img)
    except Exception as e:
        print(f"Bypass de preprocesador activado para fallback por: "
              f"{e}.")
        results = pipe(image_path)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    top_1 = sorted_results[0]
    top_3 = sorted_results[:3]

    return {
        "class": top_1["label"],
        "confidence": float(top_1["score"]),
        "top_3": [{"class": r["label"], "confidence": float(r["score"])}
                  for r in top_3]
    }
