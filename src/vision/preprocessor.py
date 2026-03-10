"""
Módulo para el preprocesamiento de imágenes.
"""
from PIL import Image


def preprocess_image(image_path: str) -> Image.Image:
    """
    Abre, convierte a RGB y redimensiona la imagen a 224x224.

    :param image_path: Ruta de la imagen de entrada.
    :return: Imagen PIL preprocesada, lista para el modelo.
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    return img
