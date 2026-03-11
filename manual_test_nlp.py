"""
Script de validacion manual de prompts — Issue #17.
Prueba el flujo completo: build_prompt -> get_recommendation con la API real.
"""

import sys
import os

# Añadir el directorio raíz al PYTHONPATH para que src sea importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.nlp.prompt_builder import build_prompt
from src.nlp.openai_client import get_recommendation

casos = [
    ("Tomato___Late_blight", 0.954),
    ("Tomato___healthy", 0.99),
    ("Pepper,_bell___Bacterial_spot", 0.87),
]

for class_name, confidence in casos:
    print("=" * 60)
    print(f"Clase: {class_name} | Confianza: {confidence}")
    print("=" * 60)
    prompt = build_prompt(class_name, confidence)
    print("PROMPT ENVIADO:")
    print(prompt)
    print()
    print("RESPUESTA GPT-5 NANO:")
    respuesta = get_recommendation(prompt)
    print(respuesta)
    print()
