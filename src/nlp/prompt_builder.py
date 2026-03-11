"""
Módulo para construir prompts agronómicos contextualizados al Valle del Cauca.
Recibe el resultado del módulo de Visión Computacional y genera un prompt
estructurado para la API de OpenAI GPT-5 Nano.
"""


def build_prompt(class_name: str, confidence: float) -> str:
    """
    Construye un prompt agronómico contextualizado para GPT-5 Nano.

    Args:
        class_name (str): Nombre de la clase detectada por el modelo CV.
                          Ejemplo: 'Tomato___Late_blight'
        confidence (float): Nivel de confianza de la clasificación (0.0 a 1.0).
                            Ejemplo: 0.954

    Returns:
        str: Prompt estructurado listo para enviarse a la API de OpenAI.

    Raises:
        ValueError: Si class_name está vacío o confidence está fuera de [0, 1].
    """
    if not class_name or not class_name.strip():
        raise ValueError("class_name no puede estar vacío.")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence debe estar entre 0.0 y 1.0.")

    clase_legible = class_name.replace("___", " - ").replace("_", " ")
    confianza_porcentaje = round(confidence * 100, 1)

    prompt = f"""Eres un agrónomo experto en cultivos del Valle del Cauca, Colombia.
Un agricultor de la región te consulta sobre una enfermedad detectada en su cultivo.

DIAGNÓSTICO DETECTADO:
- Enfermedad/Condición: {clase_legible}
- Confianza del diagnóstico: {confianza_porcentaje}%

Por favor proporciona una respuesta estructurada con las siguientes secciones:

1. DIAGNÓSTICO: Explica brevemente qué es esta enfermedad y cómo afecta al cultivo.
2. TRATAMIENTO: Indica los pasos concretos para tratar la enfermedad, incluyendo productos disponibles en Colombia si aplica.
3. PREVENCIÓN: Describe medidas preventivas adaptadas al clima del Valle del Cauca.

Responde en español, con lenguaje claro para agricultores sin formación técnica avanzada.
Sé práctico y concreto. Máximo 300 palabras."""

    return prompt
