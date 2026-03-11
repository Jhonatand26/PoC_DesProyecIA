"""
Modulo para construir prompts agronomicos contextualizados al Valle del Cauca.
"""


def build_prompt(class_name: str, confidence: float) -> str:
    """
    Construye un prompt agronomico contextualizado para GPT-5 Nano.

    Args:
        class_name (str): Clase detectada por el modelo CV.
        confidence (float): Confianza de la clasificacion
                            (0.0 a 1.0).

    Returns:
        str: Prompt estructurado listo para enviarse a OpenAI.

    Raises:
        ValueError: Si class_name esta vacio o confidence fuera de [0, 1].
    """
    if not class_name or not class_name.strip():
        raise ValueError("class_name no puede estar vacio.")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            "confidence debe estar entre 0.0 y 1.0."
        )

    clase_legible = class_name.replace("___", " - ").replace("_", " ")
    confianza_porcentaje = round(confidence * 100, 1)

    prompt = (
        "Eres un agronomo experto en cultivos "
        "del Valle del Cauca, Colombia.\n"
        "Un agricultor consulta sobre una enfermedad "
        f"detectada en su cultivo.\n\n"
        f"DIAGNOSTICO DETECTADO:\n"
        f"- Enfermedad/Condicion: {clase_legible}\n"
        f"- Confianza del diagnostico: {confianza_porcentaje}%\n\n"
        "Proporciona una respuesta estructurada "
        "con estas secciones:\n\n"
        "1. DIAGNOSTICO: Que es esta enfermedad "
        "y como afecta al cultivo.\n"
        "2. TRATAMIENTO: Pasos concretos incluyendo "
        "productos en Colombia.\n"
        "3. PREVENCION: Medidas adaptadas al clima "
        "del Valle del Cauca.\n\n"
        "Responde en espanol, lenguaje claro "
        "para agricultores. Maximo 300 palabras."
    )

    return prompt
