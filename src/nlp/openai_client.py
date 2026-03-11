"""
Cliente para la API de OpenAI GPT-5 Nano.
Gestiona la conexión, el envío de prompts y el manejo de errores
con reintentos automáticos usando backoff exponencial.
"""

import os
from openai import OpenAI, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()


def _get_client() -> OpenAI:
    """
    Crea y retorna un cliente de OpenAI autenticado.

    Returns:
        OpenAI: Cliente configurado con la API key del entorno.

    Raises:
        EnvironmentError: Si OPENAI_API_KEY no está definida en el entorno.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La variable de entorno OPENAI_API_KEY no está configurada. "
            "Asegúrate de tener un archivo .env con tu API key."
        )
    return OpenAI(api_key=api_key)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
def get_recommendation(prompt: str) -> str:
    """
    Envía un prompt a GPT-5 Nano y retorna la recomendación agronómica.

    Reintenta automáticamente hasta 3 veces si hay errores de red
    o límite de tasa, con espera exponencial entre intentos.

    Args:
        prompt (str): Prompt agronómico construido por build_prompt().

    Returns:
        str: Recomendación agronómica generada por GPT-5 Nano.

    Raises:
        EnvironmentError: Si OPENAI_API_KEY no está configurada.
        ValueError: Si el prompt está vacío.
    """
    if not prompt or not prompt.strip():
        raise ValueError("El prompt no puede estar vacío.")

    client = _get_client()

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un agrónomo experto en cultivos del Valle del Cauca, Colombia. "
                    "Siempre respondes en español con información práctica para agricultores."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=500,
        temperature=0.3,
    )

    recomendacion = response.choices[0].message.content
    return recomendacion.strip()
