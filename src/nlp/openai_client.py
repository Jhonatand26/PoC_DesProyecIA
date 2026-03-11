"""
Cliente para la API de OpenAI GPT-5 Nano con reintentos automaticos.
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
        EnvironmentError: Si OPENAI_API_KEY no esta definida.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY no esta configurada en el archivo .env"
        )
    return OpenAI(api_key=api_key)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
def get_recommendation(prompt: str) -> str:
    """
    Envia un prompt a GPT-5 Nano y retorna la recomendacion agronomica.

    Args:
        prompt (str): Prompt construido por build_prompt().

    Returns:
        str: Recomendacion agronomica generada por GPT-5 Nano.

    Raises:
        EnvironmentError: Si OPENAI_API_KEY no esta configurada.
        ValueError: Si el prompt esta vacio.
    """
    if not prompt or not prompt.strip():
        raise ValueError("El prompt no puede estar vacio.")

    client = _get_client()

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un agronomo experto en cultivos del Valle del Cauca, Colombia. "
                    "Siempre respondes en espanol con informacion practica para agricultores."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content.strip()
