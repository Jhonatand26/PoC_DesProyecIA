"""
Cliente para la API de OpenAI GPT-5 Nano con reintentos automaticos.
"""

import os
import time
from openai import OpenAI, RateLimitError, APIConnectionError
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)
from dotenv import load_dotenv
from src.nlp.mlflow_tracker import log_recommendation, setup_experiment

load_dotenv()
setup_experiment()


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
def get_recommendation(
    prompt: str, class_name: str = "", confidence: float = 0.0
) -> str:
    """
    Envia un prompt a GPT-5 Nano y retorna la recomendacion agronomica.

    Registra el experimento en MLflow con parametros y metricas.

    Args:
        prompt (str): Prompt construido por build_prompt().
        class_name (str): Clase detectada por el modelo CV.
        confidence (float): Confianza del diagnostico CV.

    Returns:
        str: Recomendacion agronomica generada por GPT-5 Nano.

    Raises:
        EnvironmentError: Si OPENAI_API_KEY no esta configurada.
        ValueError: Si el prompt esta vacio.
    """
    if not prompt or not prompt.strip():
        raise ValueError("El prompt no puede estar vacio.")

    client = _get_client()

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un agronomo experto en cultivos "
                        "del Valle del Cauca, Colombia. "
                        "Siempre respondes en espanol con "
                        "informacion practica para agricultores."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        latency_ms = (time.time() - start_time) * 1000
        recommendation = response.choices[0].message.content.strip()

        log_recommendation(
            class_name=class_name,
            confidence=confidence,
            prompt=prompt,
            recommendation=recommendation,
            success=True,
            latency_ms=latency_ms,
        )

        return recommendation

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        log_recommendation(
            class_name=class_name,
            confidence=confidence,
            prompt=prompt,
            recommendation="",
            success=False,
            error=str(e),
            latency_ms=latency_ms,
        )
        raise
