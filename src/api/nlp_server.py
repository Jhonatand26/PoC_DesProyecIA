"""
nlp_server.py — Servidor gRPC del Servicio NLP.

Expone get_recommendation() via gRPC en el puerto configurado
en NLP_SERVICE_PORT. Delega la generacion de recomendaciones
agronomicas a src/nlp/openai_client.py.
"""

import logging
import os
import sys
from concurrent import futures

import grpc
from dotenv import load_dotenv

# Agrega la raiz del proyecto al path para imports relativos
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.api import nlp_pb2, nlp_pb2_grpc
from src.nlp.openai_client import get_recommendation
from src.nlp.prompt_builder import build_prompt

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
NLP_SERVICE_PORT = os.getenv("NLP_SERVICE_PORT", "50052")
MAX_WORKERS = int(os.getenv("NLP_MAX_WORKERS", "4"))


# ------------------------------------------------------------------
# Implementacion del Servicio
# ------------------------------------------------------------------
class NLPServicer(nlp_pb2_grpc.NLPServiceServicer):
    """Implementa el contrato definido en nlp.proto."""

    def GetRecommendation(self, request, context):
        """
        Recibe clase y confianza del Servicio CV, construye el prompt
        y retorna la recomendacion agronomica generada por GPT-5 Nano.

        Args:
            request: RecommendationRequest con class_name y confidence.
            context: Contexto gRPC.

        Returns:
            RecommendationResponse con recomendacion en Markdown.
        """
        logger.info(
            "Request recibido: clase=%s confianza=%.2f",
            request.class_name,
            request.confidence,
        )

        try:
            # Construye el prompt agronomico — responsabilidad de Mateo
            prompt = build_prompt(request.class_name, request.confidence)

            # Llama a la API de OpenAI — responsabilidad de Mateo
            recommendation = get_recommendation(prompt)

            logger.info(
                "Recomendacion generada exitosamente para %s", request.class_name
            )

            return nlp_pb2.RecommendationResponse(
                recommendation=recommendation, success=True
            )

        except Exception as e:
            logger.error("Error generando recomendacion: %s", str(e))
            return nlp_pb2.RecommendationResponse(success=False, error=str(e))


# ------------------------------------------------------------------
# Arranque del servidor
# ------------------------------------------------------------------
def serve():
    """Inicia el servidor gRPC y lo mantiene activo."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServicer(), server)

    address = f"[::]:{NLP_SERVICE_PORT}"
    server.add_insecure_port(address)
    server.start()

    logger.info("Servidor NLP escuchando en puerto %s", NLP_SERVICE_PORT)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Servidor NLP detenido.")
        server.stop(0)


if __name__ == "__main__":
    serve()
