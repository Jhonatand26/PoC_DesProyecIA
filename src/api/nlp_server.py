"""
nlp_server.py — Servidor gRPC del Servicio NLP
PoC Asistente Fitosanitario con IA | UAO 2026

Responsabilidad unica: recibir RecommendationRequest via gRPC,
delegar en los modulos de NLP (prompt_builder + openai_client)
y retornar RecommendationResponse.

No contiene logica de IA. Solo traduce entre gRPC y los modulos NLP.
"""

import sys
import os
import logging
from concurrent import futures

import grpc

# ---------------------------------------------------------------------------
# Ajuste de path para importar desde src/api/protos y src/nlp
# Ejecutar desde la raiz del proyecto: uv run python src/api/nlp_server.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "protos"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nlp"))

import nlp_pb2
import nlp_pb2_grpc
from prompt_builder import build_prompt
from openai_client import get_recommendation

# ---------------------------------------------------------------------------
# Configuracion de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NLP-SERVER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Servicer — implementa el contrato definido en nlp.proto
# ---------------------------------------------------------------------------
class NLPServiceServicer(nlp_pb2_grpc.NLPServiceServicer):
    """
    Implementacion del servicio NLP definido en nlp.proto.

    Recibe clase y confianza desde Streamlit, construye el prompt
    agronomico y consulta a GPT-5 Nano para obtener recomendaciones.
    """

    def GetRecommendation(self, request, context):
        """
        Maneja una solicitud de recomendacion agronomica.

        Args:
            request (RecommendationRequest): Contiene class_name y confidence.
            context (grpc.ServicerContext): Contexto gRPC para manejo de errores.

        Returns:
            RecommendationResponse: Recomendacion en Markdown, success y error.
        """
        logger.info(
            "Solicitud recibida — clase: '%s' | confianza: %.2f",
            request.class_name,
            request.confidence,
        )

        # --- Validacion de entrada -------------------------------------------
        if not request.class_name or not request.class_name.strip():
            logger.warning("class_name vacio recibido.")
            return nlp_pb2.RecommendationResponse(
                recommendation="",
                success=False,
                error="class_name no puede estar vacio.",
            )

        if not (0.0 <= request.confidence <= 1.0):
            logger.warning("Confianza fuera de rango: %.2f", request.confidence)
            return nlp_pb2.RecommendationResponse(
                recommendation="",
                success=False,
                error=f"confidence debe estar entre 0.0 y 1.0, recibido: {request.confidence}",
            )

        # --- Pipeline: build_prompt → get_recommendation ---------------------
        try:
            prompt = build_prompt(request.class_name, request.confidence)
            logger.info("Prompt construido correctamente.")

            recommendation = get_recommendation(prompt)
            logger.info("Recomendacion recibida de GPT-5 Nano.")

            return nlp_pb2.RecommendationResponse(
                recommendation=recommendation,
                success=True,
                error="",
            )

        except EnvironmentError as e:
            # API key no configurada
            logger.error("Error de configuracion: %s", e)
            return nlp_pb2.RecommendationResponse(
                recommendation="",
                success=False,
                error=f"Error de configuracion: {e}",
            )

        except Exception as e:
            # Cualquier otro error (red, OpenAI, etc.)
            logger.error("Error inesperado: %s", e)
            return nlp_pb2.RecommendationResponse(
                recommendation="",
                success=False,
                error=f"Error interno del servidor NLP: {e}",
            )


# ---------------------------------------------------------------------------
# Funcion serve() — arranca y mantiene vivo el servidor
# ---------------------------------------------------------------------------
def serve() -> None:
    """
    Inicializa y arranca el servidor gRPC NLP.

    Lee el puerto desde la variable de entorno NLP_SERVICE_PORT.
    Por defecto usa el puerto 50052.
    """
    from dotenv import load_dotenv

    load_dotenv()

    port = os.getenv("NLP_SERVICE_PORT", "50052")
    address = f"[::]:{port}"

    # ThreadPoolExecutor: cuantos requests simultaneos puede manejar
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServiceServicer(), server)

    server.add_insecure_port(address)
    server.start()

    logger.info("Servidor NLP escuchando en %s", address)
    logger.info("Listo para recibir solicitudes. Presiona Ctrl+C para detener.")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Servidor NLP detenido.")
        server.stop(grace=5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    serve()
