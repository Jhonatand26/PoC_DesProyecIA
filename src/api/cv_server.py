"""
cv_server.py — Servidor gRPC del Servicio de Vision Computacional
PoC Asistente Fitosanitario con IA | UAO 2026

Responsabilidad unica: recibir ImageRequest via gRPC,
delegar en src/vision/classifier.py y retornar ClassificationResponse.

NOTA PARA EL EQUIPO:
    Cuando Jorge Luis entregue src/vision/classifier.py con la funcion
    classify_image(image_path) -> dict, reemplazar _stub_classify_image
    por el import real. El servidor NO necesita cambios en ninguna otra parte.

    Contrato esperado de classify_image():
        Input:  image_path (str) — ruta a un archivo de imagen temporal
        Output: dict con las siguientes claves:
            {
                "class_name":  str,         # clase con mayor confianza
                "confidence":  float,       # entre 0.0 y 1.0
                "top_3": [                  # lista de 3 predicciones
                    {"label": str, "confidence": float},
                    ...
                ]
            }
"""

import time
import sys
import os
import logging
import tempfile
from concurrent import futures

import grpc

# ---------------------------------------------------------------------------
# Ajuste de path para importar desde src/api/protos
# Ejecutar desde la raiz del proyecto: uv run python src/api/cv_server.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "protos"))

from src.vision.mlflow_tracker import log_inference
import vision_pb2
import vision_pb2_grpc

# ---------------------------------------------------------------------------
# Import del clasificador — con fallback al stub temporal
# ---------------------------------------------------------------------------
try:
    from src.vision.classifier import classify_image

    _USING_STUB = False
    logging.getLogger(__name__).info("classifier.py cargado correctamente.")
except ImportError as e:
    _USING_STUB = True
    logging.getLogger(__name__).warning(
        "classifier.py no encontrado — usando stub temporal. "
        "El servidor funciona pero retorna datos ficticios. Error: %s",
        e,
    )

# ---------------------------------------------------------------------------
# Configuracion de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CV-SERVER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub temporal — reemplaza classify_image hasta que Jorge Luis lo entregue
# ---------------------------------------------------------------------------
def _stub_classify_image(image_path: str) -> dict:
    """
    Stub temporal de classify_image para desarrollo sin el modulo CV.

    Retorna datos ficticios pero con la estructura correcta del contrato.
    ELIMINAR cuando classifier.py este disponible.

    Args:
        image_path (str): Ruta de imagen (ignorada en el stub).

    Returns:
        dict: Respuesta simulada con estructura real.
    """
    logger.warning("STUB activo — respuesta simulada para '%s'", image_path)
    return {
        "class_name": "Tomato___Late_blight",
        "confidence": 0.91,
        "top_3": [
            {"label": "Tomato___Late_blight", "confidence": 0.91},
            {"label": "Tomato___Early_blight", "confidence": 0.06},
            {"label": "Tomato___healthy", "confidence": 0.03},
        ],
    }


# ---------------------------------------------------------------------------
# Seleccion de funcion activa segun disponibilidad del modulo
# ---------------------------------------------------------------------------
_classify_fn = _stub_classify_image if _USING_STUB else classify_image


# ---------------------------------------------------------------------------
# Servicer — implementa el contrato definido en vision.proto
# ---------------------------------------------------------------------------
class VisionServiceServicer(vision_pb2_grpc.VisionServiceServicer):
    """
    Implementacion del servicio CV definido en vision.proto.

    Recibe bytes de imagen desde Streamlit, los guarda en un archivo
    temporal, delega la clasificacion a classify_image() y retorna
    el resultado como ClassificationResponse.
    """

    def ClassifyImage(self, request, context):
        """
        Maneja una solicitud de clasificacion de imagen.

        Args:
            request (ImageRequest): Contiene image_data (bytes) y filename.
            context (grpc.ServicerContext): Contexto gRPC para manejo de errores.

        Returns:
            ClassificationResponse: Clase, confianza, top-3, success y error.
        """
        logger.info("Solicitud recibida — archivo: '%s'", request.filename)

        # --- Validacion de entrada -------------------------------------------
        if not request.image_data:
            logger.warning("image_data vacio recibido.")
            return vision_pb2.ClassificationResponse(
                class_name="",
                confidence=0.0,
                top_3=[],
                success=False,
                error="image_data no puede estar vacio.",
            )

        # --- Guardar imagen en archivo temporal ------------------------------
        # gRPC transmite bytes; classifier.py espera una ruta de archivo.
        # Usamos tempfile para crear un archivo seguro y limpiarlo al final.
        suffix = self._get_suffix(request.filename)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(request.image_data)
                tmp_path = tmp_file.name

            logger.info("Imagen guardada en temporal: %s", tmp_path)
            # Medir latencia real de la inferencia
            t_start = time.time()
            result = _classify_fn(tmp_path)
            inference_time_ms = (time.time() - t_start) * 1000

            logger.info(
                "Clasificacion exitosa — clase: '%s' | confianza: %.2f",
                result["class_name"],
                result["confidence"],
            )

            # Registrar en MLflow — el servidor es dueño de esta métrica
            log_inference(
                model_name="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
                cache_dir=os.getenv("HF_HOME", "./models"),
                predicted_class=result["class_name"],
                confidence=result["confidence"],
                inference_time_ms=inference_time_ms,
                model_version="v1.0",
                model_source="https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
            )

            # --- Clasificacion ----------------------------------------------
            result = _classify_fn(tmp_path)
            logger.info(
                "Clasificacion exitosa — clase: '%s' | confianza: %.2f",
                result["class_name"],
                result["confidence"],
            )

            # --- Construir top_3 para el response ---------------------------
            top_3_predictions = [
                vision_pb2.Prediction(
                    label=pred["label"],
                    confidence=pred["confidence"],
                )
                for pred in result.get("top_3", [])
            ]

            return vision_pb2.ClassificationResponse(
                class_name=result["class_name"],
                confidence=result["confidence"],
                top_3=top_3_predictions,
                success=True,
                error="",
            )

        except Exception as e:
            logger.error("Error en clasificacion: %s", e)
            # Registrar el fallo también — trazabilidad completa
            log_inference(
                model_name="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
                cache_dir=os.getenv("HF_HOME", "./models"),
                predicted_class="ERROR",
                confidence=0.0,
                inference_time_ms=0.0,
                model_version="v1.0",
            )
            return vision_pb2.ClassificationResponse(
                class_name="",
                confidence=0.0,
                top_3=[],
                success=False,
                error=f"Error interno del servidor CV: {e}",
            )

        finally:
            # --- Limpieza del archivo temporal ------------------------------
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info("Archivo temporal eliminado: %s", tmp_path)

    @staticmethod
    def _get_suffix(filename: str) -> str:
        """
        Extrae la extension del nombre de archivo para el archivo temporal.

        Args:
            filename (str): Nombre del archivo original (ej: hoja.jpg).

        Returns:
            str: Extension con punto (ej: .jpg) o '.jpg' por defecto.
        """
        if filename and "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ".jpg"


# ---------------------------------------------------------------------------
# Funcion serve() — arranca y mantiene vivo el servidor
# ---------------------------------------------------------------------------
def serve() -> None:
    """
    Inicializa y arranca el servidor gRPC CV.

    Lee el puerto desde la variable de entorno CV_SERVICE_PORT.
    Por defecto usa el puerto 50051.
    """
    from dotenv import load_dotenv

    load_dotenv()

    port = os.getenv("CV_SERVICE_PORT", "50051")
    address = f"[::]:{port}"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    vision_pb2_grpc.add_VisionServiceServicer_to_server(VisionServiceServicer(), server)

    server.add_insecure_port(address)
    server.start()

    logger.info("Servidor CV escuchando en %s", address)
    if _USING_STUB:
        logger.warning(
            "ATENCION: Corriendo con stub temporal. Esperando classifier.py de Jorge Luis."
        )
    logger.info("Listo para recibir solicitudes. Presiona Ctrl+C para detener.")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Servidor CV detenido.")
        server.stop(grace=5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    serve()
