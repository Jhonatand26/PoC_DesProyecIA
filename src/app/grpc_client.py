"""
Clientes gRPC para los servicios de Vision Computacional y NLP.

Este modulo encapsula TODA la comunicacion con los servicios gRPC.
Es el unico archivo de src/app/ que conoce los detalles de gRPC,
protobuf y los contratos .proto. Ningun otro modulo necesita saber
como se comunican los servicios internamente.

Principio de diseno:
    - Bajo acople: app.py y utils.py NO importan nada de gRPC.
      Solo este modulo conoce grpc, protobuf y los stubs generados.
    - Interfaz estable: las funciones exponen datos como diccionarios
      y strings de Python. Si cambia el contrato .proto, solo se
      modifica este archivo.
    - Stubs simulados: en modo development (config.USE_STUBS = True),
      las funciones retornan datos de prueba sin necesidad de que
      los servidores esten corriendo.

Ejemplo de uso:
    from grpc_client import classify_image, get_recommendation
    result = classify_image(image_bytes)
    recommendation = get_recommendation(result["class_name"], result["confidence"])

Autor: Nicolas Vasquez Renjifo
Modulo: Frontend & QA
Issues: #22 grpc-client-streamlit
Repositorio: github.com/Jhonatand26/PoC_DesProyecIA
"""

from config import (
    CV_SERVICE_HOST,  # Host del servicio CV (ej: "localhost")
    CV_SERVICE_PORT,  # Puerto del servicio CV (ej: "50051")
    NLP_SERVICE_HOST,  # Host del servicio NLP (ej: "localhost")
    NLP_SERVICE_PORT,  # Puerto del servicio NLP (ej: "50052")
    USE_STUBS,  # True = usar datos simulados, False = gRPC real
)

# ---------------------------------------------------------------------------
# >>> INTEGRACION JHONATAN (Issue #10, #11, #16, #17)
# Cuando los archivos .proto esten compilados con protoc y los servidores
# gRPC esten listos, descomentar estas importaciones:
#
import grpc
from src.api.protos import vision_pb2
from src.api.protos import vision_pb2_grpc
from src.api.protos import nlp_pb2
from src.api.protos import nlp_pb2_grpc

# ==========================================================================
# FUNCIONES INTERNAS DE CONEXION
# ==========================================================================


def _create_cv_channel():
    """
    Crea un canal gRPC hacia el Servicio de Vision Computacional.

    Construye la direccion del servicio usando CV_SERVICE_HOST y
    CV_SERVICE_PORT. Retorna un canal gRPC inseguro (sin TLS),
    suficiente para comunicacion interna entre contenedores Docker.

    Returns:
        grpc.Channel: Canal conectado al Servicio CV.

    Raises:
        grpc.RpcError: Si no se puede establecer la conexion.

    Note:
        >>> INTEGRACION JHONATAN (Issue #11 grpc-server-cv)
        El host y puerto deben coincidir con la configuracion del
        servidor CV en docker-compose.yml.
    """
    # ---------------------------------------------------------------------------
    # >>> INTEGRACION JHONATAN — Descomentar cuando el servidor CV este listo:
    #
    target = f"{CV_SERVICE_HOST}:{CV_SERVICE_PORT}"
    channel = grpc.insecure_channel(target)
    return channel
    # ---------------------------------------------------------------------------
    pass  # STUB — no hay canal real en modo development


def _create_nlp_channel():
    """
    Crea un canal gRPC hacia el Servicio de NLP (Gemini Flash).

    Construye la direccion del servicio usando NLP_SERVICE_HOST y
    NLP_SERVICE_PORT. Retorna un canal gRPC inseguro.

    Returns:
        grpc.Channel: Canal conectado al Servicio NLP.

    Raises:
        grpc.RpcError: Si no se puede establecer la conexion.

    Note:
        >>> INTEGRACION JHONATAN (Issue #17 grpc-server-nlp)
        El host y puerto deben coincidir con la configuracion del
        servidor NLP en docker-compose.yml.
    """
    # ---------------------------------------------------------------------------
    # >>> INTEGRACION JHONATAN — Descomentar cuando el servidor NLP este listo:
    #
    target = f"{NLP_SERVICE_HOST}:{NLP_SERVICE_PORT}"
    channel = grpc.insecure_channel(target)
    return channel
    # ---------------------------------------------------------------------------
    pass  # STUB — no hay canal real en modo development


# ==========================================================================
# STUBS SIMULADOS (MODO DEVELOPMENT)
# ==========================================================================


def _classify_image_stub(image_bytes):
    """
    Stub simulado del servicio de Vision Computacional.

    Retorna datos de prueba que imitan la respuesta real del modelo
    MobileNetV2 entrenado con PlantVillage. Permite desarrollar y
    probar la interfaz sin necesidad de que el servidor CV este corriendo.

    Args:
        image_bytes (bytes): Imagen como bytes (no se procesa en el stub).

    Returns:
        dict: Respuesta simulada con clase, confianza y top-3.

    Note:
        Esta funcion se elimina cuando USE_STUBS pase a False
        (APP_ENV = 'production'). No forma parte del flujo real.
    """
    return {
        "class_name": "Tomato___Late_blight",
        "confidence": 0.87,
        "top_3": [
            {"class_name": "Tomato___Late_blight", "confidence": 0.87},
            {"class_name": "Tomato___Early_blight", "confidence": 0.08},
            {"class_name": "Tomato___Leaf_Mold", "confidence": 0.03},
        ],
    }


def _get_recommendation_stub(class_name, confidence):
    """
    Stub simulado del servicio de NLP (Gemini Flash).

    Retorna una recomendacion agronomica de ejemplo que imita la
    estructura real de la respuesta de Gemini Flash. Permite
    desarrollar y probar la interfaz sin la API de Gemini.

    Args:
        class_name (str): Nombre de la enfermedad detectada.
        confidence (float): Confianza de la prediccion.

    Returns:
        str: Recomendacion simulada en formato markdown.

    Note:
        >>> INTEGRACION MATEO (Issue #15 prompt_builder.py)
        El formato de esta respuesta simulada debe parecerse al
        formato real que Mateo defina en prompt_builder.py.
        Estructura esperada: Diagnostico + Tratamiento + Prevencion.
    """
    # Formatear el nombre de clase para que sea legible
    formatted_name = class_name.replace("___", " - ").replace("_", " ")

    return (
        "## Diagnostico\n"
        f"**Enfermedad detectada:** {formatted_name}\n"
        f"**Confianza del modelo:** {confidence:.0%}\n\n"
        "## Tratamiento recomendado\n"
        "- Aplicar fungicida a base de cobre (oxicloruro de cobre) "
        "en dosis de 2-3 g/L.\n"
        "- Retirar y destruir las hojas afectadas para reducir la "
        "fuente de inoculo.\n"
        "- Mejorar la ventilacion entre plantas mediante podas de "
        "aclareo.\n\n"
        "## Prevencion\n"
        "- Utilizar variedades resistentes adaptadas al clima del "
        "Valle del Cauca.\n"
        "- Evitar riego por aspersion en horas de la tarde.\n"
        "- Rotar cultivos cada temporada para romper el ciclo del "
        "patogeno.\n\n"
        "*Recomendacion generada para condiciones del Valle del Cauca, "
        "Colombia.*"
    )


# ==========================================================================
# FUNCIONES PUBLICAS — Las unicas que app.py debe usar
# ==========================================================================


def classify_image(image_bytes):
    """
    Envia una imagen al Servicio CV y retorna el diagnostico.

    Esta es la funcion publica que app.py llama. Internamente decide
    si usar el stub simulado (development) o la llamada gRPC real
    (production) basandose en config.USE_STUBS.

    Args:
        image_bytes (bytes): Imagen en formato JPG/PNG como bytes crudos.

    Returns:
        dict: Diccionario con las claves:
            - 'class_name' (str): Nombre de la enfermedad o planta sana.
            - 'confidence' (float): Confianza de la prediccion (0.0-1.0).
            - 'top_3' (list[dict]): Top 3 predicciones, cada una con
              'class_name' (str) y 'confidence' (float).
        Retorna None si hay error en la comunicacion.

    Note:
        >>> INTEGRACION JORGE LUIS + JHONATAN
        La estructura del dict retornado es el contrato entre este
        modulo y app.py. Si vision.proto cambia los nombres de campos,
        la conversion se hace AQUI para que app.py no se entere.
    """
    # En modo development, usar datos simulados
    if USE_STUBS:
        return _classify_image_stub(image_bytes)

    # ---------------------------------------------------------------------------
    # >>> INTEGRACION JHONATAN + JORGE LUIS — Llamada gRPC real
    # Descomentar cuando vision.proto este compilado y el servidor funcione:
    #
    channel = _create_cv_channel()
    #
    try:
        # Crear stub del cliente gRPC
        stub = vision_pb2_grpc.VisionServiceStub(channel)

        # Construir request con los bytes de la imagen
        request = vision_pb2.ImageRequest(image_data=image_bytes)

        # Enviar imagen y esperar respuesta (timeout 30 segundos)
        response = stub.ClassifyImage(request, timeout=30)

        # Convertir response protobuf a diccionario Python
        # >>> PUNTO CLAVE DE BAJO ACOPLE: la conversion de protobuf
        # a dict ocurre AQUI. app.py solo recibe un dict estandar.
        top_3 = [
            {"class_name": pred.label, "confidence": pred.confidence}
            for pred in response.top_3
        ]

        return {
            "class_name": response.class_name,
            "confidence": response.confidence,
            "top_3": top_3,
        }

    except grpc.RpcError as e:
        # Log del error para debugging
        print(f"[ERROR] Servicio CV: {e.code()} - {e.details()}")
        return None

    finally:
        channel.close()
    # ---------------------------------------------------------------------------

    return None  # Fallback si no hay stubs ni gRPC real


def get_recommendation(class_name, confidence):
    """
    Envia el diagnostico al Servicio NLP y retorna recomendaciones.

    Esta es la funcion publica que app.py llama. Internamente decide
    si usar el stub simulado o la llamada gRPC real.

    Args:
        class_name (str): Nombre de la enfermedad detectada por el
            modelo CV. Ejemplo: "Tomato___Late_blight"
        confidence (float): Confianza de la prediccion (0.0-1.0).

    Returns:
        str: Recomendacion agronomica en formato markdown con secciones
            de diagnostico, tratamiento y prevencion.
        Retorna None si hay error en la comunicacion.

    Note:
        >>> INTEGRACION MATEO + JHONATAN
        La respuesta string es el contrato entre este modulo y app.py.
        Si nlp.proto cambia la estructura, la conversion se hace AQUI.
    """
    # En modo development, usar recomendacion simulada
    if USE_STUBS:
        return _get_recommendation_stub(class_name, confidence)

    # ---------------------------------------------------------------------------
    # >>> INTEGRACION JHONATAN + MATEO — Llamada gRPC real
    # Descomentar cuando nlp.proto este compilado y el servidor funcione:
    #
    channel = _create_nlp_channel()

    try:
        # Crear stub del cliente gRPC
        stub = nlp_pb2_grpc.NLPServiceStub(channel)

        # Construir request con clase y confianza
        request = nlp_pb2.RecommendationRequest(
            class_name=class_name, confidence=confidence
        )

        # Enviar diagnostico y esperar recomendacion (timeout 60s)
        # Gemini Flash puede tardar mas que el modelo CV
        response = stub.GetRecommendation(request, timeout=60)

        # Retornar el texto plano de la recomendacion
        # >>> PUNTO CLAVE DE BAJO ACOPLE: si Mateo cambia el formato
        # del texto en prompt_builder.py, app.py no se entera
        # porque solo recibe un string.
        return response.recommendation

    except grpc.RpcError as e:
        print(f"[ERROR] Servicio NLP: {e.code()} - {e.details()}")
        return None

    finally:
        channel.close()
    # ---------------------------------------------------------------------------

    return None  # Fallback si no hay stubs ni gRPC real
