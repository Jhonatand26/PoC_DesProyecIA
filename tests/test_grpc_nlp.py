"""
Pruebas de contrato gRPC para el Servicio NLP y CV.
Issue #22 — validacion de contratos .proto sin servidor externo.

Estrategia:
    Las pruebas de contrato verifican que los mensajes Protobuf tienen
    exactamente los campos definidos en los archivos .proto, con los
    tipos correctos. No requieren servidores corriendo.

    Para pruebas de integracion end-to-end (servidor real), ver:
    docs/testing.md — seccion "Pruebas de integracion manual".

Contratos verificados:
    vision.proto  → ImageRequest, ClassificationResponse, Prediction
    nlp.proto     → RecommendationRequest, RecommendationResponse
"""

import pytest
from unittest.mock import MagicMock, patch
from concurrent import futures
import grpc

from src.api.protos import nlp_pb2, nlp_pb2_grpc
from src.api.protos import vision_pb2, vision_pb2_grpc


# ==========================================================================
# CLASE 1 — Contrato vision.proto
# ==========================================================================


class TestVisionProtoContrato:
    """
    Verifica que los mensajes de vision.proto tienen la estructura
    correcta definida en el contrato gRPC del servicio CV.
    """

    def test_image_request_acepta_image_data_bytes(self):
        """
        ImageRequest debe aceptar image_data como bytes.
        Es el campo principal del contrato de entrada del servidor CV.
        """
        datos = b"\xff\xd8\xff\xe0"  # Header JPEG real
        request = vision_pb2.ImageRequest(image_data=datos)
        assert request.image_data == datos

    def test_image_request_acepta_filename(self):
        """
        ImageRequest debe tener campo filename para identificar el archivo.
        """
        request = vision_pb2.ImageRequest(image_data=b"bytes", filename="hoja.jpg")
        assert request.filename == "hoja.jpg"

    def test_image_request_campos_por_defecto(self):
        """
        Un ImageRequest vacio debe tener valores por defecto validos,
        no lanzar excepcion al instanciarse.
        """
        request = vision_pb2.ImageRequest()
        assert request.image_data == b""
        assert request.filename == ""

    def test_classification_response_tiene_class_name(self):
        """ClassificationResponse debe tener campo class_name (str)."""
        response = vision_pb2.ClassificationResponse(
            class_name="Tomato___Late_blight",
            confidence=0.91,
            success=True,
            error="",
        )
        assert response.class_name == "Tomato___Late_blight"

    def test_classification_response_tiene_confidence_float(self):
        """
        confidence debe ser float entre 0.0 y 1.0.
        Es la metrica principal de calidad del diagnostico.
        """
        response = vision_pb2.ClassificationResponse(
            class_name="clase", confidence=0.87, success=True
        )
        assert isinstance(response.confidence, float)
        assert 0.0 <= response.confidence <= 1.0

    def test_classification_response_tiene_success_bool(self):
        """success debe ser bool — indica si la inferencia fue exitosa."""
        response_ok = vision_pb2.ClassificationResponse(
            class_name="clase", confidence=0.9, success=True
        )
        response_err = vision_pb2.ClassificationResponse(
            class_name="", confidence=0.0, success=False, error="error de prueba"
        )
        assert response_ok.success is True
        assert response_err.success is False

    def test_classification_response_tiene_campo_error(self):
        """
        error debe estar presente en el contrato.
        Permite al cliente mostrar mensajes descriptivos al usuario.
        """
        response = vision_pb2.ClassificationResponse(
            class_name="", confidence=0.0, success=False, error="imagen corrupta"
        )
        assert response.error == "imagen corrupta"

    def test_prediction_tiene_label_y_confidence(self):
        """
        Prediction (elemento de top_3) debe tener label y confidence.
        Es el contrato entre cv_server.py y grpc_client.py para el top_3.
        """
        pred = vision_pb2.Prediction(label="Tomato___Late_blight", confidence=0.91)
        assert pred.label == "Tomato___Late_blight"
        assert pred.confidence == pytest.approx(0.91)

    def test_classification_response_acepta_top_3(self):
        """
        ClassificationResponse debe poder contener una lista de Predictions.
        El cliente usa top_3 para mostrar las 3 clases mas probables.
        """
        predictions = [
            vision_pb2.Prediction(label="Tomato___Late_blight", confidence=0.91),
            vision_pb2.Prediction(label="Tomato___Early_blight", confidence=0.06),
            vision_pb2.Prediction(label="Tomato___healthy", confidence=0.03),
        ]
        response = vision_pb2.ClassificationResponse(
            class_name="Tomato___Late_blight",
            confidence=0.91,
            top_3=predictions,
            success=True,
        )
        assert len(response.top_3) == 3
        assert response.top_3[0].label == "Tomato___Late_blight"
        assert response.top_3[1].label == "Tomato___Early_blight"


# ==========================================================================
# CLASE 2 — Contrato nlp.proto
# ==========================================================================


class TestNLPProtoContrato:
    """
    Verifica que los mensajes de nlp.proto tienen la estructura
    correcta definida en el contrato gRPC del servicio NLP.
    """

    def test_recommendation_request_tiene_class_name(self):
        """
        RecommendationRequest debe tener class_name.
        Es la clase detectada por CV que se pasa al servicio NLP.
        """
        request = nlp_pb2.RecommendationRequest(
            class_name="Tomato___Late_blight",
            confidence=0.954,
        )
        assert request.class_name == "Tomato___Late_blight"

    def test_recommendation_request_tiene_confidence_float(self):
        """confidence debe ser float en RecommendationRequest."""
        request = nlp_pb2.RecommendationRequest(class_name="clase", confidence=0.75)
        assert isinstance(request.confidence, float)
        assert request.confidence == pytest.approx(0.75)

    def test_recommendation_request_campos_por_defecto(self):
        """Un RecommendationRequest vacio debe instanciarse sin errores."""
        request = nlp_pb2.RecommendationRequest()
        assert request.class_name == ""
        assert request.confidence == pytest.approx(0.0)

    def test_recommendation_response_tiene_recommendation(self):
        """
        RecommendationResponse debe tener campo recommendation (str).
        Es el texto en Markdown que se muestra al usuario en la GUI.
        """
        response = nlp_pb2.RecommendationResponse(
            recommendation="## Diagnostico\nTardio del tomate.",
            success=True,
            error="",
        )
        assert "Diagnostico" in response.recommendation
        assert response.success is True

    def test_recommendation_response_tiene_success_y_error(self):
        """
        success y error permiten al cliente distinguir entre
        respuesta valida y fallo del servicio NLP.
        """
        response_err = nlp_pb2.RecommendationResponse(
            recommendation="",
            success=False,
            error="API key no configurada",
        )
        assert response_err.success is False
        assert response_err.error == "API key no configurada"
        assert response_err.recommendation == ""

    def test_recommendation_response_campos_por_defecto(self):
        """Un RecommendationResponse vacio debe instanciarse sin errores."""
        response = nlp_pb2.RecommendationResponse()
        assert response.recommendation == ""
        assert response.success is False
        assert response.error == ""


# ==========================================================================
# CLASE 3 — VisionServiceServicer con servidor embebido
# ==========================================================================


class TestVisionServiceServicer:
    """
    Pruebas del VisionServiceServicer usando un servidor gRPC
    embebido en memoria. No requiere puerto ni servidor externo.
    """

    @pytest.fixture
    def servidor_cv(self):
        """
        Levanta un servidor gRPC CV en un puerto aleatorio en memoria.
        Lo detiene automaticamente al finalizar cada test.
        """
        from src.api.cv_server import VisionServiceServicer

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        vision_pb2_grpc.add_VisionServiceServicer_to_server(
            VisionServiceServicer(), server
        )
        port = server.add_insecure_port("[::]:0")  # puerto aleatorio
        server.start()
        yield port
        server.stop(grace=0)

    def test_imagen_vacia_retorna_success_false(self, servidor_cv):
        """
        Si image_data esta vacio, el servidor debe retornar success=False
        con un mensaje de error descriptivo, sin crashear.
        """
        channel = grpc.insecure_channel(f"localhost:{servidor_cv}")
        stub = vision_pb2_grpc.VisionServiceStub(channel)

        response = stub.ClassifyImage(
            vision_pb2.ImageRequest(image_data=b"", filename=""),
            timeout=5,
        )

        assert response.success is False
        assert response.error != ""
        channel.close()

    @patch("src.api.cv_server.log_inference")
    @patch("src.api.cv_server._classify_fn")
    def test_clasificacion_exitosa_retorna_contrato_completo(
        self, mock_classify, mock_log_inf, servidor_cv
    ):
        """
        Con una imagen valida, la respuesta debe cumplir el contrato completo:
        class_name, confidence, top_3, success=True, error=''.
        """
        mock_classify.return_value = {
            "class_name": "Tomato___Late_blight",
            "confidence": 0.91,
            "top_3": [
                {"label": "Tomato___Late_blight", "confidence": 0.91},
                {"label": "Tomato___Early_blight", "confidence": 0.06},
                {"label": "Tomato___healthy", "confidence": 0.03},
            ],
        }

        channel = grpc.insecure_channel(f"localhost:{servidor_cv}")
        stub = vision_pb2_grpc.VisionServiceStub(channel)

        response = stub.ClassifyImage(
            vision_pb2.ImageRequest(image_data=b"datos_imagen", filename="hoja.jpg"),
            timeout=5,
        )

        assert response.success is True
        assert response.class_name == "Tomato___Late_blight"
        assert response.confidence == pytest.approx(0.91, abs=0.01)
        assert len(response.top_3) == 3
        assert response.error == ""
        channel.close()


# ==========================================================================
# CLASE 4 — NLPServiceServicer con servidor embebido
# ==========================================================================


class TestNLPServiceServicer:
    """
    Pruebas del NLPServiceServicer usando un servidor gRPC
    embebido en memoria. No requiere puerto ni servidor externo.
    """

    @pytest.fixture
    def servidor_nlp(self):
        """
        Levanta un servidor gRPC NLP en un puerto aleatorio en memoria.
        Lo detiene automaticamente al finalizar cada test.
        """
        from src.api.nlp_server import NLPServiceServicer

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        nlp_pb2_grpc.add_NLPServiceServicer_to_server(NLPServiceServicer(), server)
        port = server.add_insecure_port("[::]:0")
        server.start()
        yield port
        server.stop(grace=0)

    def test_class_name_vacio_retorna_success_false(self, servidor_nlp):
        """
        Si class_name esta vacio, el servidor debe retornar success=False
        con un error descriptivo antes de llamar a OpenAI.
        """
        channel = grpc.insecure_channel(f"localhost:{servidor_nlp}")
        stub = nlp_pb2_grpc.NLPServiceStub(channel)

        response = stub.GetRecommendation(
            nlp_pb2.RecommendationRequest(class_name="", confidence=0.9),
            timeout=5,
        )

        assert response.success is False
        assert response.error != ""
        assert response.recommendation == ""
        channel.close()

    def test_confidence_fuera_de_rango_retorna_success_false(self, servidor_nlp):
        """
        Si confidence > 1.0, el servidor debe retornar success=False.
        Esto protege contra datos corruptos enviados por el cliente.
        """
        channel = grpc.insecure_channel(f"localhost:{servidor_nlp}")
        stub = nlp_pb2_grpc.NLPServiceStub(channel)

        response = stub.GetRecommendation(
            nlp_pb2.RecommendationRequest(
                class_name="Tomato___Late_blight", confidence=1.5
            ),
            timeout=5,
        )

        assert response.success is False
        channel.close()

    @patch("src.api.nlp_server.build_prompt")
    @patch("src.api.nlp_server.get_recommendation")
    @patch("src.api.nlp_server.mlflow_log")
    def test_request_valido_retorna_contrato_completo(
        self, mock_log, mock_rec, mock_prompt, servidor_nlp
    ):
        """
        Con request valido, la respuesta debe cumplir el contrato completo:
        recommendation (str), success=True, error=''.
        """
        mock_prompt.return_value = "prompt construido"
        mock_rec.return_value = "## Recomendacion\nAplicar fungicida."

        channel = grpc.insecure_channel(f"localhost:{servidor_nlp}")
        stub = nlp_pb2_grpc.NLPServiceStub(channel)

        response = stub.GetRecommendation(
            nlp_pb2.RecommendationRequest(
                class_name="Tomato___Late_blight", confidence=0.91
            ),
            timeout=5,
        )

        assert response.success is True
        assert "Recomendacion" in response.recommendation
        assert response.error == ""
        channel.close()

    @patch("src.api.nlp_server.build_prompt")
    @patch("src.api.nlp_server.get_recommendation")
    @patch("src.api.nlp_server.mlflow_log")
    def test_mlflow_se_llama_en_request_exitoso(
        self, mock_log, mock_rec, mock_prompt, servidor_nlp
    ):
        """
        Cuando el request es exitoso, mlflow_log debe llamarse exactamente
        una vez. Garantiza que cada inferencia queda registrada.
        """
        mock_prompt.return_value = "prompt"
        mock_rec.return_value = "recomendacion"

        channel = grpc.insecure_channel(f"localhost:{servidor_nlp}")
        stub = nlp_pb2_grpc.NLPServiceStub(channel)

        stub.GetRecommendation(
            nlp_pb2.RecommendationRequest(
                class_name="Tomato___Late_blight", confidence=0.91
            ),
            timeout=5,
        )

        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        assert call_kwargs.kwargs.get("success") is True
        channel.close()
