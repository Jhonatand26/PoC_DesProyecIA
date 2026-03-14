"""
Pruebas unitarias para el modulo de Vision Computacional.
Issue #11 — minimo 4 pruebas requeridas por el curso.

Estrategia de mocking:
    - preprocessor y model_loader: pruebas reales donde es posible.
    - classifier: mock solo de load_model (costoso, descarga HuggingFace)
      pero preprocess_image se ejecuta de verdad con imagen temporal.
    - mlflow_tracker: mock de mlflow (efecto secundario externo).
"""

import os
import tempfile
from unittest.mock import patch, MagicMock, call
from PIL import Image
import pytest

from src.vision.preprocessor import preprocess_image
from src.vision.model_loader import load_model
from src.vision.classifier import classify_image


# ==========================================================================
# FIXTURES — datos de prueba compartidos
# ==========================================================================


@pytest.fixture
def imagen_temporal_rgb():
    """
    Crea una imagen RGB temporal de 300x300 en disco.
    Simula una foto real de hoja subida por el usuario.
    Se elimina automaticamente al finalizar el test.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        Image.new("RGB", (300, 300), color=(34, 139, 34)).save(path)
    yield path
    os.remove(path)


@pytest.fixture
def imagen_temporal_escala_grises():
    """
    Crea una imagen en escala de grises (modo L) temporal.
    Prueba la conversion a RGB del preprocessor.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        Image.new("L", (100, 100)).save(path)
    yield path
    os.remove(path)


@pytest.fixture
def resultados_pipeline_mock():
    """
    Simula la salida del pipeline de HuggingFace con 4 clases.
    Estructura real: lista de dicts con 'label' y 'score'.
    """
    return [
        {"label": "Tomato___Late_blight", "score": 0.91},
        {"label": "Tomato___Early_blight", "score": 0.06},
        {"label": "Tomato___healthy", "score": 0.02},
        {"label": "Tomato___Leaf_Mold", "score": 0.01},
    ]


# ==========================================================================
# CLASE 1 — Preprocessor (pruebas REALES, sin mocks)
# ==========================================================================


class TestPreprocessor:
    """
    Pruebas del preprocessor sin mocks.
    preprocess_image es pura logica de PIL — no necesita mocks.
    """

    def test_convierte_escala_grises_a_rgb(self, imagen_temporal_escala_grises):
        """
        Una imagen en modo L (escala de grises) debe convertirse a RGB.
        Caso real: camaras moviles pueden generar imagenes en escala de grises.
        """
        resultado = preprocess_image(imagen_temporal_escala_grises)
        assert resultado.mode == "RGB"

    def test_redimensiona_a_224x224(self, imagen_temporal_rgb):
        """
        Cualquier imagen debe salir con 224x224, el tamano que espera MobileNetV2.
        """
        resultado = preprocess_image(imagen_temporal_rgb)
        assert resultado.size == (224, 224)

    def test_imagen_rgb_grande_se_redimensiona(self):
        """
        Una imagen RGB de alta resolucion (4K) debe redimensionarse correctamente.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
            Image.new("RGB", (3840, 2160)).save(path)

        try:
            resultado = preprocess_image(path)
            assert resultado.size == (224, 224)
            assert resultado.mode == "RGB"
        finally:
            os.remove(path)

    def test_retorna_objeto_pil_image(self, imagen_temporal_rgb):
        """El tipo de retorno debe ser PIL.Image.Image."""
        resultado = preprocess_image(imagen_temporal_rgb)
        assert isinstance(resultado, Image.Image)


# ==========================================================================
# CLASE 2 — ModelLoader (mock justificado: descarga HuggingFace ~50MB)
# ==========================================================================


class TestModelLoader:
    """
    Pruebas de load_model.
    Se mockea la descarga de HuggingFace porque es un efecto externo
    costoso (red + disco). Lo que se verifica es la logica de composicion.
    """

    @patch("src.vision.model_loader.pipeline")
    @patch("src.vision.model_loader.MobileNetV2ImageProcessor.from_pretrained")
    @patch("src.vision.model_loader.AutoModelForImageClassification.from_pretrained")
    def test_retorna_pipeline_configurado(
        self, mock_auto_model, mock_processor, mock_pipeline
    ):
        """
        load_model debe construir el pipeline con el modelo y processor cargados.
        Verifica que la composicion model + feature_extractor es correcta.
        """
        mock_pipeline.return_value = "pipeline_real"

        resultado = load_model("nombre-modelo", "./cache")

        assert resultado == "pipeline_real"
        mock_auto_model.assert_called_once_with(
            "nombre-modelo", cache_dir="./cache", trust_remote_code=True
        )
        mock_processor.assert_called_once_with(
            "nombre-modelo", cache_dir="./cache", trust_remote_code=True
        )
        mock_pipeline.assert_called_once()

    @patch("src.vision.model_loader.pipeline")
    @patch("src.vision.model_loader.MobileNetV2ImageProcessor.from_pretrained")
    @patch("src.vision.model_loader.AutoModelForImageClassification.from_pretrained")
    def test_usa_task_image_classification(
        self, mock_auto_model, mock_processor, mock_pipeline
    ):
        """
        El pipeline debe configurarse con task='image-classification'.
        Esto es el contrato con HuggingFace Transformers.
        """
        load_model("nombre-modelo", "./cache")

        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get("task") == "image-classification"


# ==========================================================================
# CLASE 3 — Classifier (mock de load_model, preprocess real con fixture)
# ==========================================================================


class TestClassifier:
    """
    Pruebas de classify_image.
    Se mockea load_model (descarga HuggingFace) pero preprocess_image
    se ejecuta de verdad con una imagen temporal real.
    """

    @patch("src.vision.classifier.log_inference")
    @patch("src.vision.classifier.load_model")
    def test_retorna_clase_con_mayor_confianza(
        self, mock_load, mock_log, imagen_temporal_rgb, resultados_pipeline_mock
    ):
        """
        classify_image debe retornar la clase con mayor score como class_name.
        Se usa una imagen real para que preprocess_image corra de verdad.
        """
        mock_pipe = MagicMock(return_value=resultados_pipeline_mock)
        mock_load.return_value = mock_pipe

        resultado = classify_image(imagen_temporal_rgb)

        assert resultado["class_name"] == "Tomato___Late_blight"
        assert resultado["confidence"] == 0.91

    @patch("src.vision.classifier.log_inference")
    @patch("src.vision.classifier.load_model")
    def test_top_3_tiene_exactamente_3_elementos(
        self, mock_load, mock_log, imagen_temporal_rgb, resultados_pipeline_mock
    ):
        """
        top_3 debe tener exactamente 3 elementos sin importar cuantas
        clases retorne el pipeline (el mock retorna 4).
        """
        mock_pipe = MagicMock(return_value=resultados_pipeline_mock)
        mock_load.return_value = mock_pipe

        resultado = classify_image(imagen_temporal_rgb)

        assert len(resultado["top_3"]) == 3

    @patch("src.vision.classifier.log_inference")
    @patch("src.vision.classifier.load_model")
    def test_top_3_tiene_claves_label_y_confidence(
        self, mock_load, mock_log, imagen_temporal_rgb, resultados_pipeline_mock
    ):
        """
        Cada elemento de top_3 debe tener exactamente las claves
        'label' y 'confidence'. Este es el contrato con cv_server.py.
        """
        mock_pipe = MagicMock(return_value=resultados_pipeline_mock)
        mock_load.return_value = mock_pipe

        resultado = classify_image(imagen_temporal_rgb)

        for pred in resultado["top_3"]:
            assert "label" in pred
            assert "confidence" in pred

    @patch("src.vision.classifier.log_inference")
    @patch("src.vision.classifier.load_model")
    def test_confidence_es_float(
        self, mock_load, mock_log, imagen_temporal_rgb, resultados_pipeline_mock
    ):
        """
        confidence debe ser float, no numpy.float32 ni otro tipo.
        gRPC protobuf requiere float nativo de Python.
        """
        mock_pipe = MagicMock(return_value=resultados_pipeline_mock)
        mock_load.return_value = mock_pipe

        resultado = classify_image(imagen_temporal_rgb)

        assert isinstance(resultado["confidence"], float)
        for pred in resultado["top_3"]:
            assert isinstance(pred["confidence"], float)

    @patch("src.vision.classifier.log_inference")
    @patch("src.vision.classifier.load_model")
    def test_llama_log_inference_una_vez(
        self, mock_load, mock_log, imagen_temporal_rgb, resultados_pipeline_mock
    ):
        """
        classify_image debe llamar log_inference exactamente una vez por inferencia.
        Mas de una llamada significaria runs duplicados en MLflow.
        """
        mock_pipe = MagicMock(return_value=resultados_pipeline_mock)
        mock_load.return_value = mock_pipe

        classify_image(imagen_temporal_rgb)

        mock_log.assert_called_once()


# ==========================================================================
# CLASE 4 — MLflowTracker (mock de mlflow — efecto secundario externo)
# ==========================================================================


class TestMLflowTracker:
    """
    Pruebas del tracker de MLflow para el modulo CV.
    Se mockea mlflow porque es un efecto externo (servidor HTTP).
    Lo que se verifica es que el tracker llama a mlflow con los
    parametros exactos que el servidor MLflow espera.
    """

    def _make_mock_run(self):
        """Helper: crea un mock de mlflow.start_run() como context manager."""
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        return mock_run

    @patch("src.vision.mlflow_tracker.mlflow")
    def test_log_inference_registra_parametros_correctos(self, mock_mlflow):
        """
        log_inference debe llamar log_params con exactamente los campos
        que el experimento plant-disease-cv-v2 requiere.
        """
        mock_mlflow.start_run.return_value = self._make_mock_run()

        from src.vision.mlflow_tracker import log_inference

        log_inference(
            model_name="test-model",
            cache_dir="./cache",
            predicted_class="Tomato___healthy",
            confidence=0.95,
            inference_time_ms=120.5,
            model_version="v1.0",
            model_source="https://huggingface.co/test-model",
        )

        mock_mlflow.log_params.assert_called_once_with(
            {
                "model_name": "test-model",
                "model_version": "v1.0",
                "model_source": "https://huggingface.co/test-model",
                "cache_dir": "./cache",
            }
        )

    @patch("src.vision.mlflow_tracker.mlflow")
    def test_log_inference_registra_metricas_correctas(self, mock_mlflow):
        """
        log_inference debe registrar confidence e inference_time_ms
        como metricas numericas (no params).
        """
        mock_mlflow.start_run.return_value = self._make_mock_run()

        from src.vision.mlflow_tracker import log_inference

        log_inference(
            model_name="m",
            cache_dir="./c",
            predicted_class="Tomato___healthy",
            confidence=0.87,
            inference_time_ms=95.3,
        )

        mock_mlflow.log_metrics.assert_called_once_with(
            {
                "confidence": 0.87,
                "inference_time_ms": 95.3,
            }
        )

    @patch("src.vision.mlflow_tracker.mlflow")
    def test_log_inference_registra_clase_como_tag(self, mock_mlflow):
        """
        predicted_class debe registrarse como tag, no como parametro ni metrica.
        Los tags en MLflow son para categorizar runs, no para comparar valores.
        """
        mock_mlflow.start_run.return_value = self._make_mock_run()

        from src.vision.mlflow_tracker import log_inference

        log_inference(
            model_name="m",
            cache_dir="./c",
            predicted_class="Tomato___Late_blight",
            confidence=0.91,
            inference_time_ms=110.0,
        )

        mock_mlflow.set_tag.assert_called_once_with(
            "predicted_class", "Tomato___Late_blight"
        )

    @patch("src.vision.mlflow_tracker.mlflow")
    def test_log_inference_no_lanza_excepcion_si_mlflow_falla(self, mock_mlflow):
        """
        Si MLflow no esta disponible, log_inference no debe crashear el servidor.
        El tracking es opcional — la inferencia debe continuar.
        """
        mock_mlflow.set_tracking_uri.side_effect = Exception("MLflow no disponible")

        from src.vision.mlflow_tracker import log_inference

        # No debe lanzar excepcion
        log_inference(
            model_name="m",
            cache_dir="./c",
            predicted_class="clase",
            confidence=0.9,
            inference_time_ms=100.0,
        )

    @patch("src.vision.mlflow_tracker.mlflow")
    def test_log_model_metrics_registra_accuracy_y_f1(self, mock_mlflow):
        """
        log_model_metrics debe registrar accuracy y f1_score del paper.
        """
        mock_mlflow.start_run.return_value = self._make_mock_run()

        from src.vision.mlflow_tracker import log_model_metrics

        log_model_metrics(
            accuracy=0.9970,
            f1_score=0.9932,
            model_name="test-model",
            model_version="v1.0",
        )

        mock_mlflow.log_metrics.assert_called_once_with(
            {
                "accuracy": 0.9970,
                "f1_score": 0.9932,
            }
        )
