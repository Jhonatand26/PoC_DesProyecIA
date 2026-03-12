import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image

from src.vision.model_loader import load_model
from src.vision.preprocessor import preprocess_image
from src.vision.classifier import classify_image


@patch('src.vision.model_loader.pipeline')
@patch('src.vision.model_loader.MobileNetV2ImageProcessor.from_pretrained')
@patch('src.vision.model_loader.AutoModelForImageClassification.from_pretrained')
def test_load_model_success(mock_auto_model, mock_processor, mock_pipeline):
    """Verifica que load_model retorna el pipeline del modelo principal."""
    mock_pipeline.return_value = "mock_pipe"
    pipe = load_model("model_name", "cache_dir")

    assert pipe == "mock_pipe"
    mock_pipeline.assert_called_once()


def test_preprocess_image():
    """
    Verifica que preprocess_image convierte a RGB y redimensiona a 224x224.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_img_path = f.name
        img = Image.new("L", (100, 100))
        img.save(temp_img_path)

    try:
        processed_img = preprocess_image(temp_img_path)
        assert processed_img.size == (224, 224)
        assert processed_img.mode == "RGB"
    finally:
        os.remove(temp_img_path)


@patch('src.vision.classifier.log_inference')
@patch('src.vision.classifier.load_model')
@patch('src.vision.classifier.preprocess_image')
def test_classify_image(mock_preprocess, mock_load, mock_log_inference):
    """
    Verifica que classify_image retorna clase, confianza y top_3 correctos.
    """
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {"label": "Class_A", "score": 0.95},
        {"label": "Class_B", "score": 0.03},
        {"label": "Class_C", "score": 0.01},
        {"label": "Class_D", "score": 0.01},
    ]
    mock_load.return_value = mock_pipe
    mock_preprocess.return_value = "mock_img"

    result = classify_image("dummy_path.jpg")

    assert result["class_name"] == "Class_A"
    assert result["confidence"] == 0.95
    assert len(result["top_3"]) == 3
    assert result["top_3"][0]["label"] == "Class_A"
    mock_load.assert_called_once()
    mock_preprocess.assert_called_once_with("dummy_path.jpg")
    mock_pipe.assert_called_once_with("mock_img")
    mock_log_inference.assert_called_once()

    # Verificar que se pasan model_version y model_source
    call_kwargs = mock_log_inference.call_args
    assert "model_version" in call_kwargs.kwargs
    assert "model_source" in call_kwargs.kwargs


@patch('src.vision.mlflow_tracker.mlflow')
def test_log_inference(mock_mlflow):
    """
    Verifica que log_inference llama a MLflow con los parámetros correctos,
    incluyendo model_version y model_source.
    """
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value = mock_run

    from src.vision.mlflow_tracker import log_inference
    log_inference(
        model_name="test-model",
        cache_dir="./cache",
        predicted_class="Tomato_healthy",
        confidence=0.95,
        inference_time_ms=120.5,
        model_version="v1.0",
        model_source="https://huggingface.co/test-model",
    )

    mock_mlflow.log_params.assert_called_once_with({
        "model_name": "test-model",
        "model_version": "v1.0",
        "model_source": "https://huggingface.co/test-model",
        "cache_dir": "./cache",
    })
    mock_mlflow.log_metrics.assert_called_once_with({
        "confidence": 0.95,
        "inference_time_ms": 120.5,
    })
    mock_mlflow.set_tag.assert_called_once_with(
        "predicted_class", "Tomato_healthy"
    )


@patch('src.vision.mlflow_tracker.mlflow')
def test_log_model_metrics(mock_mlflow):
    """
    Verifica que log_model_metrics registra accuracy y f1_score del paper.
    """
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value = mock_run

    from src.vision.mlflow_tracker import log_model_metrics
    log_model_metrics(
        accuracy=0.9970,
        f1_score=0.9932,
        model_name="test-model",
        model_version="v1.0",
    )

    mock_mlflow.log_params.assert_called_once_with({
        "model_name": "test-model",
        "model_version": "v1.0",
        "metrics_source": "paper",
    })
    mock_mlflow.log_metrics.assert_called_once_with({
        "accuracy": 0.9970,
        "f1_score": 0.9932,
    })


@patch('src.vision.mlflow_tracker.mlflow')
def test_register_model(mock_mlflow):
    """
    Verifica que register_model llama a MLflow con la configuración correcta.
    """
    mock_run_info = MagicMock()
    mock_run_info.info.run_id = "test-run-id"
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run_info)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value = mock_run

    from src.vision.mlflow_tracker import register_model
    register_model(
        model_name="test-model",
        model_version="v1.0",
        model_source="https://huggingface.co/test-model",
        accuracy=0.9970,
        f1_score=0.9932,
    )

    mock_mlflow.log_params.assert_called_once_with({
        "model_name": "test-model",
        "model_version": "v1.0",
        "model_source": "https://huggingface.co/test-model",
    })
    mock_mlflow.log_metrics.assert_called_once_with({
        "accuracy": 0.9970,
        "f1_score": 0.9932,
    })
    mock_mlflow.set_tag.assert_any_call("model_type", "huggingface")