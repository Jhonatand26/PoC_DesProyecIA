import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image

from src.vision.model_loader import load_model
from src.vision.preprocessor import preprocess_image
from src.vision.classifier import classify_image


@patch('src.vision.model_loader.pipeline')
def test_load_model(mock_pipeline):
    mock_pipeline.return_value = "mock_pipe"
    pipe = load_model("model_name", "cache_dir")

    assert pipe == "mock_pipe"
    mock_pipeline.assert_called_once_with(
        task="image-classification",
        model="model_name",
        model_kwargs={"cache_dir": "cache_dir"},
        trust_remote_code=True
    )


def test_preprocess_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_img_path = f.name
        img = Image.new("L", (100, 100))  # Escala de grises
        img.save(temp_img_path)

    try:
        processed_img = preprocess_image(temp_img_path)
        assert processed_img.size == (224, 224)
        assert processed_img.mode == "RGB"
    finally:
        os.remove(temp_img_path)


@patch('src.vision.classifier.load_model')
@patch('src.vision.classifier.preprocess_image')
def test_classify_image(mock_preprocess, mock_load):
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

    assert result["class"] == "Class_A"
    assert result["confidence"] == 0.95
    assert len(result["top_3"]) == 3
    assert result["top_3"][0]["class"] == "Class_A"

    mock_load.assert_called_once()
    mock_preprocess.assert_called_once_with("dummy_path.jpg")
    mock_pipe.assert_called_once_with("mock_img")
