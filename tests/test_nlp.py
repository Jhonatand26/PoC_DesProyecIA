"""
Pruebas unitarias para el modulo NLP.
Issue #18 — minimo 4 pruebas requeridas por el curso.
Las pruebas de openai_client usan mocks para no consumir la API real.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.nlp.prompt_builder import build_prompt
from src.nlp.openai_client import get_recommendation


class TestBuildPrompt:
    """Pruebas para la funcion build_prompt."""

    def test_prompt_contiene_nombre_enfermedad(self):
        """El prompt debe incluir el nombre de la enfermedad formateado."""
        prompt = build_prompt("Tomato___Late_blight", 0.95)
        assert "Tomato - Late blight" in prompt

    def test_prompt_contiene_confianza(self):
        """El prompt debe incluir el porcentaje de confianza."""
        prompt = build_prompt("Tomato___Late_blight", 0.954)
        assert "95.4%" in prompt

    def test_class_name_vacio_lanza_error(self):
        """Debe lanzar ValueError si class_name esta vacio."""
        with pytest.raises(ValueError):
            build_prompt("", 0.9)

    def test_confianza_invalida_lanza_error(self):
        """Debe lanzar ValueError si confidence esta fuera de [0, 1]."""
        with pytest.raises(ValueError):
            build_prompt("Tomato___Late_blight", 1.5)


class TestGetRecommendation:
    """Pruebas para la funcion get_recommendation (con mock de la API)."""

    @patch("src.nlp.openai_client.OpenAI")
    def test_retorna_texto_de_la_api(self, mock_openai_class):
        """Debe retornar el contenido del mensaje de la respuesta de la API."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="  Recomendacion de prueba  "
            ))]
        )
        resultado = get_recommendation("prompt de prueba")
        assert resultado == "Recomendacion de prueba"

    def test_prompt_vacio_lanza_error(self):
        """Debe lanzar ValueError si el prompt esta vacio."""
        with pytest.raises(ValueError):
            get_recommendation("")
