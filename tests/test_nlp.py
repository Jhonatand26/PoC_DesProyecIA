"""
Pruebas unitarias para el modulo NLP.
Issue #18 — minimo 4 pruebas requeridas por el curso.

Estrategia de mocking:
    - build_prompt: pruebas REALES, sin mocks. Es logica pura de strings.
    - get_recommendation: mock de OpenAI (API externa de pago).
    - mlflow_tracker: mock de mlflow (efecto secundario externo).
"""

import pytest
from unittest.mock import patch, MagicMock

from src.nlp.prompt_builder import build_prompt
from src.nlp.openai_client import get_recommendation


# ==========================================================================
# CLASE 1 — PromptBuilder (pruebas REALES, sin mocks)
# ==========================================================================


class TestBuildPrompt:
    """
    Pruebas de build_prompt sin mocks.
    Es logica pura de construccion de strings — no necesita mocks.
    """

    def test_contiene_nombre_enfermedad_formateado(self):
        """
        El prompt debe incluir el nombre legible de la enfermedad.
        '___' debe convertirse a ' - ' y '_' a espacio.
        """
        prompt = build_prompt("Tomato___Late_blight", 0.95)
        assert "Tomato - Late blight" in prompt

    def test_contiene_porcentaje_de_confianza(self):
        """El prompt debe incluir la confianza como porcentaje legible."""
        prompt = build_prompt("Tomato___Late_blight", 0.954)
        assert "95.4%" in prompt

    def test_contiene_contexto_valle_del_cauca(self):
        """
        El prompt debe mencionar el Valle del Cauca.
        Es el requisito de contextualizacion geografica del proyecto.
        """
        prompt = build_prompt("Corn___Common_rust", 0.80)
        assert "Valle del Cauca" in prompt

    def test_contiene_secciones_requeridas(self):
        """
        El prompt debe solicitar las tres secciones: DIAGNOSTICO,
        TRATAMIENTO y PREVENCION. Son el contrato con la respuesta de GPT.
        """
        prompt = build_prompt("Tomato___Late_blight", 0.95)
        assert "DIAGNOSTICO" in prompt
        assert "TRATAMIENTO" in prompt
        assert "PREVENCION" in prompt

    def test_retorna_string_no_vacio(self):
        """build_prompt siempre debe retornar un string con contenido."""
        prompt = build_prompt("Tomato___Late_blight", 0.95)
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_class_name_vacio_lanza_value_error(self):
        """Debe lanzar ValueError si class_name esta vacio."""
        with pytest.raises(ValueError, match="class_name"):
            build_prompt("", 0.9)

    def test_class_name_solo_espacios_lanza_value_error(self):
        """Un class_name de solo espacios equivale a vacio."""
        with pytest.raises(ValueError):
            build_prompt("   ", 0.9)

    def test_confianza_mayor_a_1_lanza_value_error(self):
        """Confidence > 1.0 es invalido para una probabilidad."""
        with pytest.raises(ValueError, match="confidence"):
            build_prompt("Tomato___Late_blight", 1.5)

    def test_confianza_negativa_lanza_value_error(self):
        """Confidence negativa es invalida."""
        with pytest.raises(ValueError):
            build_prompt("Tomato___Late_blight", -0.1)

    def test_confianza_en_limites_validos(self):
        """Los valores limite 0.0 y 1.0 deben ser validos."""
        prompt_min = build_prompt("Tomato___Late_blight", 0.0)
        prompt_max = build_prompt("Tomato___Late_blight", 1.0)
        assert isinstance(prompt_min, str)
        assert isinstance(prompt_max, str)


# ==========================================================================
# CLASE 2 — OpenAIClient (mock justificado: API externa de pago)
# ==========================================================================


class TestGetRecommendation:
    """
    Pruebas de get_recommendation.
    Se mockea OpenAI porque es una API externa de pago.
    Lo que se verifica es la logica del cliente: manejo de respuesta,
    errores y el strip() del contenido.
    """

    def _mock_openai_response(self, content: str):
        """Helper: construye un mock de la respuesta de OpenAI."""
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    @patch("src.nlp.openai_client.setup_experiment")
    @patch("src.nlp.openai_client._get_client")
    @patch("src.nlp.openai_client.log_recommendation")
    def test_retorna_texto_sin_espacios_extremos(self, mock_log, mock_get_client, mock_setup):
        """
        La respuesta de OpenAI puede tener espacios al inicio/final.
        get_recommendation debe aplicar strip() antes de retornar.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            "  Recomendacion con espacios  "
        )

        resultado = get_recommendation("prompt valido")

        assert resultado == "Recomendacion con espacios"

    @patch("src.nlp.openai_client.setup_experiment")
    @patch("src.nlp.openai_client._get_client")
    @patch("src.nlp.openai_client.log_recommendation")
    def test_llama_a_gpt5_nano(self, mock_log, mock_get_client, mock_setup):
        """
        get_recommendation debe usar el modelo gpt-5-nano.
        Cambiar el modelo por error consumiria creditos de un modelo mas caro.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            "respuesta"
        )

        get_recommendation("prompt valido")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5-nano"

    @patch("src.nlp.openai_client.setup_experiment")
    @patch("src.nlp.openai_client._get_client")
    @patch("src.nlp.openai_client.log_recommendation")
    def test_prompt_se_incluye_en_mensaje_user(self, mock_log, mock_get_client, mock_setup):
        """
        El prompt recibido debe enviarse en el rol 'user' del request.
        Es el contrato de entrada con el servidor NLP.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            "respuesta"
        )

        get_recommendation("mi prompt agronomico")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "mi prompt agronomico"

    @patch("src.nlp.openai_client.setup_experiment")
    @patch("src.nlp.openai_client._get_client")
    @patch("src.nlp.openai_client.log_recommendation")
    def test_sistema_tiene_contexto_agronomico(self, mock_log, mock_get_client, mock_setup):
        """
        El mensaje de sistema debe incluir contexto agronomico del Valle del Cauca.
        Sin contexto, GPT generaria respuestas genericas sin valor local.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            "respuesta"
        )

        get_recommendation("prompt")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert "Valle del Cauca" in system_messages[0]["content"]

    def test_prompt_vacio_lanza_value_error(self):
        """Debe lanzar ValueError antes de llamar a la API si el prompt esta vacio."""
        with pytest.raises(ValueError, match="prompt"):
            get_recommendation("")

    def test_prompt_solo_espacios_lanza_value_error(self):
        """Un prompt de solo espacios equivale a vacio."""
        with pytest.raises(ValueError):
            get_recommendation("   ")

    @patch("src.nlp.openai_client.setup_experiment")
    @patch("src.nlp.openai_client._get_client")
    @patch("src.nlp.openai_client.log_recommendation")
    def test_registra_en_mlflow_al_exito(self, mock_log, mock_get_client, mock_setup):
        """
        Cuando la llamada a OpenAI es exitosa, debe llamarse log_recommendation
        con success=True. Garantiza trazabilidad en MLflow.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            "recomendacion exitosa"
        )

        get_recommendation("prompt", class_name="Tomato___healthy", confidence=0.9)

        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        assert call_kwargs.kwargs.get("success") is True


# ==========================================================================
# CLASE 3 — MLflowTrackerNLP (mock de mlflow — efecto externo)
# ==========================================================================


class TestMLflowTrackerNLP:
    """
    Pruebas del tracker de MLflow para el modulo NLP.
    Se mockea mlflow porque es un efecto externo (servidor HTTP).
    """

    def _make_mock_run(self):
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        return mock_run

    @patch("src.nlp.mlflow_tracker.mlflow")
    def test_log_recommendation_registra_metricas_clave(self, mock_mlflow):
        """
        log_recommendation debe registrar latency_ms y confidence_score
        como metricas numericas comparables entre runs.
        """
        mock_mlflow.start_run.return_value = self._make_mock_run()

        from src.nlp.mlflow_tracker import log_recommendation

        log_recommendation(
            class_name="Tomato___Late_blight",
            confidence=0.91,
            prompt="prompt de prueba",
            recommendation="recomendacion de prueba",
            success=True,
            latency_ms=1500.0,
        )

        call_kwargs = mock_mlflow.log_metric.call_args_list
        metricas_registradas = {c.args[0] for c in call_kwargs}
        assert "latency_ms" in metricas_registradas or mock_mlflow.log_metrics.called

    @patch("src.nlp.mlflow_tracker.mlflow")
    def test_log_recommendation_no_lanza_excepcion_si_mlflow_falla(self, mock_mlflow):
        """
        Si MLflow no esta disponible, log_recommendation no debe
        crashear el servidor NLP. El tracking es opcional.
        """
        mock_mlflow.set_tracking_uri.side_effect = Exception("sin conexion")

        from src.nlp.mlflow_tracker import log_recommendation

        log_recommendation(
            class_name="clase",
            confidence=0.9,
            prompt="prompt",
            recommendation="rec",
            success=True,
        )
