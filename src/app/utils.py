"""
Funciones de utilidad para la interfaz del Asistente Fitosanitario.

Este modulo contiene funciones auxiliares de validacion, formateo
y componentes visuales reutilizables. No tiene ninguna dependencia
de gRPC ni de los servicios externos — solo depende de Streamlit,
Pillow y de config.py.

Principio de diseno:
    - Alta cohesion: cada funcion tiene una unica responsabilidad.
    - Bajo acople: este modulo NO conoce gRPC, protobuf ni la API
      de Gemini. Solo trabaja con datos primitivos (strings, floats,
      bytes, diccionarios).

Autor: Nicolas Vasquez Renjifo
Modulo: Frontend & QA
Repositorio: github.com/Jhonatand26/PoC_DesProyecIA
"""

import streamlit as st  # Componentes visuales de Streamlit
from PIL import Image  # Procesamiento y validacion de imagenes
from config import ALLOWED_FORMATS, CONFIDENCE_THRESHOLD  # Constantes
from styles import COLORS  # Paleta de colores del tema


# ==========================================================================
# NAVEGACION WIZARD — INDICADOR DE PASOS
# ==========================================================================

def display_wizard_nav(current_step):
    """
    Muestra la barra de navegacion wizard con 4 pasos.

    Renderiza circulos conectados por lineas. El paso activo tiene
    un resplandor verde, los completados muestran check verde,
    y los pendientes estan atenuados. Compatible con tema oscuro.

    Args:
        current_step (int): Paso actual del wizard (1 a 4).
    """
    steps = [
        ("🏠", "Bienvenida"),
        ("📷", "Carga"),
        ("🔬", "Diagnostico"),
        ("💡", "Recomendacion"),
    ]

    html = ['<div class="wizard-nav">']

    for i, (icon, label) in enumerate(steps):
        step_num = i + 1

        # Determinar estado visual del paso
        if step_num < current_step:
            state = "wizard-done"
            content = "✓"
        elif step_num == current_step:
            state = "wizard-active"
            content = icon
        else:
            state = "wizard-pending"
            content = icon

        # Agregar paso
        html.append(
            f'<div class="wizard-step {state}">'
            f'  <div class="wizard-circle">{content}</div>'
            f'  <div class="wizard-label">{label}</div>'
            f'</div>'
        )

        # Linea conectora (excepto despues del ultimo)
        if i < len(steps) - 1:
            line = "wizard-line-done" if step_num < current_step \
                else "wizard-line-pending"
            html.append(f'<div class="wizard-line {line}"></div>')

    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ==========================================================================
# VALIDACION DE IMAGEN
# ==========================================================================

def validate_image(uploaded_file):
    """
    Valida que el archivo subido sea una imagen en formato permitido.

    Verifica tres condiciones:
        1. Que se haya subido un archivo (no sea None).
        2. Que la extension este en ALLOWED_FORMATS.
        3. Que Pillow pueda abrir y verificar la imagen.

    Args:
        uploaded_file (streamlit.UploadedFile): Archivo subido.

    Returns:
        tuple[bool, str]: (es_valida, mensaje).

    Note:
        >>> INTEGRACION JORGE LUIS (Issue #9 preprocessor.py)
        Validacion de formato unicamente. El preprocesamiento
        (resize 224x224, RGB) ocurre en el servicio CV.
    """
    if uploaded_file is None:
        return False, "No se ha subido ninguna imagen."

    file_name = uploaded_file.name
    file_extension = file_name.split(".")[-1].lower()

    if file_extension not in ALLOWED_FORMATS:
        return False, (
            f"Formato '{file_extension}' no soportado. "
            f"Usa: {', '.join(ALLOWED_FORMATS)}"
        )

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        return True, "Imagen valida."
    except Exception as e:
        return False, f"El archivo no es una imagen valida: {str(e)}"


# ==========================================================================
# FORMATEO
# ==========================================================================

def format_class_name(raw_class_name):
    """
    Convierte nombre crudo de clase PlantVillage a formato legible.

    Args:
        raw_class_name (str): Ej: "Tomato___Late_blight"

    Returns:
        str: Ej: "Tomato - Late blight"

    Note:
        >>> INTEGRACION JORGE LUIS (Issue #10 classifier.py)
        Si cambia el formato de salida en classifier.py,
        actualizar esta funcion.
    """
    formatted = raw_class_name.replace("___", " - ")
    formatted = formatted.replace("_", " ")
    return formatted


def get_confidence_color(confidence):
    """
    Retorna color hexadecimal segun nivel de confianza.

    Args:
        confidence (float): Valor entre 0.0 y 1.0.

    Returns:
        str: Color hex de la paleta del tema.
    """
    if confidence >= CONFIDENCE_THRESHOLD:
        return COLORS["success"]
    elif confidence >= (CONFIDENCE_THRESHOLD * 0.6):
        return COLORS["warning"]
    else:
        return COLORS["danger"]


def get_confidence_badge(confidence):
    """
    Retorna clase CSS y texto del badge segun confianza.

    Args:
        confidence (float): Valor entre 0.0 y 1.0.

    Returns:
        tuple[str, str]: (clase_css, texto_badge).
    """
    if confidence >= CONFIDENCE_THRESHOLD:
        return "badge-success", "✓ Diagnostico confiable"
    elif confidence >= (CONFIDENCE_THRESHOLD * 0.6):
        return "badge-warning", "⚠ Confianza moderada"
    else:
        return "badge-danger", "✗ Confianza baja — intenta otra foto"


# ==========================================================================
# COMPONENTES VISUALES
# ==========================================================================

def display_image_with_diagnosis(image, cv_result):
    """
    Muestra imagen + tarjeta de diagnostico lado a lado.

    Dos columnas: imagen a la izquierda, tarjeta estilizada a la
    derecha con clase, confianza en color, y badge.

    Args:
        image (PIL.Image): Imagen subida.
        cv_result (dict): Resultado CV con class_name, confidence, top_3.

    Note:
        >>> INTEGRACION JORGE LUIS (Issue #10 classifier.py)
        MobileNetV2 es clasificador, no detector. No hay bounding box.
    """
    class_name = cv_result["class_name"]
    confidence = cv_result["confidence"]
    color = get_confidence_color(confidence)
    badge_class, badge_text = get_confidence_badge(confidence)

    col_img, col_result = st.columns([11, 9])

    with col_img:
        st.image(image, use_container_width=True)

    with col_result:
        st.markdown(
            f'<div class="diagnosis-card">'
            f'  <div class="diagnosis-label">Enfermedad detectada</div>'
            f'  <div class="diagnosis-name">'
            f'    {format_class_name(class_name)}'
            f'  </div>'
            f'  <div class="diagnosis-confidence" style="color:{color};">'
            f'    {confidence:.1%}'
            f'  </div>'
            f'  <div class="diagnosis-badge {badge_class}">'
            f'    {badge_text}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )


def display_top_predictions(top_3):
    """
    Muestra top-3 predicciones en expander colapsable con barras.

    Args:
        top_3 (list[dict]): Lista con class_name y confidence.

    Note:
        >>> INTEGRACION JORGE LUIS (Issue #10 classifier.py)
        Estructura debe coincidir con classifier.py.
    """
    with st.expander("📊 Ver top 3 predicciones del modelo"):
        for i, pred in enumerate(top_3, start=1):
            name = format_class_name(pred["class_name"])
            conf = pred["confidence"]
            color = get_confidence_color(conf)

            st.markdown(
                f"**{i}.** {name} — "
                f"<span style='color:{color}; font-weight:bold;'>"
                f"{conf:.1%}</span>",
                unsafe_allow_html=True
            )
            st.progress(conf)


def display_recommendation(recommendation):
    """
    Muestra recomendacion agronomica en tarjeta estilizada.

    Args:
        recommendation (str or None): Texto markdown o None.

    Note:
        >>> INTEGRACION MATEO (Issue #15 prompt_builder.py)
        Formato esperado: Diagnostico + Tratamiento + Prevencion.
    """
    if recommendation is not None:
        html_content = _markdown_to_html_basic(recommendation)
        st.markdown(
            f'<div class="reco-card">{html_content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.error(
            "❌ No se pudo generar la recomendacion agronomica. "
            "El servicio de NLP no esta disponible en este momento. "
            "Se muestra unicamente el diagnostico visual."
        )


def display_error(title, message, suggestion=None):
    """
    Muestra un mensaje de error amigable y consistente.

    Centraliza el estilo de errores para que todos se vean igual
    y siempre incluyan una sugerencia de accion para el usuario.

    Args:
        title (str): Titulo corto del error.
        message (str): Descripcion del problema.
        suggestion (str, optional): Que puede hacer el usuario.
    """
    error_text = f"**{title}**\n\n{message}"
    if suggestion:
        error_text += f"\n\n💡 *{suggestion}*"
    st.error(error_text)


# ==========================================================================
# UTILIDAD INTERNA — MARKDOWN A HTML
# ==========================================================================

def _markdown_to_html_basic(text):
    """
    Convierte markdown basico a HTML para usar dentro de tarjetas.

    Solo soporta: ## headers, **bold**, - listas, *italics*.
    Suficiente para la estructura de la recomendacion de Gemini.

    Args:
        text (str): Texto en markdown basico.

    Returns:
        str: HTML formateado.
    """
    import re

    lines = text.split("\n")
    html_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            html_lines.append("<br>")
        elif stripped.startswith("## "):
            content = stripped[3:]
            html_lines.append(f'<h3>{content}</h3>')
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content = stripped[2:]
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>',
                             content)
            html_lines.append(
                f'<p style="margin:0.3rem 0 0.3rem 1rem;">'
                f'• {content}</p>'
            )
        elif stripped.startswith("*") and stripped.endswith("*") \
                and not stripped.startswith("**"):
            content = stripped.strip("*")
            html_lines.append(
                f'<p class="reco-note">{content}</p>'
            )
        else:
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>',
                             stripped)
            html_lines.append(f'<p style="margin:0.2rem 0;">{content}</p>')

    return "\n".join(html_lines)