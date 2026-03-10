"""
Interfaz principal del Asistente Fitosanitario con IA.

Flujo WIZARD de 4 pasos — cada paso es una pantalla independiente:
    Paso 1 — Bienvenida: que hace el sistema, como funciona.
    Paso 2 — Cargar imagen: tips + uploader + preview.
    Paso 3 — Diagnostico: imagen + clasificacion del modelo CV.
    Paso 4 — Recomendacion: texto agronomico de Gemini Flash.

Solo se muestra el contenido del paso activo. Los demas desaparecen.
El indicador de navegacion (wizard nav) siempre esta visible arriba
mostrando en que paso esta el usuario.

Principio de diseno:
    - app.py es el ORQUESTADOR: controla el flujo del wizard.
    - Bajo acople: no conoce gRPC, protobuf ni detalles internos.
    - Alta cohesion: cada render_step_X es una pantalla completa.

Estructura de archivos de src/app/:
    config.py       -> Constantes y variables de entorno
    styles.py       -> Tema CSS y paleta de colores
    grpc_client.py  -> Comunicacion con servicios CV y NLP
    utils.py        -> Validacion, formateo, componentes visuales
    app.py          -> Interfaz Streamlit wizard (este archivo)

Ejecutar con: uv run streamlit run src/app/app.py

Autor: Nicolas Vasquez Renjifo
Modulo: Frontend & QA
Issues: #21 streamlit-app, #22 grpc-client-streamlit
Repositorio: github.com/Jhonatand26/PoC_DesProyecIA
"""

import streamlit as st  # Framework para la interfaz web
from PIL import Image  # Abrir imagen para mostrarla

# Importaciones internas del modulo src/app/
from config import (
    PAGE_TITLE,         # Titulo de la pagina
    PAGE_ICON,          # Icono de la pagina
    PAGE_LAYOUT,        # Layout de Streamlit
    ALLOWED_FORMATS     # Extensiones permitidas
)
from styles import inject_custom_css  # CSS personalizado
from grpc_client import (
    classify_image,      # Servicio CV — retorna dict
    get_recommendation   # Servicio NLP — retorna string
)
from utils import (
    validate_image,              # Valida imagen
    display_wizard_nav,          # Navegacion wizard
    display_image_with_diagnosis,  # Imagen + resultado CV
    display_top_predictions,     # Top-3 predicciones
    display_recommendation,      # Recomendacion NLP
    display_error                # Errores amigables
)


# ==========================================================================
# INICIALIZACION DEL ESTADO
# ==========================================================================

def init_session_state():
    """
    Inicializa las variables de estado del wizard en session_state.

    Streamlit re-ejecuta el script completo en cada interaccion.
    session_state persiste datos entre re-ejecuciones. Aqui se
    definen todas las variables que controlan el flujo del wizard.
    """
    # Paso actual del wizard (1 = bienvenida, 4 = recomendacion)
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1

    # Bytes de la imagen subida (persisten entre pasos)
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None

    # Nombre del archivo subido
    if "image_name" not in st.session_state:
        st.session_state.image_name = None

    # Resultado del servicio CV (dict o None)
    if "cv_result" not in st.session_state:
        st.session_state.cv_result = None

    # Recomendacion del servicio NLP (string o None)
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None


# ==========================================================================
# FUNCIONES DE NAVEGACION
# ==========================================================================

def go_to_step(step):
    """
    Cambia el wizard al paso indicado.

    Args:
        step (int): Numero del paso destino (1 a 4).
    """
    st.session_state.wizard_step = step


def reset_wizard():
    """
    Reinicia el wizard al paso 1 y limpia todos los datos.

    Se usa cuando el usuario quiere analizar una imagen nueva.
    """
    st.session_state.wizard_step = 1
    st.session_state.image_bytes = None
    st.session_state.image_name = None
    st.session_state.cv_result = None
    st.session_state.recommendation = None


# ==========================================================================
# COMPONENTES FIJOS (aparecen en todos los pasos)
# ==========================================================================

def render_page_config():
    """
    Configura metadatos de pagina e inyecta CSS. Una sola vez.
    """
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state="collapsed"
    )
    inject_custom_css()


def render_header():
    """
    Header principal — titulo y subtitulo. Fijo en todos los pasos.
    """
    st.title(f"{PAGE_ICON} Asistente Fitosanitario con IA")
    st.caption(
        "Diagnostico de enfermedades foliares con Inteligencia Artificial "
        "· Valle del Cauca, Colombia"
    )


def render_sidebar():
    """
    Sidebar con info del sistema y equipo.

    Note:
        >>> INTEGRACION JHONATAN (Issue #11, #17 — health check)
        Agregar indicadores de estado de servicios cuando esten listos.
    """
    with st.sidebar:
        st.header("Acerca del sistema")
        st.markdown(
            "**PoC Academico**\n\n"
            "Universidad Autonoma de Occidente\n\n"
            "Curso: Desarrollo de Proyectos de IA"
        )
        st.divider()
        st.subheader("Tecnologias")
        st.markdown(
            "**Vision:** MobileNetV2 · PlantVillage\n\n"
            "**NLP:** Google Gemini Flash 1.5\n\n"
            "**Comunicacion:** gRPC\n\n"
            "**Interfaz:** Streamlit"
        )
        st.divider()
        st.subheader("Equipo")
        st.markdown(
            "**Jhonatan** · Backend & Arquitectura\n\n"
            "**Jorge Luis** · Vision Computacional\n\n"
            "**Mateo** · NLP & Gemini\n\n"
            "**Nicolas** · Frontend & QA"
        )


# ==========================================================================
# PASO 1 — BIENVENIDA
# ==========================================================================

def render_step_1():
    """
    Pantalla de Bienvenida.

    Descripcion del sistema, 4 tarjetas del proceso, boton Comenzar.
    Todo el contenido de otros pasos esta oculto.
    """
    st.markdown("")  # Espacio

    st.markdown("### Identifica enfermedades en tus cultivos con una foto")
    st.markdown(
        "Sube una imagen de la hoja afectada y recibe un diagnostico "
        "con recomendaciones de tratamiento adaptadas al "
        "**Valle del Cauca**."
    )

    st.markdown("")  # Espacio

    # 4 tarjetas del proceso
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            '<div class="welcome-card">'
            '  <div class="welcome-icon">📷</div>'
            '  <div class="welcome-title">1. Sube foto</div>'
            '  <div class="welcome-desc">'
            '    Foto clara de la hoja afectada</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="welcome-card">'
            '  <div class="welcome-icon">🤖</div>'
            '  <div class="welcome-title">2. IA analiza</div>'
            '  <div class="welcome-desc">'
            '    MobileNetV2 clasifica la enfermedad</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            '<div class="welcome-card">'
            '  <div class="welcome-icon">🔬</div>'
            '  <div class="welcome-title">3. Diagnostico</div>'
            '  <div class="welcome-desc">'
            '    Resultado con nivel de confianza</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="welcome-card">'
            '  <div class="welcome-icon">💡</div>'
            '  <div class="welcome-title">4. Tratamiento</div>'
            '  <div class="welcome-desc">'
            '    Recomendacion regional</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("")  # Espacio

    # Info tecnica colapsable
    with st.expander("ℹ️ Informacion tecnica"):
        st.markdown(
            "**Modelo CV:** MobileNetV2 — PlantVillage "
            "(50,000+ imagenes, 38 clases, 95.4% accuracy).\n\n"
            "**Modelo NLP:** Google Gemini Flash 1.5 — "
            "recomendaciones para el Valle del Cauca.\n\n"
            "**Comunicacion:** gRPC (Google Remote Procedure Call).\n\n"
            "*Proyecto academico · UAO · 2026*"
        )

    st.markdown("")  # Espacio

    # Boton Comenzar
    st.button(
        "🚀 Comenzar diagnostico",
        type="primary",
        use_container_width=True,
        on_click=go_to_step,
        args=(2,)
    )


# ==========================================================================
# PASO 2 — CARGA DE IMAGEN
# ==========================================================================

def render_step_2():
    """
    Pantalla de carga de imagen.

    Tips para buenas fotos, file uploader, preview de la imagen,
    botones Volver y Diagnosticar.
    """
    st.markdown("### 📷 Sube la imagen de la hoja")

    st.markdown("")

    # Tips en 3 columnas
    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown(
            '<div class="tip-card">'
            '  <div class="tip-icon">☀️</div>'
            '  <div class="tip-title">Buena luz</div>'
            '  <div class="tip-desc">Evita sombras fuertes</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with t2:
        st.markdown(
            '<div class="tip-card">'
            '  <div class="tip-icon">🍃</div>'
            '  <div class="tip-title">Hoja centrada</div>'
            '  <div class="tip-desc">Que ocupe toda la foto</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with t3:
        st.markdown(
            '<div class="tip-card">'
            '  <div class="tip-icon">📱</div>'
            '  <div class="tip-title">JPG o PNG</div>'
            '  <div class="tip-desc">Hasta 10MB</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # File uploader
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona la imagen",
        type=ALLOWED_FORMATS,
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    # Procesar imagen subida
    if uploaded_file is not None:
        # Validar imagen
        is_valid, message = validate_image(uploaded_file)

        if not is_valid:
            display_error(
                "Imagen no valida",
                message,
                "Intenta con otra imagen en formato JPG o PNG."
            )
        else:
            # Guardar en session_state para que persista
            uploaded_file.seek(0)
            st.session_state.image_bytes = uploaded_file.read()
            st.session_state.image_name = uploaded_file.name

            # Mostrar preview
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption=f"📎 {uploaded_file.name}",
                use_container_width=True
            )

    st.markdown("")

    # Botones de navegacion
    col_back, col_next = st.columns(2)

    with col_back:
        st.button(
            "← Volver",
            use_container_width=True,
            on_click=go_to_step,
            args=(1,)
        )

    with col_next:
        # Solo habilitar si hay imagen cargada
        has_image = st.session_state.image_bytes is not None
        if has_image:
            st.button(
                "🔍 Diagnosticar enfermedad",
                type="primary",
                use_container_width=True,
                on_click=run_diagnosis
            )
        else:
            st.button(
                "🔍 Diagnosticar enfermedad",
                type="primary",
                use_container_width=True,
                disabled=True
            )


# ==========================================================================
# LOGICA DE DIAGNOSTICO (entre paso 2 y 3)
# ==========================================================================

def run_diagnosis():
    """
    Ejecuta el diagnostico completo: CV + NLP.

    Llamada como callback del boton Diagnosticar. Guarda los
    resultados en session_state y avanza al paso 3.
    """
    image_bytes = st.session_state.image_bytes

    if image_bytes is None:
        return

    # Llamar al servicio CV
    cv_result = classify_image(image_bytes)
    st.session_state.cv_result = cv_result

    if cv_result is not None:
        # Llamar al servicio NLP
        recommendation = get_recommendation(
            cv_result["class_name"],
            cv_result["confidence"]
        )
        st.session_state.recommendation = recommendation
        # Avanzar al paso 3
        st.session_state.wizard_step = 3
    else:
        # Error en CV — avanzar al paso 3 para mostrar error
        st.session_state.wizard_step = 3


# ==========================================================================
# PASO 3 — DIAGNOSTICO
# ==========================================================================

def render_step_3():
    """
    Pantalla de diagnostico visual.

    Si el servicio CV respondio: muestra imagen + resultado + top-3.
    Si fallo: muestra error amigable con sugerencia.
    Boton para avanzar a Recomendacion o para volver a intentar.
    """
    cv_result = st.session_state.cv_result

    if cv_result is None:
        # Error del servicio CV
        display_error(
            "Servicio no disponible",
            "No se pudo conectar al servicio de Vision Computacional. "
            "El servidor puede estar apagado o en mantenimiento.",
            "Verifica que los servicios gRPC esten corriendo y "
            "vuelve a intentarlo."
        )

        st.markdown("")

        col_back, col_retry = st.columns(2)
        with col_back:
            st.button(
                "← Subir otra imagen",
                use_container_width=True,
                on_click=go_to_step,
                args=(2,)
            )
        with col_retry:
            st.button(
                "🔄 Reintentar",
                type="primary",
                use_container_width=True,
                on_click=run_diagnosis
            )
        return

    # Diagnostico exitoso
    st.markdown("### 🔬 Diagnostico Visual")
    st.markdown("")

    # Reconstruir imagen desde bytes guardados
    import io
    image = Image.open(io.BytesIO(st.session_state.image_bytes))

    # Imagen + tarjeta de resultado
    display_image_with_diagnosis(image, cv_result)

    st.markdown("")

    # Top 3 predicciones
    display_top_predictions(cv_result["top_3"])

    st.markdown("")

    # Botones de navegacion
    col_back, col_next = st.columns(2)

    with col_back:
        st.button(
            "← Subir otra imagen",
            use_container_width=True,
            on_click=go_to_step,
            args=(2,)
        )

    with col_next:
        st.button(
            "💡 Ver recomendacion agronomica →",
            type="primary",
            use_container_width=True,
            on_click=go_to_step,
            args=(4,)
        )


# ==========================================================================
# PASO 4 — RECOMENDACION AGRONOMICA
# ==========================================================================

def render_step_4():
    """
    Pantalla de recomendacion agronomica.

    Muestra la recomendacion de Gemini Flash en tarjeta estilizada.
    Botones para volver al diagnostico o analizar nueva imagen.
    """
    st.markdown("### 💡 Recomendacion Agronomica")
    st.markdown("")

    # Mostrar recomendacion
    display_recommendation(st.session_state.recommendation)

    st.markdown("")

    # Botones de navegacion
    col_back, col_new = st.columns(2)

    with col_back:
        st.button(
            "← Volver al diagnostico",
            use_container_width=True,
            on_click=go_to_step,
            args=(3,)
        )

    with col_new:
        st.button(
            "📷 Analizar otra imagen",
            type="primary",
            use_container_width=True,
            on_click=reset_wizard
        )


# ==========================================================================
# FUNCION PRINCIPAL — ORQUESTADOR DEL WIZARD
# ==========================================================================

def main():
    """
    Funcion principal — orquesta el wizard de 4 pasos.

    Renderiza SOLO el contenido del paso activo. Los demas pasos
    no existen visualmente. El wizard nav siempre esta arriba
    mostrando el progreso.
    """
    # Configuracion inicial (una sola vez por sesion)
    render_page_config()
    init_session_state()

    # Header fijo
    render_header()

    # Sidebar fija
    render_sidebar()

    # Wizard nav — SIEMPRE visible, SIEMPRE arriba, UNA sola vez
    display_wizard_nav(current_step=st.session_state.wizard_step)

    # Renderizar SOLO el paso activo
    current = st.session_state.wizard_step

    if current == 1:
        render_step_1()
    elif current == 2:
        render_step_2()
    elif current == 3:
        render_step_3()
    elif current == 4:
        render_step_4()


# ==========================================================================
# PUNTO DE ENTRADA
# ==========================================================================

if __name__ == "__main__":
    main()