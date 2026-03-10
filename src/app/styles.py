"""
Estilos visuales y tema CSS del Asistente Fitosanitario.

Este modulo centraliza TODOS los estilos CSS personalizados de la
interfaz Streamlit. Ningun otro archivo debe contener bloques <style>.
Si se necesita cambiar un color, fuente o espaciado, se hace aqui.

IMPORTANTE: Los estilos estan disenados para funcionar sobre el
tema OSCURO por defecto de Streamlit, sin forzar fondo claro.
La paleta verde/agricola se adapta a fondo oscuro.

Principio de diseno:
    - Alta cohesion: un unico archivo para todo lo visual/CSS.
    - Bajo acople: app.py y utils.py no conocen detalles de CSS,
      solo llaman inject_custom_css() una vez al inicio.

Autor: Nicolas Vasquez Renjifo
Modulo: Frontend & QA
Repositorio: github.com/Jhonatand26/PoC_DesProyecIA
"""

import streamlit as st  # Para inyectar CSS con st.markdown


# ==========================================================================
# PALETA DE COLORES — COMPATIBLE CON TEMA OSCURO
# ==========================================================================

COLORS = {
    # Verdes principales
    "primary": "#4CAF50",           # Verde principal — botones, acentos
    "primary_dark": "#2E7D32",      # Verde oscuro — hover
    "primary_light": "#81C784",     # Verde claro — textos destacados
    "primary_glow": "rgba(76,175,80,0.15)",  # Resplandor verde suave

    # Superficies (sobre fondo oscuro de Streamlit)
    "surface": "rgba(255,255,255,0.05)",     # Tarjetas — casi transparente
    "surface_hover": "rgba(255,255,255,0.08)",  # Tarjetas hover
    "border": "rgba(255,255,255,0.1)",       # Bordes sutiles

    # Texto (sobre fondo oscuro)
    "text_primary": "#FFFFFF",       # Texto principal — blanco
    "text_secondary": "#9E9E9E",     # Texto secundario — gris claro
    "text_muted": "#616161",         # Texto muy secundario

    # Estados de confianza
    "success": "#4CAF50",            # Verde — confianza alta
    "warning": "#FFA726",            # Naranja — confianza media
    "danger": "#EF5350",             # Rojo — confianza baja

    # Indicador de pasos
    "step_done": "#4CAF50",          # Verde — completado
    "step_active": "#FFFFFF",        # Blanco — activo
    "step_active_bg": "#4CAF50",     # Fondo verde — circulo activo
    "step_pending": "#424242",       # Gris oscuro — pendiente
    "step_pending_text": "#616161",  # Texto gris — pendiente
    "step_line_done": "#4CAF50",     # Linea verde — completada
    "step_line_pending": "#333333",  # Linea oscura — pendiente
}


# ==========================================================================
# CSS PRINCIPAL
# ==========================================================================

def get_main_css():
    """
    Retorna el CSS principal compatible con tema oscuro de Streamlit.

    Returns:
        str: Bloque CSS completo encerrado en tags <style>.
    """
    return f"""
    <style>
    /* =============================================================
       OCULTAMIENTO DE ELEMENTOS DEFAULT DE STREAMLIT
       ============================================================= */

    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}

    /* =============================================================
       TITULOS
       ============================================================= */

    /* Titulo principal — verde */
    h1 {{
        color: {COLORS["primary"]} !important;
        font-weight: 700 !important;
    }}

    /* =============================================================
       BOTONES
       ============================================================= */

    /* Boton primario — verde */
    .stButton > button[kind="primary"] {{
        background-color: {COLORS["primary"]} !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background-color: {COLORS["primary_dark"]} !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px {COLORS["primary_glow"]} !important;
    }}

    /* Boton secundario — outline verde */
    .stButton > button[kind="secondary"] {{
        background-color: transparent !important;
        color: {COLORS["primary"]} !important;
        border: 1px solid {COLORS["primary"]} !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button[kind="secondary"]:hover {{
        background-color: {COLORS["primary_glow"]} !important;
    }}

    /* =============================================================
       PROGRESS BAR — VERDE
       ============================================================= */

    .stProgress > div > div > div > div {{
        background-color: {COLORS["primary"]} !important;
    }}

    /* =============================================================
       IMAGENES — BORDES REDONDEADOS
       ============================================================= */

    [data-testid="stImage"] img {{
        border-radius: 12px !important;
        border: 1px solid {COLORS["border"]} !important;
    }}

    /* =============================================================
       INDICADOR DE PASOS (WIZARD NAV)
       ============================================================= */

    .wizard-nav {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1.2rem 0;
        margin-bottom: 1.5rem;
    }}

    .wizard-step {{
        display: flex;
        flex-direction: column;
        align-items: center;
        cursor: default;
        position: relative;
    }}

    .wizard-circle {{
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        z-index: 1;
    }}

    /* Paso completado */
    .wizard-done .wizard-circle {{
        background-color: {COLORS["step_done"]};
        color: #FFFFFF;
        font-weight: bold;
    }}

    .wizard-done .wizard-label {{
        color: {COLORS["step_done"]};
        font-weight: 500;
    }}

    /* Paso activo */
    .wizard-active .wizard-circle {{
        background-color: {COLORS["step_active_bg"]};
        color: #FFFFFF;
        box-shadow: 0 0 0 4px {COLORS["primary_glow"]},
                    0 0 20px {COLORS["primary_glow"]};
        font-weight: bold;
    }}

    .wizard-active .wizard-label {{
        color: {COLORS["text_primary"]};
        font-weight: 700;
    }}

    /* Paso pendiente */
    .wizard-pending .wizard-circle {{
        background-color: {COLORS["step_pending"]};
        color: {COLORS["step_pending_text"]};
    }}

    .wizard-pending .wizard-label {{
        color: {COLORS["step_pending_text"]};
    }}

    .wizard-label {{
        font-size: 0.75rem;
        text-align: center;
        transition: all 0.3s ease;
    }}

    /* Linea conectora */
    .wizard-line {{
        flex: 1;
        height: 3px;
        margin: 0 8px;
        margin-bottom: 26px;
        border-radius: 2px;
        z-index: 0;
    }}

    .wizard-line-done {{
        background-color: {COLORS["step_line_done"]};
    }}

    .wizard-line-pending {{
        background-color: {COLORS["step_line_pending"]};
    }}

    /* =============================================================
       TARJETAS GENERICAS (FONDO OSCURO)
       ============================================================= */

    .card {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
    }}

    .card:hover {{
        background-color: {COLORS["surface_hover"]};
    }}

    /* =============================================================
       TARJETAS DE BIENVENIDA
       ============================================================= */

    .welcome-card {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.3rem 1rem;
        text-align: center;
        transition: all 0.2s ease;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .welcome-card:hover {{
        background-color: {COLORS["surface_hover"]};
        border-color: {COLORS["primary"]};
        transform: translateY(-2px);
    }}

    .welcome-icon {{
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}

    .welcome-title {{
        font-weight: 700;
        color: {COLORS["text_primary"]};
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }}

    .welcome-desc {{
        font-size: 0.78rem;
        color: {COLORS["text_secondary"]};
        line-height: 1.3;
    }}

    /* =============================================================
       TARJETA DE DIAGNOSTICO
       ============================================================= */

    .diagnosis-card {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.5rem;
    }}

    .diagnosis-label {{
        font-size: 0.78rem;
        color: {COLORS["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }}

    .diagnosis-name {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }}

    .diagnosis-confidence {{
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
    }}

    .diagnosis-badge {{
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }}

    .badge-success {{
        background-color: rgba(76,175,80,0.15);
        color: {COLORS["success"]};
    }}

    .badge-warning {{
        background-color: rgba(255,167,38,0.15);
        color: {COLORS["warning"]};
    }}

    .badge-danger {{
        background-color: rgba(239,83,80,0.15);
        color: {COLORS["danger"]};
    }}

    /* =============================================================
       TARJETA DE RECOMENDACION
       ============================================================= */

    .reco-card {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.5rem 2rem;
    }}

    .reco-card h3 {{
        color: {COLORS["primary_light"]} !important;
        margin-top: 1rem !important;
        font-size: 1.1rem !important;
    }}

    .reco-card p {{
        color: {COLORS["text_primary"]};
        line-height: 1.6;
    }}

    .reco-card .reco-note {{
        font-style: italic;
        color: {COLORS["text_secondary"]};
        font-size: 0.85rem;
        margin-top: 1rem;
    }}

    /* =============================================================
       TIPS DE CARGA
       ============================================================= */

    .tip-card {{
        text-align: center;
        padding: 0.8rem;
    }}

    .tip-icon {{
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }}

    .tip-title {{
        font-weight: 600;
        color: {COLORS["text_primary"]};
        font-size: 0.85rem;
    }}

    .tip-desc {{
        font-size: 0.75rem;
        color: {COLORS["text_secondary"]};
    }}

    </style>
    """


# ==========================================================================
# FUNCION PUBLICA
# ==========================================================================

def inject_custom_css():
    """
    Inyecta todos los estilos CSS personalizados en la pagina.

    Debe llamarse UNA SOLA VEZ al inicio, despues de set_page_config().
    """
    st.markdown(get_main_css(), unsafe_allow_html=True)