"""
Configuracion centralizada de la aplicacion Streamlit.

Este modulo centraliza todas las constantes, variables de entorno
y parametros configurables de la interfaz del Asistente Fitosanitario.
Ningun otro modulo de src/app/ debe leer variables de entorno
directamente ni definir constantes propias — todo se importa de aqui.

Ejemplo de uso:
    from config import CV_SERVICE_PORT, ALLOWED_FORMATS

Autor: Nicolas Vasquez Renjifo
Modulo: Frontend & QA
Repositorio: github.com/Jhonatand26/PoC_DesProyecIA
"""

import os  # Acceso a variables de entorno del sistema
from dotenv import load_dotenv  # Carga variables desde archivo .env

# Cargar variables de entorno desde el archivo .env
# El archivo .env NO se sube al repositorio (esta en .gitignore)
# Cada integrante crea su propio .env localmente a partir de .env.example
load_dotenv()


# ==========================================================================
# CONFIGURACION DE SERVICIOS gRPC
# ==========================================================================
# >>> INTEGRACION JHONATAN (Issue #6 env-vars, #7 dockerfile)
# Estos valores deben coincidir con los que Jhonatan configure en
# docker-compose.yml y en el archivo .env del proyecto.
# En desarrollo local ambos servicios corren en localhost.
# En Docker, los nombres de host seran los nombres de los contenedores
# (ej: "cv-service", "nlp-service") definidos en docker-compose.yml.

CV_SERVICE_HOST = os.getenv("CV_SERVICE_HOST", "localhost")
"""str: Host del Servicio de Vision Computacional. Default: localhost."""

CV_SERVICE_PORT = os.getenv("CV_SERVICE_PORT", "50051")
"""str: Puerto del Servicio de Vision Computacional. Default: 50051."""

NLP_SERVICE_HOST = os.getenv("NLP_SERVICE_HOST", "localhost")
"""str: Host del Servicio de NLP (Gemini Flash). Default: localhost."""

NLP_SERVICE_PORT = os.getenv("NLP_SERVICE_PORT", "50052")
"""str: Puerto del Servicio de NLP (Gemini Flash). Default: 50052."""


# ==========================================================================
# CONFIGURACION DE LA INTERFAZ
# ==========================================================================

ALLOWED_FORMATS = ["jpg", "jpeg", "png"]
"""list[str]: Extensiones de imagen permitidas para subir al sistema."""

MAX_FILE_SIZE_MB = 10
"""int: Tamano maximo de archivo permitido en megabytes."""

CONFIDENCE_THRESHOLD = 0.60
"""float: Umbral minimo de confianza (0.0-1.0) para considerar un
diagnostico como confiable. Por debajo de este valor se muestra
una advertencia al usuario."""


# ==========================================================================
# CONFIGURACION DE LA PAGINA STREAMLIT
# ==========================================================================

PAGE_TITLE = "Asistente Fitosanitario IA"
"""str: Titulo que aparece en la pestana del navegador."""

PAGE_ICON = "🌿"
"""str: Icono que aparece en la pestana del navegador."""

PAGE_LAYOUT = "centered"
"""str: Layout de Streamlit. Opciones: 'centered' o 'wide'."""


# ==========================================================================
# CONFIGURACION DEL MODO DE EJECUCION
# ==========================================================================
# >>> INTEGRACION JHONATAN (Issue #6 env-vars)
# APP_ENV controla si la app usa stubs simulados o servicios reales.
# En desarrollo se usan stubs; en produccion se conecta a gRPC real.

APP_ENV = os.getenv("APP_ENV", "development")
"""str: Entorno de ejecucion. 'development' usa stubs simulados,
'production' conecta a los servicios gRPC reales."""

USE_STUBS = APP_ENV == "development"
"""bool: True si se deben usar stubs simulados en vez de gRPC real.
Se calcula automaticamente a partir de APP_ENV."""