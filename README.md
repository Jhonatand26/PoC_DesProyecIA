# PoC Asistente Fitosanitario con IA

**Prueba de Concepto** — Diagnóstico de enfermedades foliares en cultivos del Valle del Cauca mediante Inteligencia Artificial.

> Proyecto académico · Universidad Autónoma de Occidente · Curso: Desarrollo de Proyectos de IA · 2026

---

## Descripción

Sistema que permite a agricultores del Valle del Cauca (Colombia) diagnosticar enfermedades en sus cultivos a partir de una fotografía de la hoja afectada. El flujo es:

1. El usuario sube una foto de la hoja desde la interfaz web (Streamlit)
2. El servicio de **Visión Computacional** clasifica la enfermedad usando MobileNetV2 entrenado con PlantVillage (~50,000 imágenes, 38 clases)
3. El servicio de **NLP** genera una recomendación agronómica contextualizada al Valle del Cauca usando GPT-5 Nano
4. **MLflow** registra cada inferencia para trazabilidad de experimentos

La comunicación entre servicios se realiza mediante **gRPC** (Google Remote Procedure Call).

---

## Arquitectura

```
┌─────────────┐     gRPC      ┌──────────────────┐
│  Streamlit   │─────────────►│  Servicio CV      │
│  (Frontend)  │              │  MobileNetV2      │
│  :8501       │              │  :50051           │
│              │     gRPC     ├──────────────────┤
│              │─────────────►│  Servicio NLP     │
│              │              │  GPT-5 Nano       │
└─────────────┘              │  :50052           │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │  MLflow Server    │
                              │  Tracking & UI    │
                              │  :5000            │
                              └──────────────────┘
```

**4 servicios:**

| Servicio | Puerto | Tecnología | Responsabilidad |
|----------|--------|------------|----------------|
| **App (Streamlit)** | 8501 | Streamlit + gRPC client | Interfaz web wizard de 4 pasos |
| **CV Server** | 50051 | gRPC + MobileNetV2 | Clasificación de enfermedades foliares |
| **NLP Server** | 50052 | gRPC + OpenAI GPT-5 Nano | Recomendaciones agronómicas |
| **MLflow** | 5000 | MLflow Tracking Server | Registro de experimentos y métricas |

---

## Tecnologías

- **Python 3.11** — Lenguaje principal
- **UV** — Gestor de paquetes y entornos virtuales
- **gRPC + Protobuf** — Comunicación entre servicios
- **Streamlit** — Interfaz web
- **PyTorch + Transformers** — Modelo de visión computacional
- **MobileNetV2 (PlantVillage)** — Clasificador de enfermedades (38 clases, 99.7% accuracy)
- **OpenAI GPT-5 Nano** — Generación de recomendaciones agronómicas
- **MLflow** — Tracking de experimentos
- **Docker + Docker Compose** — Contenedorización

---

## Estructura del Proyecto

```
PoC_DesProyecIA/
├── src/
│   ├── api/                    # Servidores gRPC
│   │   ├── protos/             # Archivos .proto y código generado
│   │   │   ├── vision.proto    # Contrato gRPC del servicio CV
│   │   │   ├── nlp.proto       # Contrato gRPC del servicio NLP
│   │   │   ├── vision_pb2.py
│   │   │   ├── vision_pb2_grpc.py
│   │   │   ├── nlp_pb2.py
│   │   │   └── nlp_pb2_grpc.py
│   │   ├── cv_server.py        # Servidor gRPC de Visión Computacional
│   │   └── nlp_server.py       # Servidor gRPC de NLP
│   ├── app/                    # Interfaz Streamlit
│   │   ├── app.py              # Wizard principal (4 pasos)
│   │   ├── config.py           # Variables de entorno y constantes
│   │   ├── grpc_client.py      # Cliente gRPC (conecta con CV y NLP)
│   │   ├── styles.py           # Tema CSS personalizado
│   │   └── utils.py            # Validación y componentes visuales
│   ├── nlp/                    # Módulo NLP
│   │   ├── openai_client.py    # Cliente OpenAI con reintentos
│   │   ├── prompt_builder.py   # Constructor de prompts agronómicos
│   │   └── mlflow_tracker.py   # Tracking MLflow para NLP
│   └── vision/                 # Módulo Visión Computacional
│       ├── classifier.py       # Pipeline de clasificación
│       ├── model_loader.py     # Carga del modelo MobileNetV2
│       ├── preprocessor.py     # Preprocesamiento de imágenes
│       └── mlflow_tracker.py   # Tracking MLflow para CV
├── scripts/
│   ├── mlflow_server.py        # Servidor MLflow
│   └── register_model.py       # Registro del modelo en MLflow
├── tests/                      # Tests unitarios
├── data/                       # Datos (raw, processed, external)
├── docs/                       # Documentación adicional
├── Dockerfile                  # Imagen Docker optimizada
├── docker-compose.yml          # Orquestación de 4 servicios
├── .dockerignore               # Exclusiones del build context
├── pyproject.toml              # Dependencias del proyecto
├── uv.lock                     # Lock de versiones exactas
├── Makefile                    # Comandos rápidos
├── .env.example                # Plantilla de variables de entorno
└── README.md
```

---

## Prerrequisitos

### Ejecución Local

- **Python 3.11** — [python.org](https://www.python.org/downloads/)
- **UV** — Gestor de paquetes: [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- **API Key de OpenAI** — [platform.openai.com](https://platform.openai.com/api-keys)

### Ejecución con Docker

- **Docker Desktop** — [docker.com](https://www.docker.com/products/docker-desktop/)
- **API Key de OpenAI** — Misma que para local

---

## Instalación y Ejecución Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/Jhonatand26/PoC_DesProyecIA.git
cd PoC_DesProyecIA
```

### 2. Instalar dependencias

```bash
uv sync
```

Esto crea el entorno virtual `.venv/` e instala todas las dependencias desde `uv.lock`.

### 3. Configurar variables de entorno

```bash
# Copiar la plantilla
cp .env.example .env

# Editar .env y agregar tu OPENAI_API_KEY
```

Abrir `.env` y configurar:

```env
APP_ENV=production
OPENAI_API_KEY=tu_api_key_aqui
```

### 4. Levantar los servicios (4 terminales)

**Terminal 1 — MLflow:**
```bash
make mlflow
```

**Terminal 2 — Servicio CV (Visión Computacional):**
```bash
make visback
```
> La primera ejecución descarga el modelo MobileNetV2 (~50MB). Las siguientes usan el cache local `hf_cache/`.

**Terminal 3 — Servicio NLP:**
```bash
make nlpback
```

**Terminal 4 — Interfaz Streamlit:**
```bash
make front
```

### 5. Abrir en el navegador

- **Aplicación:** [http://localhost:8501](http://localhost:8501)
- **MLflow UI:** [http://localhost:5000](http://localhost:5000)

---

## Ejecución con Docker

### 1. Clonar y configurar

```bash
git clone https://github.com/Jhonatand26/PoC_DesProyecIA.git
cd PoC_DesProyecIA
cp .env.example .env
# Editar .env y agregar OPENAI_API_KEY
```

### 2. Levantar todos los servicios

```bash
docker compose up --build
```

Esto construye la imagen y levanta los 4 servicios automáticamente. La primera vez tarda varios minutos por la descarga de dependencias (PyTorch, Transformers, etc.). Las siguientes ejecuciones usan cache de Docker.

### 3. Abrir en el navegador

- **Aplicación:** [http://localhost:8501](http://localhost:8501)
- **MLflow UI:** [http://localhost:5000](http://localhost:5000)

### 4. Detener servicios

```bash
docker compose down
```

### Notas sobre Docker

- Los servicios gRPC (CV y NLP) usan puertos **50051** y **50052**. Si tienes procesos locales en esos puertos, deténlos antes de levantar Docker.
- El modelo de HuggingFace se cachea en un volumen Docker (`poc_hf_cache`) para no descargarlo en cada arranque.
- Los datos de MLflow persisten en volúmenes Docker (`poc_mlflow_data`, `poc_mlflow_artifacts`).

---

## Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|--------|
| `APP_ENV` | Entorno: `development` (stubs) o `production` (gRPC real) | `development` |
| `APP_PORT` | Puerto de Streamlit | `8501` |
| `CV_SERVICE_HOST` | Host del servicio CV | `localhost` |
| `CV_SERVICE_PORT` | Puerto del servicio CV | `50051` |
| `NLP_SERVICE_HOST` | Host del servicio NLP | `localhost` |
| `NLP_SERVICE_PORT` | Puerto del servicio NLP | `50052` |
| `OPENAI_API_KEY` | API key de OpenAI (requerida) | — |
| `HUGGINGFACE_TOKEN` | Token de HuggingFace (opcional) | — |
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow | `http://localhost:5000` |
| `MAX_FILE_SIZE_MB` | Tamaño máximo de imagen | `10` |
| `CONFIDENCE_THRESHOLD` | Umbral mínimo de confianza | `0.60` |

> **Importante:** En Docker, los hosts de los servicios se resuelven por nombre de contenedor (`cv`, `nlp`, `mlflow`). El `docker-compose.yml` ya configura esto automáticamente.

---

## Comandos Rápidos (Makefile)

| Comando | Descripción |
|---------|-------------|
| `make mlflow` | Inicia el servidor MLflow |
| `make visback` | Inicia el servidor gRPC de CV |
| `make nlpback` | Inicia el servidor gRPC de NLP |
| `make front` | Inicia la interfaz Streamlit |
| `make docker` | Levanta todo con Docker Compose |
| `make docker-down` | Detiene los contenedores Docker |
| `make arbol` | Muestra el árbol de archivos |

---

## Tests

```bash
uv run pytest
```

---

## Equipo

| Integrante | Rol |
|-----------|-----|
| **Jhonatan David Rengifo** | Backend & Arquitectura |
| **Jorge Luis** | Visión Computacional |
| **Mateo** | NLP & Gemini |
| **Nicolás Vásquez Renjifo** | Frontend & QA |

---

## Licencia

Proyecto académico — Universidad Autónoma de Occidente, 2026.
