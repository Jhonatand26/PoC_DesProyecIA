# MLflow — Servidor de Tracking y Model Registry

**PoC Asistente Fitosanitario con IA | UAO 2026**

---

## ¿Qué es MLflow en este proyecto?

MLflow registra los experimentos del modelo MobileNetV2 (Servicio CV).
Jorge Luis lo usa para guardar métricas, parámetros y el modelo en el
Model Registry. Jhonatan levanta y mantiene el servidor.

```
[Jorge Luis — classifier.py] → registra → [Servidor MLflow] → UI en :5000
```

---

## Arranque local (Módulo 3)

### 1. Configurar `.env`

Asegúrate de tener estas variables en tu `.env`:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
MLFLOW_BACKEND_URI=./mlruns
MLFLOW_ARTIFACT_ROOT=./mlartifacts
```

### 2. Correr el servidor

```bash
uv run python mlflow_server.py
```

### 3. Verificar

Abre el navegador en: **http://localhost:5000**

Deberías ver la UI de MLflow con el experimento `plant-disease-classifier`.

---

## Arranque con Docker Compose (Módulo 3 y 4)

```bash
# Levantar todos los servicios (MLflow + CV + NLP + Streamlit)
docker compose up --build

# Solo MLflow
docker compose up mlflow

# Ver logs en tiempo real
docker compose logs -f mlflow

# Detener todo
docker compose down
```

La UI queda disponible en: **http://localhost:5000**

---

## Cómo conectarse al servidor — Para Jorge Luis

En tu `src/vision/classifier.py` o donde registres el modelo,
agrega al inicio:

```python
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("plant-disease-classifier")
```

### Registrar una corrida de experimento

```python
with mlflow.start_run(run_name="mobilenetv2-baseline"):

    # Loguear parámetros del modelo
    mlflow.log_param("model_name", "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
    mlflow.log_param("num_classes", 38)
    mlflow.log_param("framework", "transformers")

    # Loguear métricas
    mlflow.log_metric("accuracy", 0.954)

    # Registrar el modelo en el Model Registry
    mlflow.set_tag("task", "plant-disease-classification")
    mlflow.set_tag("dataset", "PlantVillage")
```

---

## Estructura de carpetas generada por MLflow

```
PoC_DesProyecIA/
├── mlruns/          ← experimentos (NO subir al repo — en .gitignore)
└── mlartifacts/     ← artefactos   (NO subir al repo — en .gitignore)
```

> **Importante:** `mlruns/` y `mlartifacts/` están en `.gitignore`.
> En Docker, se persisten como volúmenes nombrados (`poc_mlflow_data`, `poc_mlflow_artifacts`).

---

## Puertos del sistema completo

| Servicio  | Puerto | URL                    |
|-----------|--------|------------------------|
| MLflow UI | 5000   | http://localhost:5000  |
| CV gRPC   | 50051  | interno                |
| NLP gRPC  | 50052  | interno                |
| Streamlit | 8501   | http://localhost:8501  |

---

## Troubleshooting

**MLflow no arranca:**
```bash
# Verificar que mlflow está instalado
uv run mlflow --version

# Si no está, sincronizar dependencias
uv sync
```

**Jorge Luis no puede conectarse al servidor:**
```bash
# Verificar que el servidor está corriendo
curl http://localhost:5000/health

# Verificar la variable de entorno
echo $MLFLOW_TRACKING_URI
```

**En Docker, los servicios no se ven entre sí:**
```bash
# Verificar que todos están en la misma red
docker network inspect poc_network
```