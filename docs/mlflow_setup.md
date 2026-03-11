# Configuración y Uso de MLflow

Guía para configurar el servidor MLflow local y acceder a la UI de seguimiento
de experimentos del modelo de clasificación de enfermedades de plantas.

## 1. Iniciar el servidor MLflow

Desde la raíz del proyecto:

```bash
uv run mlflow server --host 0.0.0.0 --port 5000
```

El servidor quedará escuchando en `http://localhost:5000`.

> **Nota:** Asegúrate de que la variable `MLFLOW_TRACKING_URI` en tu archivo
> `.env` apunte a `http://localhost:5000`.

## 2. Acceder a la UI de MLflow

Abre tu navegador en:

```
http://localhost:5000
```

Desde la UI podrás:

- Ver el experimento **`plant-disease-cv`** en la barra lateral
- Explorar los runs con métricas (`confidence`, `inference_time_ms`, `accuracy`, `f1_score`)
- Revisar los parámetros logueados (`model_name`, `model_version`, `model_source`)
- Acceder al **Model Registry** para ver el modelo registrado `mobilenet-plant-disease`

## 3. Registrar el modelo en Model Registry

Ejecutar el script de registro (una sola vez):

```bash
uv run python scripts/register_model.py
```

Esto creará un run en el experimento `plant-disease-cv` con:

| Tipo       | Clave            | Valor                                |
|------------|------------------|--------------------------------------|
| Parámetro  | `model_name`     | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` |
| Parámetro  | `model_version`  | `v1.0`                               |
| Parámetro  | `model_source`   | URL de HuggingFace                   |
| Métrica    | `accuracy`       | `0.9970`                             |
| Métrica    | `f1_score`       | `0.9932`                             |

El modelo quedará registrado en **Model Registry** como `mobilenet-plant-disease`.

## 4. Verificar el registro

1. Abre la UI → pestaña **Models** (o Model Registry)
2. Busca el modelo `mobilenet-plant-disease`
3. Verifica que la versión aparezca con sus métricas y parámetros asociados
