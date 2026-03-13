# =====================================================
# Dockerfile — PoC Asistente Fitosanitario con IA
# UAO 2026 — Desarrollo de Proyectos de IA
# =====================================================
# Imagen base: Python 3.11 slim (liviana)
# Gestor de paquetes: UV (copiado desde imagen oficial)
# Build context filtrado por .dockerignore
# =====================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="Jhonatan David Rengifo"
LABEL project="PoC Asistente Fitosanitario con IA — UAO 2026"

# Evita archivos .pyc y fuerza logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala curl (necesario para healthcheck de MLflow) y limpia cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copia UV desde la imagen oficial (mas rapido que pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia primero solo dependencias (aprovecha layer caching)
# Si pyproject.toml y uv.lock no cambian, Docker reutiliza esta capa
COPY pyproject.toml uv.lock .python-version ./

# Instala dependencias con UV
# --frozen: usa exactamente las versiones del uv.lock
# --no-install-project: solo dependencias, no el paquete local
RUN uv sync --frozen --no-install-project

# Copia el resto del codigo fuente (.dockerignore filtra lo innecesario)
COPY . .

# Instala el paquete local (src/) despues de copiar el codigo
RUN uv sync --frozen
