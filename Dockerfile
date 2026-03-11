# =====================================================
# Dockerfile base — PoC Asistente Fitosanitario con IA
# UAO 2026 — Desarrollo de Proyectos de IA
# =====================================================
# Imagen base: Python 3.11 slim (liviana, sin extras innecesarios)
# Gestor de paquetes: UV
# =====================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="Jhonatan David Rengifo"
LABEL project="PoC Asistente Fitosanitario con IA — UAO 2026"

# Evita que Python genere archivos .pyc y bufferiza stdout/stderr
# para que los logs aparezcan en tiempo real en Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala UV
RUN pip install --no-cache-dir uv

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia primero solo las dependencias (aprovecha layer caching)
COPY pyproject.toml uv.lock ./

# Instala dependencias con UV
# --frozen garantiza que usa exactamente las versiones del uv.lock
RUN uv sync --frozen

# Copia el resto del código
COPY . .