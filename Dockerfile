# syntax=docker/dockerfile:1.7

# ---------- Stage 1 : build des dépendances avec uv ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# uv est installé via l'image officielle (binaire statique).
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /usr/local/bin/uv

WORKDIR /app

# Dépendances système minimales (compilation éventuelle de faiss / numpy).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# On copie uniquement les fichiers de résolution pour tirer parti du cache.
COPY pyproject.toml uv.lock ./

# Installation déterministe dans un venv local au projet (.venv).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------- Stage 2 : image finale ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    FAISS_INDEX_PATH=/app/data/faiss_index

# Dépendances runtime : libgomp requis par faiss-cpu.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app -d /app app

WORKDIR /app

# venv prêt à l'emploi depuis le builder.
COPY --from=builder /app/.venv /app/.venv

# Code source du projet.
COPY src ./src
COPY scripts ./scripts
COPY main.py evaluate_rag.py verify_search.py ./
COPY pyproject.toml ./

# Dossier data par défaut (monté en volume en production).
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

# Healthcheck simple sur la racine de l'API.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3).status == 200 else sys.exit(1)" \
    || exit 1

# Démarrage de l'API FastAPI avec uvicorn.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
