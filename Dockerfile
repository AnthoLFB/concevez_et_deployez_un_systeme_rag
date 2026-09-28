# Utilisation d'une image Python stable et légère
FROM python:3.12-slim

# Empêcher Python d'écrire des fichiers .pyc et forcer l'affichage des logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Mise à jour de pip
RUN pip install --no-cache-dir --upgrade pip

# Copie et installation des dépendances
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copie du code source et des fichiers nécessaires
COPY src/ ./src/
COPY data/ ./data/
COPY build_index.py .

# Création du dossier pour l'index FAISS
RUN mkdir -p data/faiss_index

# Exposition du port API
EXPOSE 8000

# Commande par défaut (sera surchargée par docker-compose si besoin)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
