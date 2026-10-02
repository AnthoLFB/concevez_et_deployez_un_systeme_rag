# Système RAG pour Événements Culturels

Ce projet est un POC (Proof of Concept) d'un chatbot intelligent capable de répondre aux questions des utilisateurs sur les événements culturels. Il utilise un système de génération augmentée par récupération (RAG - Retrieval-Augmented Generation) combinant recherche vectorielle et génération de réponse en langage naturel.

## Fonctionnalités

- **Recherche Vectorielle** : Utilise **FAISS** et les embeddings officiels de **Mistral AI** pour trouver les informations pertinentes par similarité sémantique.
- **Génération de Réponses** : Intégration avec Mistral AI pour générer des réponses naturelles basées sur le contexte récupéré.
- **Chatbot Interactif** : Interface en ligne de commande pour poser des questions en langage naturel sur les événements.
- **Framework** : Développé avec **LangChain** pour orchestrer le pipeline RAG (Chunking, Vectorisation, Indexation, Retrieval).

## Prérequis

- Python >= 3.12, < 3.14 (contrainte définie dans `pyproject.toml`).
- [uv](https://github.com/astral-sh/uv) installé sur votre machine.
- Une clé API Mistral AI valide.

## Fonctionnement du système

Le projet suit un pipeline de données en plusieurs étapes :
1. **Ingestion** : Les événements culturels sont récupérés depuis l'API OpenDataSoft filtrés par ville et par date.
2. **Traitement & Nettoyage** : Les données sont structurées dans un DataFrame Pandas et les descriptions sont nettoyées (suppression du HTML).
3. **Découpage (Chunking)** : Les descriptions longues sont découpées en morceaux plus petits (chunks) avec recouvrement pour conserver le contexte.
4. **Vectorisation & Indexation** : Chaque morceau de texte est transformé en vecteurs via l'API Mistral (`mistral-embed`) et stocké dans un index local **FAISS**.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <https://github.com/AnthoLFB/concevez_et_deployez_un_systeme_rag.git>
   cd concevez_et_deployez_un_systeme_rag
   ```

2. Installez les dépendances avec `uv` :
   ```bash
   uv sync
   ```
   `uv sync` installe les versions **exactes** verrouillées dans `uv.lock`,
   garantissant un environnement reproductible sur n'importe quelle machine.

## Configuration

Le projet utilise `python-dotenv` pour gérer les variables d'environnement. Créez un fichier `.env` à la racine du projet (en vous basant sur `.env.example`) et configurez les variables suivantes :

```env
# Clé API Mistral
MISTRAL_API_KEY=votre_cle_api_ici
# Modèle utilisé pour la génération de réponses (défaut : mistral-small-latest).
MISTRAL_MODEL=mistral-small-latest

# Paramètres de filtrage
CITY=Lille
HISTORY_YEARS=1

# Paramètres Vector Database
FAISS_INDEX_PATH=data/faiss_index
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## Utilisation

### Lancer l'API REST
Pour lancer l'API et accéder à la documentation interactive (Swagger) :
```bash
uv run python -m src.api
```
L'API sera accessible sur `http://127.0.0.1:8000`. La documentation est disponible sur `/docs`.

Points de terminaison principaux :
- `POST /ask` : Pose une question au chatbot.
- `POST /rebuild` : Déclenche en arrière-plan la récupération des données et la reconstruction de l'index FAISS. Répond immédiatement en `202 Accepted`.
- `GET /rebuild/status` : Retourne l'état de la dernière reconstruction (`idle`, `running`, `success`, `warning`, `error`).
- `POST /evaluate` : Lance une évaluation Ragas via l'API.

### Évaluer la qualité du RAG (Ragas)
Pour lancer l'évaluation automatique de la pertinence des réponses (utilise les modèles Mistral) :
```bash
uv run python evaluate_rag.py
```
L'évaluation calcule plusieurs métriques :
- **Faithfulness** : Fidélité de la réponse par rapport au contexte.
- **Answer Relevancy** : Pertinence de la réponse par rapport à la question.
- **Context Recall** : Capacité à retrouver les informations nécessaires.
- **Context Precision** : Précision du contexte récupéré.

Les résultats sont sauvegardés dans `rag_evaluation_results.csv`.

#### Jeu de vérités de terrain (ground truth)

Le script d'évaluation s'appuie sur un jeu de questions / `ground_truth`
situé par défaut dans `tests/ground_truth.json` (format : liste d'objets
`{"question": ..., "ground_truth": ...}`).

Ce fichier est généré automatiquement à partir des événements réellement
indexés (`data/processed_events.pkl`), de sorte que les réponses attendues
sont factuellement vérifiables :

```bash
uv run python scripts/generate_ground_truth.py
```

Le script produit ~75 questions variées (lieu, date, description) à partir
d'un tirage reproductible (`RANDOM_SEED = 42`).

Pour utiliser un autre fichier :

```bash
# PowerShell
$env:RAG_EVAL_DATASET = "chemin\vers\mon_dataset.json"
uv run python evaluate_rag.py
```

### Lancer le chatbot (Pipeline & Interface CLI)
Pour lancer le chatbot (indexation automatique au premier lancement, puis mode interactif) :
```bash
uv run python main.py
```
L'index sera sauvegardé dans `data/faiss_index/`. Si l'index existe déjà, le programme passera directement au mode discussion.

### Tester la recherche sémantique
Pour vérifier l'efficacité de l'indexation avec des requêtes de test :
```bash
uv run python verify_search.py
```

### Exécuter les tests
Pour valider la logique de traitement, du chatbot et du découpage :
```bash
uv run pytest
```

## Déploiement Docker

Le projet est entièrement conteneurisé. Deux artefacts sont fournis :

- `Dockerfile` : image multi-étapes basée sur `python:3.12-slim`, dépendances
  installées avec `uv sync --frozen` depuis `uv.lock`.
- `docker-compose.yml` : orchestration locale (API + service one-shot de
  construction de l'index).
- `scripts/build_index.py` : script Python autonome qui exécute le pipeline
  complet d'indexation (ingestion → chunking → vectorisation → sauvegarde FAISS).

### 1. Préparer le fichier `.env`
À la racine du projet (il est monté dans le conteneur via `env_file`) :
```env
MISTRAL_API_KEY=sk-...
OPENDATA_API_URL=https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records
CITY=Lille
HISTORY_YEARS=1
FAISS_INDEX_PATH=/app/data/faiss_index
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

### 2. Construire l'image
```bash
docker compose build
# ou directement :
docker build -t puls-events-rag:latest .
```

### 3. Construire l'index vectoriel (une fois)
L'index est persisté dans `./data` via un volume monté :
```bash
docker compose --profile tools run --rm build-index
```

### 4. Lancer l'API
```bash
docker compose up -d api
```
L'API est disponible sur [http://localhost:8000](http://localhost:8000) et la
documentation Swagger sur [http://localhost:8000/docs](http://localhost:8000/docs).

Pour arrêter :
```bash
docker compose down
```

### Alternative sans Docker Compose
```bash
docker run --rm -it \
  --env-file .env \
  -v ${PWD}/data:/app/data \
  -p 8000:8000 \
  puls-events-rag:latest
```

## Structure du Projet

- `src/` :
    - `__init__.py` : Chargement centralisé des variables d'environnement (`.env`).
    - `api.py` : API REST (FastAPI) exposant le système RAG.
    - `chatbot.py` : Logique de la chaîne RAG et interaction avec Mistral AI via LangChain.
    - `data_ingestion.py` : Récupération et nettoyage des données OpenAgenda.
    - `vector_store.py` : Gestion du chunking, de la vectorisation Mistral et de l'index FAISS.
    - `evaluation.py` : Pipeline d'évaluation Ragas (utilisé par `/evaluate` et `evaluate_rag.py`).
- `tests/` :
    - `api_test.py` : Tests fonctionnels de l'API.
    - `test_chatbot.py` : Tests unitaires de la chaîne RAG.
    - `test_data_ingestion.py` : Tests du nettoyage et de l'ingestion.
    - `test_vector_store.py` : Tests du chunking et de l'indexation FAISS.
    - `test_environment.py` : Vérifie que les imports critiques (FAISS, LangChain, Mistral) sont disponibles.
- `main.py` : Point d'entrée principal pour le mode CLI.
- `evaluate_rag.py` : Script d'évaluation des performances avec Ragas.
- `verify_search.py` : Script utilitaire pour tester la recherche dans l'index.
- `data/` : Dossier contenant l'index FAISS (géré automatiquement).
- `pyproject.toml` : Configuration et dépendances.

### À propos des imports LangChain/Mistral

Le brief pédagogique mentionne les imports historiques :
```python
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from mistral import MistralClient
```
Ces chemins ont été **dépréciés** par LangChain 0.3 et par le SDK Mistral.
Le projet utilise les équivalents modernes maintenus :
- `from langchain_community.vectorstores import FAISS`
- `from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI`

Un test (`tests/test_environment.py`) valide que ces imports fonctionnent.
