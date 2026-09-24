# Système RAG pour Événements Culturels

Ce projet est un POC (Proof of Concept) d'un chatbot intelligent capable de répondre aux questions des utilisateurs sur les événements culturels. Il utilise un système de génération augmentée par récupération (RAG - Retrieval-Augmented Generation) combinant recherche vectorielle et génération de réponse en langage naturel.

## Fonctionnalités

- **Recherche Vectorielle** : Utilise **FAISS** et les embeddings officiels de **Mistral AI** pour trouver les informations pertinentes par similarité sémantique.
- **Génération de Réponses** : Intégration avec Mistral AI pour générer des réponses naturelles basées sur le contexte récupéré.
- **Chatbot Interactif** : Interface en ligne de commande pour poser des questions en langage naturel sur les événements.
- **Framework** : Développé avec **LangChain** pour orchestrer le pipeline RAG (Chunking, Vectorisation, Indexation, Retrieval).

## Prérequis

- Python >= 3.14
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

## Configuration

Le projet utilise `python-dotenv` pour gérer les variables d'environnement. Créez un fichier `.env` à la racine du projet (en vous basant sur `.env.example`) et configurez les variables suivantes :

```env
# Clé API Mistral
MISTRAL_API_KEY=votre_cle_api_ici
MISTRAL_MODEL=mistral-tiny

# Paramètres de filtrage
CITY=Lille
HISTORY_YEARS=1

# Paramètres Vector Database
FAISS_INDEX_PATH=data/faiss_index
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## Utilisation

### Lancer le chatbot (Pipeline & Interface)
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

## Structure du Projet

- `main.py` : Point d'entrée principal (Indexation + Chatbot).
- `verify_search.py` : Script utilitaire pour tester la recherche dans l'index.
- `src/` :
    - `data_ingestion.py` : Récupération et nettoyage des données OpenAgenda.
    - `vector_store.py` : Gestion du chunking, de la vectorisation Mistral et de l'index FAISS.
    - `chatbot.py` : Logique de la chaîne RAG et interaction avec Mistral AI via LangChain.
    - `data_processing.py` : Fonctions utilitaires pour les embeddings (legacy).
- `data/` : Dossier contenant l'index FAISS et le cache des événements (géré automatiquement).
- `tests/` : Tests unitaires (validation du chunking, du chatbot, etc.).
- `pyproject.toml` : Configuration et dépendances.
