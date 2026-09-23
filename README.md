# Système RAG pour Événements Culturels

Ce projet est un POC (Proof of Concept) d'un chatbot intelligent capable de répondre aux questions des utilisateurs sur les événements culturels. Il utilise un système de génération augmentée par récupération (RAG - Retrieval-Augmented Generation) combinant recherche vectorielle et génération de réponse en langage naturel.

## Fonctionnalités

- **Recherche Vectorielle** : Utilise FAISS et des embeddings de Sentence-Transformers pour trouver les informations pertinentes.
- **Génération de Réponses** : Intégration avec Mistral AI pour générer des réponses naturelles basées sur le contexte récupéré.
- **Framework** : Développé avec LangChain pour orchestrer la chaîne RAG.

## Prérequis

- Python >= 3.14
- [uv](https://github.com/astral-sh/uv) installé sur votre machine.

## Fonctionnement du système

Le projet suit un pipeline de données en plusieurs étapes :
1. **Ingestion** : Les événements culturels sont récupérés depuis l'API OpenDataSoft filtrés par ville et par date.
2. **Traitement & Nettoyage** : Les données sont structurées dans un DataFrame Pandas, les balises HTML sont supprimées et les descriptions sont normalisées.
3. **Vectorisation** : Le texte combiné (titre + descriptions) est transformé en vecteurs (embeddings) via l'API Mistral (`mistral-embed`).
4. **Stockage** : Le DataFrame résultant, incluant les vecteurs, est sauvegardé au format Pickle pour une utilisation ultérieure par le moteur de recherche FAISS.

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
# Clé API pour la génération de réponses et d'embeddings
MISTRAL_API_KEY=votre_cle_api_ici

# URL de l'API OpenDataSoft OpenAgenda
OPENDATA_API_URL=https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records

# Paramètres de filtrage des événements
CITY=Lille
HISTORY_YEARS=1
```

## Utilisation

### Lancer l'ingestion et le traitement
Pour récupérer les données depuis OpenAgenda, les nettoyer et générer les embeddings Mistral :
```bash
uv run python main.py
```
Les données traitées seront sauvegardées dans le dossier `data/processed_events.pkl`.

### Exécuter les tests
Pour vérifier l'installation et les fonctionnalités :
```bash
uv run pytest
```

## Structure du Projet

- `main.py` : Point d'entrée pour la chaîne d'ingestion et de traitement.
- `src/` :
    - `data_ingestion.py` : Récupération des données via l'API OpenDataSoft et premier nettoyage.
    - `data_processing.py` : Génération des embeddings en utilisant l'API Mistral AI.
- `data/` : Dossier contenant les données traitées (généré après exécution).
- `tests/` : Tests unitaires pour valider les composants.
- `pyproject.toml` : Configuration du projet et dépendances (géré par `uv`).
- `.env.example` : Modèle pour les variables d'environnement.
