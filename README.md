# Système RAG pour Événements Culturels

Ce projet est un POC (Proof of Concept) d'un chatbot intelligent capable de répondre aux questions des utilisateurs sur les événements culturels. Il utilise un système de génération augmentée par récupération (RAG - Retrieval-Augmented Generation) combinant recherche vectorielle et génération de réponse en langage naturel.

## Fonctionnalités

- **Recherche Vectorielle** : Utilise FAISS et des embeddings de Sentence-Transformers pour trouver les informations pertinentes.
- **Génération de Réponses** : Intégration avec Mistral AI pour générer des réponses naturelles basées sur le contexte récupéré.
- **Framework** : Développé avec LangChain pour orchestrer la chaîne RAG.

## Prérequis

- Python >= 3.14
- [uv](https://github.com/astral-sh/uv) installé sur votre machine.

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

Le projet utilise `python-dotenv` pour gérer les variables d'environnement. Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Mistral :

```env
MISTRAL_API_KEY=votre_cle_api_ici
```

## Utilisation

### Lancer l'application

Pour exécuter le point d'entrée principal :

```bash
uv run python main.py
```

### Exécuter les tests

Pour vérifier l'installation et les imports :

```bash
uv run pytest
```

## Structure du Projet

- `main.py` : Point d'entrée de l'application.
- `pyproject.toml` : Configuration du projet et dépendances.
- `tests/` : Dossier contenant les tests unitaires.
- `uv.lock` : Fichier de verrouillage des dépendances.
