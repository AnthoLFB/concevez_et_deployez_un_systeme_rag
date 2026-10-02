"""
Package `src` du système RAG Puls-Events.

Le chargement du fichier `.env` est centralisé ici afin que tous les
modules du package puissent lire leurs variables d'environnement
sans dupliquer d'appels à `load_dotenv()`.
"""

from dotenv import load_dotenv

load_dotenv()
