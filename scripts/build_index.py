"""
Construit (ou reconstruit) l'index vectoriel FAISS en local.

Pipeline :
1. Récupération des événements via l'API OpenAgenda.
2. Nettoyage / structuration des données.
3. Découpage en chunks.
4. Vectorisation via Mistral + sauvegarde FAISS.

Utilisation :
    uv run python scripts/build_index.py
    # ou dans un conteneur Docker :
    python scripts/build_index.py

Variables d'environnement attendues :
    MISTRAL_API_KEY, OPENDATA_API_URL, FAISS_INDEX_PATH, CITY, HISTORY_YEARS,
    CHUNK_SIZE, CHUNK_OVERLAP (voir README).
"""
from __future__ import annotations

import logging
import os
import sys

# Permet d'exécuter le script depuis la racine du projet.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import fetch_openagenda_events, process_events  # noqa: E402
from src.vector_store import (  # noqa: E402
    build_vector_store,
    create_chunks,
    save_vector_store,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")


def main() -> int:
    required = ["MISTRAL_API_KEY", "OPENDATA_API_URL"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error("Variables manquantes : %s", ", ".join(missing))
        return 2

    logger.info("1/4 - Récupération des événements OpenAgenda...")
    events = fetch_openagenda_events()
    if not events:
        logger.error("Aucun événement récupéré. Abandon.")
        return 1

    logger.info("2/4 - Traitement et nettoyage (%d événements)...", len(events))
    df = process_events(events)
    if df.empty:
        logger.error("Aucun événement exploitable après traitement. Abandon.")
        return 1

    logger.info("3/4 - Découpage en chunks (%d événements)...", len(df))
    documents = create_chunks(df)
    if not documents:
        logger.error("Aucun document créé pour l'indexation. Abandon.")
        return 1

    logger.info("4/4 - Vectorisation Mistral + sauvegarde FAISS (%d chunks)...", len(documents))
    vector_store = build_vector_store(documents)
    save_vector_store(vector_store)

    faiss_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    logger.info("Index construit avec succès dans '%s'.", faiss_path)
    logger.info("Résumé : %d événements -> %d documents.", len(df), len(documents))
    return 0


if __name__ == "__main__":
    sys.exit(main())
