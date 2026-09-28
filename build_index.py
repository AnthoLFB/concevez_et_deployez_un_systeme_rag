import os
import logging
from src.data_ingestion import fetch_openagenda_events, process_events
from src.vector_store import create_chunks, build_vector_store, save_vector_store

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_index():
    """
    Construit l'index vectoriel initial à partir des données OpenAgenda.
    """
    try:
        logger.info("Début de la construction de l'index vectoriel...")
        
        # Récupération des données depuis l'API
        events = fetch_openagenda_events()
        if not events:
            logger.warning("Aucun événement trouvé à indexer.")
            return
            
        # Prétraitement des données
        df = process_events(events)
        logger.info(f"{len(df)} événements traités.")
        
        # Création des documents/chunks
        docs = create_chunks(df)
        logger.info(f"{len(docs)} chunks créés.")
        
        # Construction du vector store FAISS
        vs = build_vector_store(docs)
        
        # Sauvegarde de l'index sur le disque
        save_vector_store(vs)
        
        logger.info("Index vectoriel construit et sauvegardé avec succès.")
        
    except Exception as e:
        logger.error(f"Erreur lors de la construction de l'index : {e}")
        raise

if __name__ == "__main__":
    build_index()
