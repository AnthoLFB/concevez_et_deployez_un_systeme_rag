import os
import pandas as pd
from src.data_ingestion import fetch_openagenda_events, process_events
from src.vector_store import create_chunks, build_vector_store, save_vector_store

def main():
    print("Démarrage du pipeline RAG (Ingestion -> Indexation FAISS)...")
    
    # Ingestion des données
    print("Récupération des événements depuis OpenAgenda pour Lille en 2026...")
    events = fetch_openagenda_events()
    print(f"{len(events)} événements récupérés.")
    
    if not events:
        print("Aucun événement trouvé. Fin du programme.")
        return

    # Nettoyage et structuration
    print("Nettoyage et structuration des données...")
    df = process_events(events)
    
    # Découpage en chunks
    print("Découpage des descriptions en chunks...")
    documents = create_chunks(df)
    print(f"{len(documents)} chunks créés.")
    
    # Vectorisation et Indexation FAISS
    print("Vectorisation et création de l'index FAISS (via Mistral Embeddings)...")
    vector_store = build_vector_store(documents)
    
    # Sauvegarde de l'index
    save_vector_store(vector_store)
    
    # Sauvegarde également du DataFrame traité pour référence
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_pickle(os.path.join(output_dir, "processed_events.pkl"))
    
    print("Pipeline terminé avec succès.")

if __name__ == "__main__":
    main()
