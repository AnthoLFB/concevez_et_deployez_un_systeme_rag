import os
import pandas as pd
from src.data_ingestion import fetch_openagenda_events, process_events
from src.data_processing import add_embeddings_to_df

def main():
    print("Démarrage du pré-processing des données RAG...")
    
    # Récupération des données
    print("Récupération des événements depuis OpenAgenda...")
    events = fetch_openagenda_events()
    print(f"{len(events)} événements récupérés.")
    
    if not events:
        print("Aucun événement trouvé. Fin du programme.")
        return

    # Structuration et nettoyage
    print("Nettoyage et structuration des données...")
    df = process_events(events)
    
    # Vectorisation (Embeddings Mistral)
    print("Génération des vecteurs avec Mistral (ceci peut prendre quelques instants)...")
    # Pour le POC, on peut limiter le nombre d'événements à vectoriser si besoin
    # Ici on traite tout ce qui a été récupéré (limité à 100 dans ingestion)
    df = add_embeddings_to_df(df)
    
    # Sauvegarde des résultats
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "processed_events.pkl")
    df.to_pickle(output_file)
    print(f"Pré-processing terminé. Données sauvegardées dans {output_file}")
    print(df[['title_fr', 'location_name']].head())

if __name__ == "__main__":
    main()
