import os
import pandas as pd
from src.data_ingestion import fetch_openagenda_events, process_events
from src.vector_store import create_chunks, build_vector_store, save_vector_store, load_vector_store
from src.chatbot import get_chatbot_chain, ask_chatbot

def main():
    print("--- Pipeline RAG & Chatbot ---")
    
    # Etape 1 : Indexation si nécessaire
    index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    if not os.path.exists(index_path):
        print("L'index FAISS n'existe pas. Lancement de l'ingestion...")
        # Ingestion des données
        events = fetch_openagenda_events()
        print(f"{len(events)} événements récupérés.")
        
        if not events:
            print("Aucun événement trouvé. Fin du programme.")
            return

        # Nettoyage et structuration
        df = process_events(events)
        
        # Découpage en chunks
        documents = create_chunks(df)
        print(f"{len(documents)} chunks créés.")
        
        # Vectorisation et Indexation FAISS
        vector_store = build_vector_store(documents)
        
        # Sauvegarde de l'index
        save_vector_store(vector_store)
        
        # Sauvegarde également du DataFrame traité pour référence
        df.to_pickle("data/processed_events.pkl")
        print("Indexation terminée.")
    else:
        print("Index FAISS trouvé. Passage au mode chatbot.")

    # Etape 2 : Mode Chatbot
    print("\nInitialisation du Chatbot RAG (Mistral AI)...")
    rag_chain = get_chatbot_chain()
    
    print("\nBienvenue dans le chatbot des événements culturels de Lille !")
    print("(Tapez 'exit' ou 'quitter' pour arrêter)")
    
    while True:
        query = input("\nVotre question : ")
        if query.lower() in ['exit', 'quitter']:
            break
        
        if not query.strip():
            continue
            
        print("Recherche en cours...")
        try:
            answer = ask_chatbot(query, rag_chain)
            print(f"\nRéponse : {answer}")
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse : {e}")

    print("\nMerci d'avoir utilisé le chatbot. Au revoir !")

if __name__ == "__main__":
    main()
