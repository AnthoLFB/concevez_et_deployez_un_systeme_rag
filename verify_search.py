import os
from dotenv import load_dotenv
from src.vector_store import load_vector_store, search_events

# Chargement des variables d'environnement
load_dotenv()

def test_search():
    """
    Script simple pour vérifier l'efficacité de la recherche sémantique.
    """
    index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    
    # Vérification si l'index existe
    if not os.path.exists(index_path):
        print(f"Erreur : L'index FAISS n'a pas été trouvé à l'emplacement {index_path}")
        print("Veuillez d'abord exécuter 'python main.py' pour générer l'index.")
        return

    print(f"Chargement de l'index depuis {index_path}...")
    try:
        vector_store = load_vector_store(index_path)
    except Exception as e:
        print(f"Erreur lors du chargement de l'index : {e}")
        return

    # Liste de requêtes de test
    queries = [
        "concert de jazz ou de musique classique",
        "exposition d'art moderne",
        "activités pour enfants et famille",
        "théâtre et spectacle vivant à Lille"
    ]

    for query in queries:
        print(f"\n--- Recherche pour : '{query}' ---")
        results = search_events(query, vector_store, k=3)
        
        if not results:
            print("Aucun résultat trouvé.")
            continue
            
        for i, doc in enumerate(results):
            title = doc.metadata.get('title', 'Sans titre')
            loc = doc.metadata.get('location', 'Lieu inconnu')
            date = doc.metadata.get('start_date', 'Date inconnue')
            print(f"{i+1}. {title}")
            print(f"   Lieu : {loc}")
            print(f"   Date : {date}")
            # print(f"   Extrait : {doc.page_content[:150]}...")

if __name__ == "__main__":
    test_search()
