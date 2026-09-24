import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

# Chargement des variables d'environnement
load_dotenv()

def create_chunks(df, text_column='full_description'):
    """
    Découpe les textes du DataFrame en chunks plus petits pour la vectorisation.
    """
    # Récupération des paramètres de chunking depuis le .env
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    
    # Initialisation du splitter de texte
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    documents = []
    # Parcours du DataFrame pour créer des documents LangChain avec métadonnées
    for _, row in df.iterrows():
        # On s'assure que le texte n'est pas vide
        text = row[text_column]
        if not text or not isinstance(text, str):
            continue
            
        # Métadonnées à conserver pour chaque chunk
        metadata = {
            "uid": row.get("uid"),
            "title": row.get("title_fr"),
            "location": row.get("location_name"),
            "address": row.get("location_address"),
            "start_date": row.get("firstdate_begin"),
            "end_date": row.get("lastdate_end")
        }
        
        # Création des chunks pour ce document
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))
            
    return documents

def build_vector_store(documents):
    """
    Crée un index FAISS à partir d'une liste de documents.
    """
    # Utilisation des embeddings Mistral via l'API
    embeddings = MistralAIEmbeddings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-embed"
    )
    
    # Création de l'index vectoriel à partir des documents
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def save_vector_store(vector_store, path=None):
    """
    Sauvegarde l'index FAISS localement.
    """
    if path is None:
        path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
        
    # Création du dossier si il n'existe pas
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Sauvegarde locale de l'index
    vector_store.save_local(path)
    print(f"Index FAISS sauvegardé dans : {path}")

def load_vector_store(path=None):
    """
    Charge un index FAISS depuis le disque.
    """
    if path is None:
        path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
        
    # Chargement des embeddings Mistral
    embeddings = MistralAIEmbeddings(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-embed"
    )
    
    # Chargement de l'index avec désactivation de la désérialisation dangereuse
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def search_events(query, vector_store, k=4):
    """
    Effectue une recherche de similarité dans l'index.
    """
    # Recherche des k documents les plus proches sémantiquement
    results = vector_store.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    from data_ingestion import fetch_openagenda_events, process_events
    
    # Pipeline de test rapide
    print("Récupération des événements...")
    events = fetch_openagenda_events()
    if events:
        df = process_events(events)
        print(f"Nombre d'événements à indexer : {len(df)}")
        
        print("Création des chunks...")
        docs = create_chunks(df)
        
        print("Construction de l'index FAISS (ceci peut prendre un moment)...")
        vs = build_vector_store(docs)
        
        print("Sauvegarde de l'index...")
        save_vector_store(vs)
        
        # Test de recherche
        query = "concert de musique à Lille"
        print(f"\nTest de recherche pour : '{query}'")
        matches = search_events(query, vs)
        for i, doc in enumerate(matches):
            print(f"{i+1}. {doc.metadata.get('title')} ({doc.metadata.get('location')})")
    else:
        print("Aucun événement trouvé pour Lille en 2026.")
