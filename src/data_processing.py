import os
import pandas as pd
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

def get_mistral_client():
    api_key = os.getenv("MISTRAL_API_KEY")
    return Mistral(api_key=api_key)

def generate_embeddings(texts, model="mistral-embed"):
    """
    Génère des vecteurs pour une liste de textes en utilisant l'API Mistral.
    Note: MISTRAL_MODEL dans le .env est mistral-tiny (pour le chat), 
    mais pour les embeddings on utilise généralement mistral-embed.
    """
    client = get_mistral_client()
    
    # Mistral API supporte le batching
    response = client.embeddings.create(
        model=model,
        inputs=texts
    )
    
    return [e.embedding for e in response.data]

def add_embeddings_to_df(df, text_column='full_description'):
    """
    Ajoute une colonne 'embedding' au DataFrame.
    """
    if df.empty:
        return df
        
    texts = df[text_column].tolist()
    # On pourrait faire du batching ici si la liste est très longue
    # L'API Mistral a des limites de taille par requête.
    embeddings = generate_embeddings(texts)
    df['embedding'] = embeddings
    return df

if __name__ == "__main__":
    from data_ingestion import fetch_openagenda_events, process_events
    
    events = fetch_openagenda_events()
    if events:
        df = process_events(events[:5]) # Test sur 5 événements
        df = add_embeddings_to_df(df)
        print(df[['title_fr', 'embedding']].head())
