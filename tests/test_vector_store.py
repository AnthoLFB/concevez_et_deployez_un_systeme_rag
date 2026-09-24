import pytest
import pandas as pd
from src.vector_store import create_chunks
from langchain_core.documents import Document

def test_create_chunks_logic():
    """
    Vérifie que le découpage en chunks fonctionne correctement et conserve les métadonnées.
    """
    # Données de test
    data = {
        'uid': [123],
        'title_fr': ['Evénement Test'],
        'location_name': ['Lille Grand Palais'],
        'location_address': ['1 Boulevard des Cités Unies, 59777 Lille'],
        'firstdate_begin': ['2026-10-01'],
        'lastdate_end': ['2026-10-02'],
        'full_description': ['Ceci est une description de test qui est assez longue pour être découpée si on réduit la taille des chunks.']
    }
    df = pd.DataFrame(data)
    
    # On appelle la fonction (utilise les valeurs par défaut du .env ou les défauts du code)
    documents = create_chunks(df)
    
    # Vérifications
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], Document)
    
    # Vérification des métadonnées
    doc = documents[0]
    assert doc.metadata['title'] == 'Evénement Test'
    assert doc.metadata['uid'] == 123
    assert doc.metadata['location'] == 'Lille Grand Palais'
    
    # Vérification du contenu
    assert 'description de test' in doc.page_content

def test_create_chunks_empty_df():
    """Vérifie le comportement avec un DataFrame vide."""
    df = pd.DataFrame()
    documents = create_chunks(df)
    assert documents == []

if __name__ == "__main__":
    pytest.main([__file__])
