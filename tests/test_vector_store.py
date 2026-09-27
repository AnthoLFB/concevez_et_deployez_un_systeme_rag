import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.vector_store import create_chunks, build_vector_store, save_vector_store, load_vector_store, search_events
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

@patch('src.vector_store.FAISS')
@patch('src.vector_store.MistralAIEmbeddings')
def test_build_vector_store(mock_embeddings, mock_faiss):
    """Vérifie la construction du vector store."""
    docs = [Document(page_content="test", metadata={})]
    mock_vs = MagicMock()
    mock_faiss.from_documents.return_value = mock_vs
    
    vs = build_vector_store(docs)
    
    assert vs == mock_vs
    mock_faiss.from_documents.assert_called_once()

def test_save_vector_store(tmp_path):
    """Vérifie la sauvegarde du vector store."""
    mock_vs = MagicMock()
    save_path = str(tmp_path / "faiss_index")
    
    save_vector_store(mock_vs, path=save_path)
    
    mock_vs.save_local.assert_called_once_with(save_path)

@patch('src.vector_store.FAISS')
@patch('src.vector_store.MistralAIEmbeddings')
def test_load_vector_store(mock_embeddings, mock_faiss, tmp_path):
    """Vérifie le chargement du vector store."""
    mock_vs = MagicMock()
    mock_faiss.load_local.return_value = mock_vs
    save_path = str(tmp_path / "faiss_index")
    
    vs = load_vector_store(path=save_path)
    
    assert vs == mock_vs
    mock_faiss.load_local.assert_called_once()

def test_search_events():
    """Vérifie la fonction de recherche."""
    mock_vs = MagicMock()
    mock_results = [Document(page_content="match", metadata={})]
    mock_vs.similarity_search.return_value = mock_results
    
    results = search_events("query", mock_vs, k=2)
    
    assert results == mock_results
    mock_vs.similarity_search.assert_called_once_with("query", k=2)

if __name__ == "__main__":
    pytest.main([__file__])
