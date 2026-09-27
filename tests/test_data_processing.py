import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data_processing import generate_embeddings, add_embeddings_to_df

@patch('src.data_processing.Mistral')
def test_generate_embeddings(mock_mistral):
    """Teste la génération d'embeddings avec un mock de l'API Mistral."""
    mock_client = MagicMock()
    mock_mistral.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6])
    ]
    mock_client.embeddings.create.return_value = mock_response
    
    texts = ["Texte 1", "Texte 2"]
    embeddings = generate_embeddings(texts)
    
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once()

@patch('src.data_processing.generate_embeddings')
def test_add_embeddings_to_df(mock_gen_emb):
    """Teste l'ajout de la colonne embedding au DataFrame."""
    mock_gen_emb.return_value = [[0.1, 0.1], [0.2, 0.2]]
    
    df = pd.DataFrame({
        'full_description': ["Desc 1", "Desc 2"]
    })
    
    df_result = add_embeddings_to_df(df)
    
    assert 'embedding' in df_result.columns
    assert len(df_result) == 2
    assert df_result.iloc[0]['embedding'] == [0.1, 0.1]
    mock_gen_emb.assert_called_once_with(["Desc 1", "Desc 2"])

def test_add_embeddings_to_empty_df():
    """Vérifie le comportement avec un DataFrame vide."""
    df = pd.DataFrame()
    df_result = add_embeddings_to_df(df)
    assert df_result.empty

if __name__ == "__main__":
    pytest.main([__file__])
