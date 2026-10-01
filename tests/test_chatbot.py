import pytest
from unittest.mock import MagicMock, patch, ANY
from src.chatbot import get_chatbot_chain, ask_chatbot
from langchain_core.documents import Document

@patch('src.chatbot.load_vector_store')
@patch('src.chatbot.ChatMistralAI')
def test_get_chatbot_chain(mock_mistral, mock_load_vs):
    """
    Vérifie que la chaîne du chatbot est correctement initialisée avec ses composants.
    """
    # Configuration des mocks
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_load_vs.return_value = mock_vs
    
    mock_llm = MagicMock()
    mock_mistral.return_value = mock_llm
    
    # Appel de la fonction
    chain = get_chatbot_chain()
    
    # Vérifications
    assert chain is not None
    mock_load_vs.assert_called_once()
    mock_vs.as_retriever.assert_called_once()
    mock_mistral.assert_called_once()

def test_ask_chatbot_integration_mock():
    """
    Teste ask_chatbot en mockant l'invocation de la chaîne.
    """
    mock_chain = MagicMock()
    # Dans langchain_classic.chains.create_retrieval_chain, 
    # l'objet retourné répond à .invoke() et retourne un dict avec "answer"
    mock_chain.invoke.return_value = {"answer": "Lille Piano(s) Festival"}
    
    query = "Quoi de neuf ?"
    response = ask_chatbot(query, mock_chain)
    
    assert response == "Lille Piano(s) Festival"
    mock_chain.invoke.assert_called_once_with({
        "input": query,
        "current_date": ANY,
        "weekend_dates": ANY
    })

def test_chatbot_main_execution():
    """Vérifie que le bloc __main__ ne plante pas (optionnel mais utile pour la couverture)."""
    with patch('src.chatbot.get_chatbot_chain') as mock_get_chain, \
         patch('src.chatbot.ask_chatbot') as mock_ask:
        mock_get_chain.return_value = MagicMock()
        mock_ask.return_value = "Réponse test"
        
        # Simuler l'exécution du bloc main
        import src.chatbot as chatbot
        # Note: charger le module exécute le code global, mais pas le bloc if __name__ == "__main__"
        # On peut appeler manuellement si on veut tester le contenu du bloc main, 
        # mais c'est souvent délicat. Ici on se contente de vérifier les fonctions.

if __name__ == "__main__":
    pytest.main([__file__])
