import requests
import time
import sys
import pandas as pd
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_api():
    print("--- Test de l'API Puls-Events RAG (TestClient) ---")
    
    # Test de la racine
    response = client.get("/")
    print(f"GET / : {response.status_code} - {response.json()}")
    assert response.status_code == 200

    # Test de /ask
    question = "Quels sont les événements culturels à Lille ?"
    print(f"\nPose d'une question : '{question}'")
    
    # Note: L'initialisation du RAG se fait au startup, TestClient le gère avec 'with'
    with TestClient(app) as client_startup:
        response = client_startup.post(
            "/ask",
            json={"question": question}
        )
        if response.status_code == 200:
            print(f"Réponse reçue (200 OK)")
            print(f"Chatbot : {response.json()['answer'][:200]}...")
            assert "answer" in response.json()
        elif response.status_code == 503:
            print("Système RAG non initialisé (attendu si pas d'index FAISS)")
            assert True
        else:
            print(f"Erreur /ask : {response.status_code} - {response.text}")
            assert False, f"Erreur inattendue : {response.status_code}"

    # Test de /ask avec une question vide
    print("\nTest avec une question vide...")
    response = client.post(
        "/ask",
        json={"question": ""}
    )
    print(f"Statut attendu 400 : {response.status_code}")
    assert response.status_code == 400

    # Test de /rebuild (Mocké)
    from unittest.mock import patch, MagicMock
    print("\nTest de /rebuild...")
    with patch('src.api.fetch_openagenda_events') as mock_fetch, \
         patch('src.api.process_events') as mock_process, \
         patch('src.api.create_chunks') as mock_chunks, \
         patch('src.api.build_vector_store') as mock_vs, \
         patch('src.api.save_vector_store') as mock_save, \
         patch('src.api.get_chatbot_chain') as mock_chain:
        
        mock_fetch.return_value = [{"uid": 1, "title_fr": "Event"}]
        mock_process.return_value = pd.DataFrame([{"uid": 1, "title_fr": "Event", "full_description": "Test Event Content"}])
        mock_chunks.return_value = [MagicMock()]
        mock_vs.return_value = MagicMock()
        mock_chain.return_value = MagicMock()
        
        response = client.post("/rebuild")
        print(f"POST /rebuild : {response.status_code} - {response.json()}")
        assert response.status_code == 200
        assert "success" in response.json()["status"]

if __name__ == "__main__":
    test_api()
