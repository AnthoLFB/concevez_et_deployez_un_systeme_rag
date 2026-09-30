from fastapi.testclient import TestClient
from src.api import app
from unittest.mock import patch, MagicMock
import pandas as pd

client = TestClient(app)

def test_evaluate_route():
    print("--- Test de la route /evaluate ---")
    
    # On mocke run_rag_evaluation pour éviter les appels LLM réels
    with patch('src.api.run_rag_evaluation') as mock_eval:
        app.state.rag_chain = MagicMock()
        
        # Le nouveau run_rag_evaluation retourne un dict
        mock_eval.return_value = {
            "scores": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_recall": 0.7,
                "context_precision": 0.85
            },
            "details": [{
                "question": "Test ?",
                "answer": "Reponse",
                "contexts": ["Ctx"],
                "ground_truth": "Truth",
                "faithfulness": 0.9
            }]
        }
        
        # Test 1: Évaluation par défaut
        print("Test éval avec données par défaut...")
        default_data = {
            "test_data": [
                {"question": "Quels sont les événements ?", "ground_truth": "Il y a des concerts."}
            ]
        }
        response = client.post("/evaluate", json=default_data)
        assert response.status_code == 200
        data = response.json()
        assert "scores" in data
        assert "details" in data
        assert data["scores"]["faithfulness"] == 0.9
        print("OK")
        
        # Test 2: Évaluation avec données personnalisées
        print("Test éval avec données personnalisées...")
        custom_data = {
            "test_data": [
                {"question": "Quelle heure est-il ?", "ground_truth": "Il est midi."}
            ]
        }
        response = client.post("/evaluate", json=custom_data)
        assert response.status_code == 200
        data = response.json()
        assert data["scores"]["answer_relevancy"] == 0.8
        print("OK")

if __name__ == "__main__":
    try:
        test_evaluate_route()
        print("\nTous les tests de la route /evaluate sont passés !")
    except Exception as e:
        print(f"\nÉchec du test : {e}")
        import traceback
        traceback.print_exc()
