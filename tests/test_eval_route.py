from fastapi.testclient import TestClient
from src.api import app
from unittest.mock import patch, MagicMock
import pandas as pd

client = TestClient(app)

def test_evaluate_route():
    print("--- Test de la route /evaluate ---")
    
    # On mocke run_rag_evaluation pour éviter les appels LLM réels
    with patch('src.api.run_rag_evaluation') as mock_eval, \
         patch('src.api.rag_chain', new=MagicMock()):
        
        # Mock du résultat Ragas
        mock_result = MagicMock()
        mock_result.scores = [
            {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_recall": 0.7,
                "context_precision": 0.85
            }
        ]
        mock_result.to_pandas.return_value = pd.DataFrame([{
            "question": "Test ?",
            "answer": "Reponse",
            "contexts": ["Ctx"],
            "ground_truth": "Truth",
            "faithfulness": 0.9
        }])
        
        mock_eval.return_value = mock_result
        
        # Test 1: Évaluation par défaut
        print("Test éval par défaut...")
        response = client.post("/evaluate")
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
