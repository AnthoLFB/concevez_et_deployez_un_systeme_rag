import requests
import json

def test_evaluation():
    url = "http://localhost:8000/evaluate"
    payload = {
        "test_data": [
            {
                "question": "Où se déroule le festival de jazz ?",
                "ground_truth": "Le festival de jazz se déroule à Lille."
            }
        ]
    }
    
    # On a besoin que l'API tourne.
    # Au lieu de ça, on va tester via TestClient dans un nouveau fichier de test
    # ou simplement lancer api_test.py s'il couvre tout.
    pass

if __name__ == "__main__":
    # Ce script est juste pour référence.
    pass
