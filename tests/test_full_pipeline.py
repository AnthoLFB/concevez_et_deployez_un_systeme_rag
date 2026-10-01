from fastapi.testclient import TestClient
from src.api import app
import os
import time

def test_full_pipeline():
    print("--- Vérification du Pipeline Complet ---")
    with TestClient(app) as client:
        # 1. Rebuild
        print("Lancement de /rebuild...")
        start_time = time.time()
        response = client.post("/rebuild")
        duration = time.time() - start_time
        print(f"Statut /rebuild : {response.status_code}")
        print(f"Réponse : {response.json()}")
        print(f"Durée : {duration:.2f}s")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # 2. Ask
        question = "Quels événements y a-t-il à Lille ?"
        print(f"\nPose d'une question : '{question}'")
        response = client.post("/ask", json={"question": question})
        print(f"Statut /ask : {response.status_code}")
        if response.status_code == 200:
            print(f"Réponse : {response.json()['answer']}")
            assert "answer" in response.json()
        else:
            print(f"Erreur : {response.text}")
            assert False

if __name__ == "__main__":
    try:
        test_full_pipeline()
        print("\nPipeline validé avec succès !")
    except Exception as e:
        print(f"\nÉchec de la validation : {e}")
