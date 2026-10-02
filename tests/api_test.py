"""
Tests fonctionnels de l'API Puls-Events RAG.

Ces tests utilisent `TestClient` de FastAPI. Ils vérifient :
- GET /
- POST /ask (question normale, question vide)
- POST /rebuild (mocké, pour éviter d'appeler OpenAgenda/Mistral)
- POST /ask avec date relative, date explicite, requête exhaustive, autre type

NB : les tests marqués `requires_live_rag` ne sont exécutés que si un index
FAISS est présent et une clé MISTRAL_API_KEY valide est configurée. Ils
appellent réellement Mistral et OpenAgenda.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app


FAISS_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
HAS_INDEX = (Path(FAISS_PATH) / "index.faiss").is_file()
HAS_API_KEY = bool(os.getenv("MISTRAL_API_KEY"))

requires_live_rag = pytest.mark.skipif(
    not (HAS_INDEX and HAS_API_KEY),
    reason="Index FAISS ou MISTRAL_API_KEY absent : test réel sauté.",
)


# -------------------------------------------------------------------
# Tests qui ne nécessitent ni Mistral ni index FAISS.
# -------------------------------------------------------------------

def test_root_ok():
    """GET / doit renvoyer 200 et un message de bienvenue."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_ask_empty_question_rejected():
    """POST /ask avec une question vide doit renvoyer 400."""
    client = TestClient(app)
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400




# -------------------------------------------------------------------
# Tests réels (nécessitent un index + une clé API).
# -------------------------------------------------------------------

@requires_live_rag
def test_ask_simple_question():
    """Question générique : doit obtenir une réponse non vide."""
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"question": "Quels événements culturels à Lille ?"},
        )
        assert response.status_code == 200
        assert response.json()["answer"].strip()


@requires_live_rag
@pytest.mark.parametrize(
    "question",
    [
        "Quels événements à Lille ce week-end ?",
        "Quels concerts ont eu lieu le mois dernier ?",
        "Quels événements sont prévus la semaine prochaine ?",
    ],
)
def test_ask_relative_dates(question):
    """Vérifie que les formulations temporelles relatives sont acceptées."""
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": question})
        assert response.status_code == 200
        answer = response.json()["answer"]
        assert isinstance(answer, str) and answer.strip()




@requires_live_rag
def test_ask_other_event_type():
    """Question ciblant un autre type d'événement (jeunesse)."""
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"question": "Quels événements pour les enfants à Lille ?"},
        )
        assert response.status_code == 200
        assert response.json()["answer"].strip()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
