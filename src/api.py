import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.chatbot import ask_chatbot, get_chatbot_chain
from src.data_ingestion import fetch_openagenda_events, process_events
from src.evaluation import run_rag_evaluation
from src.vector_store import build_vector_store, create_chunks, save_vector_store


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Le système RAG est partagé entre les requêtes.
# app.state permet d'éviter une variable globale dédiée à la chaîne.
REBUILD_LOCK = Lock()


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question en langage naturel posée au chatbot.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quels concerts ont lieu à Lille ce week-end ?"
            }
        }
    }


class AnswerResponse(BaseModel):
    question: str
    answer: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quels concerts ont lieu à Lille ce week-end ?",
                "answer": "**Concert jazz au Flow** le samedi 5 octobre 2024 à 20h00...",
            }
        }
    }


class EvalItem(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    ground_truth: str = Field(..., min_length=1)


class EvalRequest(BaseModel):
    test_data: list[EvalItem]

    model_config = {
        "json_schema_extra": {
            "example": {
                "test_data": [
                    {
                        "question": "Où se tient le festival Series Mania ?",
                        "ground_truth": "Le festival Series Mania se tient à Lille.",
                    }
                ]
            }
        }
    }


class EvalResponse(BaseModel):
    scores: dict[str, float]
    details: list[dict[str, Any]]


class RebuildStatusResponse(BaseModel):
    """État courant d'une reconstruction déclenchée via /rebuild."""
    state: str = Field(
        description="idle | running | success | warning | error",
    )
    message: str | None = None
    events_count: int | None = None
    documents_count: int | None = None


def vector_store_exists(path: str) -> bool:
    """Vérifie que l'index FAISS local semble complet."""
    index_dir = Path(path)

    return (
        index_dir.is_dir()
        and (index_dir / "index.faiss").is_file()
        and (index_dir / "index.pkl").is_file()
    )


def get_rag_chain(app: FastAPI):
    """Retourne la chaîne RAG actuellement chargée."""
    rag_chain = getattr(app.state, "rag_chain", None)

    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Le système RAG n'est pas initialisé. "
                "Veuillez reconstruire l'index avec /rebuild."
            ),
        )

    return rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise le système RAG avant que l'API accepte des requêtes.
    """
    app.state.rag_chain = None
    app.state.rebuild_status = {
        "state": "idle",
        "message": None,
        "events_count": None,
        "documents_count": None,
    }

    # Vérification des variables d'environnement critiques au démarrage
    required_vars = ["MISTRAL_API_KEY", "OPENDATA_API_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Variables d'environnement manquantes : %s", ", ".join(missing_vars))
        # On ne lève pas d'exception pour permettre à l'API de démarrer
        # et de laisser l'utilisateur configurer son env si besoin (via redémarrage).
    
    faiss_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

    logger.info("Initialisation du système RAG...")

    if not vector_store_exists(faiss_path):
        logger.warning(
            "Index FAISS introuvable dans '%s'. "
            "Le système devra être construit avec /rebuild.",
            faiss_path,
        )
        yield
        return

    try:
        app.state.rag_chain = get_chatbot_chain()
        logger.info("Système RAG prêt.")
    except Exception:
        # L'API peut tout de même démarrer afin de permettre un /rebuild.
        logger.exception("Impossible de charger le système RAG au démarrage.")
        app.state.rag_chain = None

    yield

    # Aucun nettoyage spécifique n'est nécessaire pour ce POC.
    app.state.rag_chain = None


class PrettyJSONResponse(JSONResponse):
    """
    Une classe de réponse personnalisée qui retourne du JSON indenté (pretty-print).
    """

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(",", ": "),
        ).encode("utf-8")


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API pour interroger un système RAG sur les événements culturels "
        "à Lille et dans les Hauts-de-France."
    ),
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=PrettyJSONResponse,
    swagger_ui_parameters={
        "defaultModelExpandDepth": 3,
        "defaultModelsExpandDepth": 3,
    },
)


@app.get("/")
def root():
    """Vérifie simplement que l'API est disponible."""
    return {
        "message": (
            "Bienvenue sur l'API Puls-Events RAG. "
            "Consultez /docs pour la documentation."
        )
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Pose une question au système RAG.

    - **question** : texte libre (1 à 2000 caractères).
    - **Réponse** : contient la question reçue et la réponse générée.
    """
    # Pydantic a déjà validé min_length=1 ; on protège uniquement contre
    # une chaîne composée uniquement d'espaces.
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide.",
        )

    rag_chain = get_rag_chain(app)

    try:
        answer = ask_chatbot(question, rag_chain)

        return AnswerResponse(
            question=question,
            answer=answer,
        )
    except Exception:
        logger.exception("Erreur lors du traitement de la question.")
        raise HTTPException(
            status_code=500,
            detail="Une erreur interne est survenue lors du traitement de la question.",
        )


def _run_rebuild(app: FastAPI) -> None:
    """
    Exécute la reconstruction complète de l'index FAISS.

    Cette fonction est destinée à être lancée dans un `BackgroundTasks`
    pour éviter de bloquer la requête HTTP /rebuild (opération longue :
    appels embeddings Mistral sur l'ensemble des chunks).
    """
    status = app.state.rebuild_status

    try:
        logger.info("Début de la reconstruction de la base vectorielle...")

        # 1. Ingestion
        events = fetch_openagenda_events()
        if not events:
            logger.warning("Aucun événement trouvé à indexer.")
            status.update(state="warning", message="Aucun événement trouvé à indexer.")
            return

        # 2. Traitement
        df = process_events(events)
        if df.empty:
            logger.warning("Aucun événement exploitable après traitement.")
            status.update(state="warning", message="Aucun événement exploitable après traitement.")
            return

        # 3. Chunking
        documents = create_chunks(df)
        if not documents:
            logger.warning("Aucun document créé pour l'indexation.")
            status.update(state="warning", message="Aucun document n'a pu être créé pour l'indexation.")
            return

        # 4. Construction de l'index + 5. Sauvegarde
        vector_store = build_vector_store(documents)
        save_vector_store(vector_store)

        # 6. Création de la nouvelle chaîne RAG. On remplace l'ancienne
        # uniquement si tout a réussi, pour conserver un service opérationnel
        # en cas d'échec en cours de route.
        app.state.rag_chain = get_chatbot_chain()

        logger.info(
            "Base vectorielle reconstruite : %d événements, %d documents.",
            len(df), len(documents),
        )
        status.update(
            state="success",
            message=f"Base reconstruite avec {len(df)} événements et {len(documents)} documents.",
            events_count=len(df),
            documents_count=len(documents),
        )

    except Exception as exc:
        logger.exception("Erreur lors de la reconstruction de la base vectorielle.")
        status.update(state="error", message=f"Erreur : {exc}")
    finally:
        REBUILD_LOCK.release()


@app.post("/rebuild", status_code=202)
def rebuild(background_tasks: BackgroundTasks):
    """
    Déclenche la reconstruction asynchrone de l'index FAISS.

    La réponse est renvoyée immédiatement (202 Accepted). L'avancement
    peut être suivi via `GET /rebuild/status`.
    """
    if not REBUILD_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Une reconstruction est déjà en cours.",
        )

    app.state.rebuild_status.update(
        state="running",
        message="Reconstruction en cours...",
        events_count=None,
        documents_count=None,
    )
    background_tasks.add_task(_run_rebuild, app)

    return {
        "status": "accepted",
        "message": "Reconstruction lancée en arrière-plan. Consultez /rebuild/status.",
    }


@app.get("/rebuild/status", response_model=RebuildStatusResponse)
def rebuild_status():
    """État courant de la dernière reconstruction déclenchée."""
    return RebuildStatusResponse(**app.state.rebuild_status)


@app.post("/evaluate", response_model=EvalResponse)
def evaluate_rag_api(request: EvalRequest):
    """
    Évalue le système RAG avec un jeu de questions/réponses de référence.
    """
    rag_chain = get_rag_chain(app)

    if not request.test_data:
        raise HTTPException(
            status_code=400,
            detail="Le jeu de données d'évaluation ne peut pas être vide.",
        )

    try:
        test_data = [item.model_dump() for item in request.test_data]

        result = run_rag_evaluation(rag_chain, test_data)

        return EvalResponse(
            scores=result["scores"],
            details=result["details"],
        )

    except ValueError as exc:
        logger.warning("Erreur de validation pendant l'évaluation : %s", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    except Exception:
        logger.exception("Erreur lors de l'évaluation Ragas.")
        raise HTTPException(
            status_code=500,
            detail="Une erreur interne est survenue lors de l'évaluation.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
