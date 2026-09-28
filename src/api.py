from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import logging
import asyncio
import nest_asyncio

# Patch nest_asyncio pour la compatibilité avec Python 3.12+ et uvicorn.
# nest_asyncio patche asyncio.run mais ne supporte pas l'argument loop_factory introduit dans les versions récentes de Python.
# Ce patch intercepte l'appel pour ignorer loop_factory s'il est absent du patch de nest_asyncio.
nest_asyncio.apply()
original_asyncio_run = asyncio.run

def patched_asyncio_run(main, *, debug=None, loop_factory=None):
    if loop_factory is None:
        return original_asyncio_run(main, debug=debug)
    # L'appel à original_asyncio_run ici utilise la version patchée par nest_asyncio.
    return original_asyncio_run(main, debug=debug)

asyncio.run = patched_asyncio_run

from contextlib import asynccontextmanager
from src.chatbot import get_chatbot_chain, ask_chatbot
from src.vector_store import build_vector_store, create_chunks, save_vector_store
from src.data_ingestion import fetch_openagenda_events, process_events
from src.evaluation import run_rag_evaluation

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variable globale pour stocker la chaîne RAG
rag_chain = None

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

class EvalItem(BaseModel):
    question: str
    ground_truth: str

class EvalRequest(BaseModel):
    test_data: Optional[List[EvalItem]] = None

class EvalResponse(BaseModel):
    scores: Dict[str, float]
    details: List[Dict]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application (remplace on_event startup/shutdown).
    """
    global rag_chain
    try:
        logger.info("Initialisation du système RAG...")
        # Vérifier si l'index existe avant de charger
        faiss_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
        if os.path.exists(faiss_path):
            rag_chain = get_chatbot_chain()
            logger.info("Système RAG prêt.")
        else:
            logger.warning("Index FAISS non trouvé. Le système RAG devra être reconstruit via /rebuild.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du RAG : {e}")
    
    yield
    # Code de nettoyage ici si besoin

app = FastAPI(
    title="Puls-Events RAG API",
    description="API pour interroger le chatbot intelligent sur les événements culturels à Lille.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """
    Point d'entrée simple pour vérifier que l'API est en ligne.
    """
    return {"message": "Bienvenue sur l'API Puls-Events RAG. Consultez /docs pour la documentation."}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    Pose une question au chatbot et reçoit une réponse augmentée par le contexte.
    """
    global rag_chain
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")
    
    if rag_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="Le système RAG n'est pas initialisé. Veuillez lancer /rebuild."
        )
    
    try:
        answer = ask_chatbot(request.question, rag_chain)
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la question : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild")
async def rebuild():
    """
    Reconstruit la base vectorielle en récupérant les dernières données.
    """
    global rag_chain
    try:
        logger.info("Début de la reconstruction de la base vectorielle...")
        
        # Ingestion
        events = fetch_openagenda_events()
        if not events:
            return {"status": "warning", "message": "Aucun événement trouvé à indexer."}
            
        # Processing
        df = process_events(events)
        
        # Vectorisation
        docs = create_chunks(df)
        vs = build_vector_store(docs)
        
        # Sauvegarde
        save_vector_store(vs)
        
        # Re-initialisation de la chaîne
        rag_chain = get_chatbot_chain()
        
        logger.info("Base vectorielle reconstruite avec succès.")
        return {"status": "success", "message": f"Base reconstruite avec {len(df)} événements."}
    except Exception as e:
        logger.error(f"Erreur lors de la reconstruction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_rag_api(request: Optional[EvalRequest] = None):
    """
    Évalue le système RAG à l'aide de Ragas.
    """
    global rag_chain
    if rag_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="Le système RAG n'est pas initialisé. Veuillez lancer /rebuild."
        )
    
    try:
        test_data = None
        if request and request.test_data:
            test_data = [item.model_dump() for item in request.test_data]
            
        result = run_rag_evaluation(rag_chain, test_data)
        
        # Préparation de la réponse
        # Calcul des moyennes pour Ragas 0.2+
        scores = {}
        if len(result.scores) > 0:
            metrics = result.scores[0].keys()
            for metric in metrics:
                values = [s[metric] for s in result.scores if metric in s and s[metric] is not None]
                if values:
                    scores[metric] = float(sum(values) / len(values))
        
        details = result.to_pandas().to_dict(orient="records")
        
        return EvalResponse(scores=scores, details=details)
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
