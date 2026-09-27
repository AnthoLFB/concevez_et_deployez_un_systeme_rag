import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_mistralai import ChatMistralAI
import logging

logger = logging.getLogger(__name__)

DEFAULT_TEST_DATA = [
    {
        "question": "Quels sont les événements musicaux prévus à Lille ?",
        "ground_truth": "Il y a plusieurs concerts à Lille, notamment au Zénith et à l'Aéronef."
    },
    {
        "question": "Où se déroule l'exposition sur l'art contemporain ?",
        "ground_truth": "L'exposition d'art contemporain se tient au Palais des Beaux-Arts de Lille."
    },
    {
        "question": "Y a-t-il des activités pour enfants ce weekend ?",
        "ground_truth": "Oui, il y a des ateliers créatifs à la Gare Saint Sauveur pour les familles."
    }
]

def run_rag_evaluation(rag_chain, test_data=None):
    """
    Exécute l'évaluation du système RAG.
    """
    if test_data is None:
        test_data = DEFAULT_TEST_DATA
        
    logger.info(f"Lancement de l'évaluation sur {len(test_data)} questions.")
    
    # 1. Collecte des réponses du système
    questions = [item["question"] for item in test_data]
    ground_truths = [item["ground_truth"] for item in test_data]
    answers = []
    contexts = []
    
    for q in questions:
        logger.info(f"Traitement de la question : {q}")
        response = rag_chain.invoke({"input": q})
        answers.append(response["answer"])
        contexts.append([doc.page_content for doc in response["context"]])
        
    # 2. Création du dataset Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    # 3. Configuration du modèle d'évaluation
    eval_llm = ChatMistralAI(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model=os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    )
    
    # 4. Exécution de l'évaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=eval_llm
    )
    
    return result
