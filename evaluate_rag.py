import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from src.chatbot import get_chatbot_chain
from src.evaluation import run_rag_evaluation

# Chargement des variables d'environnement
load_dotenv()

def evaluate_rag():
    """
    Évalue les performances du système RAG à l'aide de Ragas.
    """
    print("--- Évaluation du système RAG avec Ragas ---")
    
    # Initialisation du chatbot
    print("Initialisation du chatbot pour génération des réponses...")
    rag_chain = get_chatbot_chain()
    
    # Exécution de l'évaluation
    print("\nLancement de l'évaluation Ragas (ceci peut prendre quelques minutes)...")
    result = run_rag_evaluation(rag_chain)
    
    # Affichage et sauvegarde des résultats
    print("\nRésultats de l'évaluation :")
    df_results = result.to_pandas()
    print(df_results)
    
    # Calcul des moyennes
    print("\nScores moyens :")
    for metric, score in result.items():
        print(f"- {metric}: {score:.4f}")
        
    # Sauvegarde optionnelle
    df_results.to_csv("rag_evaluation_results.csv", index=False)
    print("\nRésultats détaillés sauvegardés dans 'rag_evaluation_results.csv'")

if __name__ == "__main__":
    try:
        evaluate_rag()
    except Exception as e:
        print(f"Une erreur est survenue lors de l'évaluation : {e}")
