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
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from src.chatbot import get_chatbot_chain

# Chargement des variables d'environnement
load_dotenv()

def evaluate_rag():
    """
    Évalue les performances du système RAG à l'aide de Ragas.
    """
    print("--- Évaluation du système RAG avec Ragas ---")
    
    # 1. Préparation du dataset de test (exemples annotés manuellement ou fictifs pour le POC)
    # Dans un vrai cas, ces questions/réponses viendraient d'un fichier JSON/CSV
    test_data = [
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
    
    # 2. Initialisation du chatbot
    print("Initialisation du chatbot pour génération des réponses...")
    rag_chain = get_chatbot_chain()
    
    # 3. Collecte des réponses du système
    questions = [item["question"] for item in test_data]
    ground_truths = [item["ground_truth"] for item in test_data]
    answers = []
    contexts = []
    
    for q in questions:
        print(f"Traitement de la question : {q}")
        # On invoque la chaîne pour obtenir la réponse et le contexte
        response = rag_chain.invoke({"input": q})
        answers.append(response["answer"])
        # Extraction du texte des documents récupérés
        contexts.append([doc.page_content for doc in response["context"]])
        
    # 4. Création du dataset Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    # 5. Configuration des modèles pour l'évaluation
    # Ragas utilise un LLM et des Embeddings pour calculer certaines métriques
    eval_llm = ChatMistralAI(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model=os.getenv("MISTRAL_MODEL", "mistral-large-latest") # Utiliser un modèle plus puissant pour l'éval si possible
    )
    
    # 6. Exécution de l'évaluation
    print("\nLancement de l'évaluation Ragas (ceci peut prendre quelques minutes)...")
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
    
    # 7. Affichage et sauvegarde des résultats
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
