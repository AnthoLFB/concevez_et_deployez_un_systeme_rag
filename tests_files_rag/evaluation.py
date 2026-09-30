import logging
import os
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)


load_dotenv()

logger = logging.getLogger(__name__)


def _generate_rag_results(rag_chain, test_data):
    """
    Exécute le RAG sur chaque question et récupère les réponses/contextes.
    """
    questions = [item["question"] for item in test_data]

    answers = []
    contexts = []

    for question in questions:
        logger.info(
            "Évaluation de la question : %s",
            question,
        )

        response = rag_chain.invoke(
            {"input": question}
        )

        answer = response.get("answer")
        retrieved_contexts = response.get("context", [])

        if answer is None:
            raise RuntimeError(
                "La chaîne RAG n'a pas retourné de réponse "
                f"pour la question : {question}"
            )

        answers.append(answer)
        contexts.append(
            [
                document.page_content
                for document in retrieved_contexts
            ]
        )

    return answers, contexts


def _build_evaluation_result_dataframe(
    result,
) -> pd.DataFrame:
    """Convertit proprement le résultat Ragas en DataFrame."""
    if not hasattr(result, "to_pandas"):
        raise RuntimeError(
            "La version installée de Ragas ne fournit pas "
            "la méthode to_pandas() sur le résultat."
        )

    dataframe = result.to_pandas()

    # Les NaN sont remplacés par None pour obtenir un JSON propre.
    return dataframe.astype(object).where(
        pd.notna(dataframe),
        None,
    )


def _calculate_average_scores(dataframe: pd.DataFrame) -> dict[str, float]:
    """
    Calcule la moyenne de chaque métrique en ignorant les valeurs NaN.
    """
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]

    scores = {}

    for metric_name in metric_names:
        if metric_name not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[metric_name],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        scores[metric_name] = float(values.mean())

    return scores


def run_rag_evaluation(
    rag_chain,
    test_data: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Évalue le système RAG avec Ragas.

    Cette implémentation cible l'API d'évaluation Ragas 0.4.x
    et désactive explicitement le patch nest_asyncio.
    """
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError(
            "La variable d'environnement MISTRAL_API_KEY n'est pas définie."
        )

    answers, contexts = _generate_rag_results(
        rag_chain,
        test_data,
    )

    # On ajoute les réponses et les contextes produits par le RAG.
    dataset = Dataset.from_dict(
        {
            "question": [
                item["question"] for item in test_data
            ],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [
                item["ground_truth"] for item in test_data
            ],
        }
    )

    evaluation_model = os.getenv(
        "MISTRAL_EVAL_MODEL",
        "mistral-large-latest",
    )

    logger.info(
        "Évaluation Ragas avec le modèle : %s",
        evaluation_model,
    )

    # Ragas fournit des adaptateurs pour utiliser un LLM LangChain.
    evaluator_llm = LangchainLLMWrapper(
        ChatMistralAI(
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            model=evaluation_model,
            temperature=0.0,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            model=os.getenv(
                "MISTRAL_EMBEDDING_MODEL",
                "mistral-embed",
            ),
        )
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
    ]

    # allow_nest_asyncio=False évite que Ragas modifie la boucle
    # asyncio de l'application.
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        allow_nest_asyncio=False,
    )

    result_dataframe = _build_evaluation_result_dataframe(result)

    scores = _calculate_average_scores(result_dataframe)

    details = result_dataframe.to_dict(
        orient="records"
    )

    logger.info(
        "Évaluation terminée. Scores : %s",
        scores,
    )

    return {
        "scores": scores,
        "details": details,
    }
