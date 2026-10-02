"""
Script d'évaluation locale du système RAG avec Ragas.

Charge un jeu de questions / `ground_truth` au format JSON. Par défaut,
le fichier `tests/ground_truth.json` (généré par
`scripts/generate_ground_truth.py` à partir des événements réellement
indexés) est utilisé. On peut forcer un autre chemin via la variable
d'environnement `RAG_EVAL_DATASET`.
"""
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.chatbot import get_chatbot_chain
from src.evaluation import run_rag_evaluation


load_dotenv()
logging.basicConfig(level=logging.INFO)


# Jeu par défaut si aucun chemin n'est explicitement fourni.
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "tests" / "ground_truth.json"

# Jeu minimal de secours si aucun fichier n'est disponible.
FALLBACK_TEST_DATA = [
    {
        "question": "Quels événements à Lille ce week-end ?",
        "ground_truth": "À adapter manuellement selon les données réellement indexées.",
    },
]


def _load_test_data() -> list[dict]:
    """Charge le jeu d'évaluation depuis un fichier JSON si disponible."""
    dataset_path = os.getenv("RAG_EVAL_DATASET")
    candidate = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH

    if candidate.is_file():
        with candidate.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError(
                f"Le fichier {candidate} doit contenir une liste non vide."
            )
        print(f"Jeu d'évaluation chargé depuis : {candidate}")
        return data

    print(
        "Aucun fichier de ground truth trouvé "
        f"({candidate}), utilisation du jeu de secours."
    )
    return FALLBACK_TEST_DATA


def evaluate_rag() -> None:
    print("--- Évaluation du système RAG avec Ragas ---")

    test_data = _load_test_data()
    print(f"Jeu d'évaluation : {len(test_data)} question(s).")

    print("Initialisation du chatbot...")
    rag_chain = get_chatbot_chain()

    print("Lancement de l'évaluation Ragas (peut prendre quelques minutes)...")
    result = run_rag_evaluation(rag_chain, test_data)

    scores = result["scores"]
    details = result["details"]

    print("\nScores moyens :")
    for metric, value in scores.items():
        print(f"- {metric}: {value:.4f}")

    output_path = Path("rag_evaluation_results.csv")
    pd.DataFrame(details).to_csv(output_path, index=False)
    print(f"\nDétails sauvegardés dans : {output_path}")


if __name__ == "__main__":
    try:
        evaluate_rag()
    except Exception as exc:
        print(f"Une erreur est survenue lors de l'évaluation : {exc}")
        sys.exit(1)
