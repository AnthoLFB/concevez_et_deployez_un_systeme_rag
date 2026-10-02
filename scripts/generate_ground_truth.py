"""
Génère un jeu de vérités de terrain (ground truth) pour l'évaluation RAG.

Les questions/réponses sont dérivées **directement** des événements réellement
indexés (`data/processed_events.pkl`) afin que les réponses attendues soient
vérifiables de manière objective, puis sauvegardées au format JSON attendu
par `src/evaluation.run_rag_evaluation` (clés `question` et `ground_truth`).

Usage :
    uv run python scripts/generate_ground_truth.py
    # puis :
    $env:RAG_EVAL_DATASET = "tests/ground_truth.json"
    uv run python evaluate_rag.py
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_events.pkl"
OUTPUT_PATH = ROOT / "tests" / "ground_truth.json"

# Reproductibilité du tirage.
RANDOM_SEED = 42
# Nombre cible d'items dans le jeu d'évaluation.
TARGET_SIZE = 75


# --- Utilitaires de formatage ---------------------------------------------- #

FRENCH_MONTHS = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}


def _parse_date(value) -> datetime | None:
    if not value or pd.isna(value):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _format_date_fr(dt: datetime) -> str:
    return f"{dt.day} {FRENCH_MONTHS[dt.month]} {dt.year}"


def _clean(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip()
    # Compacter les espaces internes.
    return re.sub(r"\s+", " ", text)


def _location_phrase(row) -> str:
    name = _clean(row.get("location_name"))
    address = _clean(row.get("location_address"))

    if name and address and name.lower() != address.lower():
        return f"{name} ({address})"
    return name or address or "lieu non précisé"


# --- Générateurs de couples question / ground_truth ------------------------ #

def _qa_where(row) -> dict | None:
    """Où a lieu l'événement ?"""
    title = _clean(row.get("title_fr"))
    location = _location_phrase(row)

    if not title or location == "lieu non précisé":
        return None

    return {
        "question": f"Où se déroule l'événement « {title} » ?",
        "ground_truth": (
            f"L'événement « {title} » se déroule à {location}."
        ),
    }


def _qa_when(row) -> dict | None:
    """Quand a lieu l'événement ?"""
    title = _clean(row.get("title_fr"))
    start = _parse_date(row.get("firstdate_begin"))
    end = _parse_date(row.get("lastdate_end"))

    if not title or start is None:
        return None

    start_str = _format_date_fr(start)

    if end is not None and end.date() != start.date():
        end_str = _format_date_fr(end)
        when = f"du {start_str} au {end_str}"
    else:
        when = f"le {start_str}"

    return {
        "question": f"Quand a lieu l'événement « {title} » ?",
        "ground_truth": (
            f"L'événement « {title} » a lieu {when}."
        ),
    }


def _qa_what(row) -> dict | None:
    """De quoi parle l'événement ?"""
    title = _clean(row.get("title_fr"))
    description = _clean(row.get("description_fr"))

    if not title or not description:
        return None

    # On garde la description courte comme résumé de référence.
    if len(description) > 350:
        description = description[:347].rstrip() + "..."

    return {
        "question": f"De quoi parle l'événement « {title} » ?",
        "ground_truth": (
            f"L'événement « {title} » est décrit comme suit : {description}"
        ),
    }


GENERATORS = [_qa_where, _qa_when, _qa_what]


# --- Script principal ------------------------------------------------------ #

def generate() -> list[dict]:
    dataframe = pd.read_pickle(DATA_PATH)

    rng = random.Random(RANDOM_SEED)
    indices = list(dataframe.index)
    rng.shuffle(indices)

    results: list[dict] = []
    seen_questions: set[str] = set()

    for position, index in enumerate(indices):
        row = dataframe.loc[index].to_dict()

        # On alterne les familles de questions pour varier l'évaluation
        # (lieu / date / description).
        generator = GENERATORS[position % len(GENERATORS)]
        item = generator(row)

        if item is None:
            # Repli : on essaie les autres générateurs sur cette ligne.
            for alternate in GENERATORS:
                if alternate is generator:
                    continue
                item = alternate(row)
                if item is not None:
                    break

        if item is None:
            continue

        if item["question"] in seen_questions:
            continue

        seen_questions.add(item["question"])
        results.append(item)

        if len(results) >= TARGET_SIZE:
            break

    return results


def main() -> None:
    items = generate()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=False, indent=2)

    print(f"{len(items)} items générés dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
