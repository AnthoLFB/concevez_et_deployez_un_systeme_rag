import logging
import os
import re
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


def fetch_openagenda_events():
    """
    Récupère les événements depuis l'API OpenDataSoft d'OpenAgenda.

    La période récupérée commence à HISTORY_YEARS avant la date du jour.
    """
    base_url = os.getenv("OPENDATA_API_URL")
    city = os.getenv("CITY", "Lille")
    history_years = int(os.getenv("HISTORY_YEARS", "1"))
    limit = int(os.getenv("OPENDATA_LIMIT", "100"))
    timeout = int(os.getenv("OPENDATA_TIMEOUT", "30"))

    if not base_url:
        raise ValueError(
            "La variable d'environnement OPENDATA_API_URL n'est pas définie."
        )

    if history_years < 0:
        raise ValueError("HISTORY_YEARS doit être supérieur ou égal à 0.")

    if limit <= 0:
        raise ValueError("OPENDATA_LIMIT doit être supérieur à 0.")

    current_date = datetime.now().date()
    start_date = current_date - timedelta(days=history_years * 365)
    start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    # On conserve ici le principe de filtrage utilisé dans le projet :
    # adresse contenant la ville + événement non terminé avant la date limite historique.
    where_query = (
        f"location_address like '{city}' "
        f"and lastdate_end >= '{start_date_str}'"
    )

    params = {
        "where": where_query,
        "limit": limit,
    }

    logger.info(
        "Récupération OpenAgenda : ville=%s, période depuis=%s, limite=%d.",
        city,
        start_date,
        limit,
    )

    response = requests.get(
        base_url,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    events = data.get("results", [])

    logger.info("%d événements récupérés.", len(events))

    return events


def clean_html(value):
    """Supprime les balises HTML d'un texte et nettoie les espaces."""
    if not isinstance(value, str):
        return ""

    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def process_events(events):
    """
    Nettoie et structure les événements avec Pandas.
    """
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)

    columns = [
        "uid",
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "location_name",
        "location_address",
        "firstdate_begin",
        "lastdate_end",
    ]

    # On conserve uniquement les colonnes disponibles dans la réponse API.
    available_columns = [
        column for column in columns if column in df.columns
    ]

    df = df[available_columns].copy()

    # Les colonnes utilisées plus bas doivent toujours exister.
    for column in columns:
        if column not in df.columns:
            df[column] = ""

    # Nettoyage du texte.
    for column in [
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "location_name",
        "location_address",
    ]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .map(clean_html)
        )

    # Création d'un texte riche pour l'index vectoriel.
    # Les informations de lieu et de dates sont volontairement incluses
    # dans le texte indexé pour aider la recherche sémantique.
    df["full_description"] = (
        "Titre : " + df["title_fr"] + "\n"
        "Description : " + df["description_fr"] + "\n"
        "Description détaillée : " + df["longdescription_fr"] + "\n"
        "Lieu : " + df["location_name"] + "\n"
        "Adresse : " + df["location_address"] + "\n"
        "Début : " + df["firstdate_begin"].astype(str) + "\n"
        "Fin : " + df["lastdate_end"].astype(str)
    )

    return df


if __name__ == "__main__":
    events = fetch_openagenda_events()

    print(f"Nombre d'événements récupérés : {len(events)}")

    if events:
        df = process_events(events)
        print(df.head())
