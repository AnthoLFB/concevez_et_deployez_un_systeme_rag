import html
import logging
import os
import re
from datetime import date, datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Colonnes dont on a besoin pour le traitement et le RAG.
EVENT_COLUMNS = [
    "uid",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "location_city",
    "location_name",
    "location_address",
    "firstdate_begin",
    "lastdate_end",
]

TEXT_COLUMNS = [
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "location_city",
    "location_name",
    "location_address",
]

CONTENT_COLUMNS = [
    "title_fr",
    "description_fr",
    "longdescription_fr",
]


def one_year_before_or_after(value: date, years: int) -> date:
    """Ajoute/soustrait un nombre entier d'années sans dépendance supplémentaire."""
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # Cas du 29 février lorsqu'on arrive sur une année non bissextile.
        return value.replace(month=2, day=28, year=value.year + years)


def clean_text(value) -> str:
    """
    Nettoie un texte OpenAgenda sans supprimer les accents, la ponctuation
    ou les informations utiles à la recherche sémantique.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)

    # Décodage des entités HTML (&amp;, &quot;, etc.).
    text = html.unescape(text)

    # Transforme quelques balises de structure en séparateurs lisibles.
    text = re.sub(r"<\s*(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Supprime les balises HTML restantes.
    text = re.sub(r"<[^>]+>", " ", text)

    # Supprime certains caractères invisibles / de contrôle.
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Supprime les URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalise les espaces, sans toucher aux accents ni à la casse.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    # Si le texte est entièrement en majuscules (fréquent sur certains titres),
    # on le normalise pour ne pas perturber les embeddings.
    if text.isupper() and len(text) > 10:
        text = text.capitalize()

    return text.strip()


def format_date_fr(iso_str: str) -> str:
    """
    Transforme une date ISO (ex: 2024-12-01T18:30:00Z) en format texte
    naturel pour le LLM (ex: dimanche 1 décembre 2024 à 18h30).
    """
    if not iso_str or not isinstance(iso_str, str):
        return ""

    try:
        # On gère le suffixe Z de manière compatible ISO.
        clean_iso = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_iso)

        days = [
            "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"
        ]
        months = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]

        jour_semaine = days[dt.weekday()]
        jour = dt.day
        mois_str = months[dt.month - 1]
        annee = dt.year
        heure = dt.hour
        minute = dt.minute

        res = f"{jour_semaine} {jour} {mois_str} {annee}"
        if heure != 0 or minute != 0:
            res += f" à {heure:02d}h{minute:02d}"

        return res
    except Exception:
        # En cas d'erreur de parsing, on renvoie la chaîne originale nettoyée.
        return iso_str


def fetch_openagenda_events() -> list[dict]:
    """
    Récupère tous les événements correspondant au besoin du projet :

    - ville = CITY (Lille par défaut)
    - événement chevauchant la fenêtre J-1 an -> J+1 an

    L'endpoint /records renvoie au maximum 100 résultats par requête.
    On utilise donc la pagination avec offset pour récupérer tous les
    résultats tant que le volume reste dans la limite de cet endpoint.
    """
    base_url = os.getenv("OPENDATA_API_URL")
    city = os.getenv("CITY", "Lille")
    page_size = 100
    timeout = int(os.getenv("OPENDATA_TIMEOUT", "60"))

    if not base_url:
        raise ValueError(
            "La variable d'environnement OPENDATA_API_URL n'est pas définie."
        )

    today = date.today()
    start_date = one_year_before_or_after(today, -1)
    end_date = one_year_before_or_after(today, 1)

    # On prend les événements qui chevauchent la fenêtre :
    # fin >= début de fenêtre ET début <= fin de fenêtre.
    start_date_str = f"{start_date.isoformat()}T00:00:00Z"
    end_date_str = f"{end_date.isoformat()}T23:59:59Z"

    where_query = (
        f'location_city = "{city}" '
        f'and lastdate_end >= "{start_date_str}" '
        f'and firstdate_begin <= "{end_date_str}"'
    )

    logger.info(
        "Récupération OpenAgenda : ville=%s, période=%s -> %s",
        city,
        start_date.isoformat(),
        end_date.isoformat(),
    )

    # Première requête : récupérer le total correspondant au filtre.
    count_response = requests.get(
        base_url,
        params={
            "where": where_query,
            "limit": 1,
        },
        timeout=timeout,
    )
    count_response.raise_for_status()
    count_data = count_response.json()

    total_count = int(count_data.get("total_count", 0))
    max_events = int(os.getenv("MAX_EVENTS", "1000"))
    
    # On limite le nombre d'événements à récupérer
    effective_total = min(total_count, max_events)

    logger.info(
        "OpenAgenda indique %d événements correspondant au filtre (limite fixée à %d).",
        total_count,
        effective_total,
    )

    if effective_total == 0:
        return []

    # L'API records documente une limite cumulée de offset + limit < 10000.
    if effective_total >= 10000:
        raise RuntimeError(
            "Le filtre retourne 10 000 événements ou plus. "
            "L'endpoint /records atteint alors sa limite de pagination. "
            "Il faudra utiliser l'endpoint /exports pour ce volume."
        )

    all_events: list[dict] = []
    offset = 0

    while offset < effective_total:
        current_limit = min(page_size, effective_total - offset)
        response = requests.get(
            base_url,
            params={
                "where": where_query,
                "limit": current_limit,
                "offset": offset,
            },
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        all_events.extend(results)
        offset += len(results)

        logger.info(
            "Progression OpenAgenda : %d/%d événements récupérés.",
            min(offset, effective_total),
            effective_total,
        )

        if len(results) < current_limit:
            break

    # On vérifie qu'on n'a pas silencieusement récupéré moins que ce qu'on pouvait.
    if len(all_events) < effective_total and len(all_events) < total_count:
        logger.warning(
            "L'API a retourné moins d'événements que prévu : %d au lieu de %d",
            len(all_events),
            effective_total
        )

    logger.info(
        "%d événements récupérés au total.",
        len(all_events),
    )

    return all_events


def process_events(events: list[dict]) -> pd.DataFrame:
    """
    Nettoie et prépare les événements avant chunking et vectorisation.

    Nettoyage réalisé :
    - sélection des colonnes utiles ;
    - normalisation des champs texte ;
    - suppression du HTML et des entités HTML ;
    - suppression des doublons exacts ;
    - suppression des doublons par UID quand l'UID est disponible ;
    - suppression des événements sans contenu textuel exploitable ;
    - création du texte final destiné aux embeddings.

    Les statistiques sont stockées dans df.attrs["processing_stats"]
    afin de pouvoir les afficher dans les logs ou l'API.
    """
    stats = {
        "received": len(events) if events else 0,
        "exact_duplicates_removed": 0,
        "uid_duplicates_removed": 0,
        "empty_content_removed": 0,
        "noise_events_removed": 0,
        "retained": 0,
    }

    if not events:
        empty_df = pd.DataFrame(columns=EVENT_COLUMNS + ["full_description"])
        empty_df.attrs["processing_stats"] = stats
        return empty_df

    df = pd.DataFrame(events).copy()

    # Garantit l'existence des colonnes attendues.
    for column in EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[EVENT_COLUMNS].copy()

    # Nettoyage des champs texte.
    for column in TEXT_COLUMNS:
        df[column] = df[column].map(clean_text)

    # Les identifiants sont stockés sous forme de chaîne propre.
    df["uid"] = df["uid"].replace({"": None})
    df["uid"] = df["uid"].map(
        lambda value: "" if value is None or pd.isna(value) else str(value).strip()
    )

    # Normalisation légère des dates : on les garde sous forme de texte ISO
    # fourni par la source, après suppression d'espaces superflus.
    for column in ["firstdate_begin", "lastdate_end"]:
        df[column] = df[column].map(clean_text)

    # 1. Doublons strictement identiques.
    exact_duplicate_mask = df.duplicated(keep="first")
    stats["exact_duplicates_removed"] = int(exact_duplicate_mask.sum())
    df = df.loc[~exact_duplicate_mask].copy()

    # 2. Doublons ayant le même UID.
    has_uid = df["uid"].ne("")
    uid_duplicate_mask = has_uid & df["uid"].duplicated(keep="first")
    stats["uid_duplicates_removed"] = int(uid_duplicate_mask.sum())
    df = df.loc[~uid_duplicate_mask].copy()

    # 3. Suppression des événements qui n'ont aucun contenu descriptif.
    has_content = df[CONTENT_COLUMNS].apply(
        lambda row: any(bool(value.strip()) for value in row),
        axis=1,
    )
    stats["empty_content_removed"] = int((~has_content).sum())
    df = df.loc[has_content].copy()

    # 4. Suppression des événements de "bruit" (tests, annulations).
    noise_keywords = ["test ", "test_", "lorem ipsum", "à supprimer", "annulé", "reporté"]
    is_noise = df["title_fr"].str.lower().apply(
        lambda x: any(kw in x for kw in noise_keywords) or x.strip() == "test"
    )
    stats["noise_events_removed"] = int(is_noise.sum())
    df = df.loc[~is_noise].copy()

    # 5. Construit un texte enrichi pour les embeddings.
    def build_full_description(row) -> str:
        sections = []

        # On extrait l'année pour la mettre en évidence au début du texte
        if row["firstdate_begin"]:
            try:
                dt = datetime.fromisoformat(row["firstdate_begin"].replace("Z", "+00:00"))
                sections.append(f"ÉVÉNEMENT PRÉVU EN {dt.year}")
            except:
                pass

        if row["title_fr"]:
            sections.append(f"Titre : {row['title_fr']}")

        if row["description_fr"]:
            sections.append(f"Description : {row['description_fr']}")

        if row["longdescription_fr"]:
            sections.append(
                f"Description détaillée : {row['longdescription_fr']}"
            )

        if row["location_name"]:
            sections.append(f"Lieu : {row['location_name']}")

        if row["location_address"]:
            sections.append(f"Adresse : {row['location_address']}")

        # On utilise le formatage naturel pour les dates.
        if row["firstdate_begin"]:
            date_text = format_date_fr(row["firstdate_begin"])
            sections.append(f"Début : {date_text}")

        if row["lastdate_end"]:
            date_text = format_date_fr(row["lastdate_end"])
            sections.append(f"Fin : {date_text}")

        return "\n".join(sections)

    df["full_description"] = df.apply(
        build_full_description,
        axis=1,
    )

    stats["retained"] = len(df)
    df.attrs["processing_stats"] = stats

    logger.info(
        "Nettoyage terminé : reçus=%d, doublons exacts supprimés=%d, "
        "doublons UID supprimés=%d, sans contenu supprimés=%d, "
        "bruit supprimés=%d, conservés=%d.",
        stats["received"],
        stats["exact_duplicates_removed"],
        stats["uid_duplicates_removed"],
        stats["empty_content_removed"],
        stats["noise_events_removed"],
        stats["retained"],
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    events = fetch_openagenda_events()
    print(f"Événements récupérés : {len(events)}")

    df = process_events(events)
    print(f"Événements conservés : {len(df)}")
    print("Statistiques :", df.attrs.get("processing_stats", {}))
    print(df[["uid", "title_fr", "location_city", "firstdate_begin", "lastdate_end"]].head())
