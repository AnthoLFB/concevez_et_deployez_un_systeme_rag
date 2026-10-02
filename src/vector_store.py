import logging
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def get_embeddings():
    """Crée l'objet d'embeddings utilisé pour construire et charger FAISS."""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable d'environnement MISTRAL_API_KEY n'est pas définie."
        )

    return MistralAIEmbeddings(
        mistral_api_key=api_key,
        model=os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
    )


def clean_metadata_value(value):
    """Convertit les valeurs Pandas problématiques en valeurs JSON simples."""
    if value is None:
        return None

    try:
        if value != value:  # NaN
            return None
    except Exception:
        pass

    text = str(value).strip()
    return text if text else None


def create_chunks(df, text_column="full_description") -> list[Document]:
    """
    Découpe les événements nettoyés en chunks avant vectorisation.

    Chaque chunk devient un Document LangChain avec ses métadonnées.
    """
    if df.empty:
        return []

    if text_column not in df.columns:
        raise ValueError(
            f"La colonne '{text_column}' est absente du DataFrame."
        )

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE doit être supérieur à 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP doit être compris entre 0 et CHUNK_SIZE - 1."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    documents: list[Document] = []
    skipped_events = 0

    for row in df.itertuples(index=False):
        text = getattr(row, text_column, "")

        if not isinstance(text, str) or not text.strip():
            skipped_events += 1
            continue

        metadata = {
            "uid": clean_metadata_value(getattr(row, "uid", None)),
            "title": clean_metadata_value(getattr(row, "title_fr", None)),
            "location_city": clean_metadata_value(
                getattr(row, "location_city", None)
            ),
            "location": clean_metadata_value(
                getattr(row, "location_name", None)
            ),
            "address": clean_metadata_value(
                getattr(row, "location_address", None)
            ),
            "start_date": clean_metadata_value(
                getattr(row, "firstdate_begin", None)
            ),
            "end_date": clean_metadata_value(
                getattr(row, "lastdate_end", None)
            ),
        }

        chunks = text_splitter.split_text(text)

        for chunk_index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                    },
                )
            )

    logger.info(
        "Chunking terminé : %d événements en entrée, %d chunks créés, "
        "%d événements ignorés.",
        len(df),
        len(documents),
        skipped_events,
    )

    return documents


def build_vector_store(documents: list[Document]):
    """Vectorise chaque chunk avec Mistral et construit l'index FAISS."""
    if not documents:
        raise ValueError(
            "Impossible de construire FAISS : aucun chunk à vectoriser."
        )

    embeddings = get_embeddings()

    logger.info(
        "Vectorisation et indexation de %d chunks dans FAISS...",
        len(documents),
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    logger.info("Index FAISS créé avec succès.")

    return vector_store


def save_vector_store(vector_store, path=None):
    """Sauvegarde l'index FAISS localement."""
    path = path or os.getenv(
        "FAISS_INDEX_PATH",
        "data/faiss_index",
    )

    index_path = Path(path)
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(str(index_path))

    logger.info(
        "Index FAISS sauvegardé dans : %s",
        index_path,
    )


def load_vector_store(path=None):
    """Charge un index FAISS local depuis le disque."""
    path = path or os.getenv(
        "FAISS_INDEX_PATH",
        "data/faiss_index",
    )

    index_path = Path(path)

    if not (index_path / "index.faiss").is_file():
        raise FileNotFoundError(
            f"Fichier index.faiss introuvable dans : {index_path}"
        )

    if not (index_path / "index.pkl").is_file():
        raise FileNotFoundError(
            f"Fichier index.pkl introuvable dans : {index_path}"
        )

    embeddings = get_embeddings()

    # SECURITE : allow_dangerous_deserialization=True désérialise un pickle.
    # Ce flag est acceptable UNIQUEMENT parce que index.pkl est produit par
    # notre propre pipeline (/rebuild) et stocké localement.
    # Ne JAMAIS charger un index.pkl provenant d'une source externe non fiable.
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search_events(query, vector_store, k=4):
    """Recherche les chunks les plus proches sémantiquement."""
    if not isinstance(query, str) or not query.strip():
        return []

    if k <= 0:
        raise ValueError("k doit être supérieur à 0.")

    return vector_store.similarity_search(
        query,
        k=k,
    )


if __name__ == "__main__":
    from src.data_ingestion import (
        fetch_openagenda_events,
        process_events,
    )

    logging.basicConfig(level=logging.INFO)

    print("Récupération des événements...")
    events = fetch_openagenda_events()

    if not events:
        print("Aucun événement trouvé.")
        raise SystemExit(0)

    print(f"Événements récupérés : {len(events)}")

    df = process_events(events)
    print(f"Événements conservés après nettoyage : {len(df)}")
    print("Statistiques de nettoyage :", df.attrs.get("processing_stats", {}))

    print("Création des chunks...")
    documents = create_chunks(df)
    print(f"Chunks créés : {len(documents)}")

    print("Construction de l'index FAISS...")
    vector_store = build_vector_store(documents)

    print("Sauvegarde de l'index...")
    save_vector_store(vector_store)

    query = "concert de musique à Lille"
    print(f"\nTest de recherche pour : '{query}'")

    matches = search_events(query, vector_store)

    for index, document in enumerate(matches, start=1):
        print(
            f"{index}. "
            f"{document.metadata.get('title')} "
            f"({document.metadata.get('location')})"
        )
