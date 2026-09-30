import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

logger = logging.getLogger(__name__)


def get_embeddings():
    """Crée l'objet d'embeddings Mistral utilisé par l'index."""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable d'environnement MISTRAL_API_KEY n'est pas définie."
        )

    return MistralAIEmbeddings(
        mistral_api_key=api_key,
        model=os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
    )


def create_chunks(df, text_column="full_description"):
    """
    Découpe les événements en documents LangChain pour l'indexation.
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

    documents = []

    for row in df.itertuples(index=False):
        text = getattr(row, text_column, "")

        if not isinstance(text, str) or not text.strip():
            continue

        metadata = {
            "uid": _clean_metadata_value(getattr(row, "uid", None)),
            "title": _clean_metadata_value(getattr(row, "title_fr", None)),
            "location": _clean_metadata_value(
                getattr(row, "location_name", None)
            ),
            "address": _clean_metadata_value(
                getattr(row, "location_address", None)
            ),
            "start_date": _clean_metadata_value(
                getattr(row, "firstdate_begin", None)
            ),
            "end_date": _clean_metadata_value(
                getattr(row, "lastdate_end", None)
            ),
        }

        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    logger.info(
        "%d documents/chunks créés à partir de %d événements.",
        len(documents),
        len(df),
    )

    return documents


def _clean_metadata_value(value):
    """Évite de stocker NaN/NaT dans les métadonnées FAISS."""
    if value is None:
        return None

    try:
        # pandas n'est pas importé volontairement dans ce helper ;
        # la conversion textuelle reste suffisante pour les métadonnées.
        if str(value) in {"nan", "NaT"}:
            return None
    except Exception:
        return None

    return value


def build_vector_store(documents):
    """Construit un index FAISS à partir des documents."""
    if not documents:
        raise ValueError(
            "Impossible de construire un index FAISS sans documents."
        )

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    return vector_store


def save_vector_store(vector_store, path=None):
    """Sauvegarde l'index FAISS localement."""
    path = path or os.getenv(
        "FAISS_INDEX_PATH",
        "data/faiss_index",
    )

    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

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

    if not index_path.is_dir():
        raise FileNotFoundError(
            f"Le dossier de l'index FAISS est introuvable : {index_path}"
        )

    embeddings = get_embeddings()

    # allow_dangerous_deserialization est nécessaire avec LangChain
    # pour charger le fichier pickle associé à l'index.
    # Ne charger que des fichiers d'index générés/contrôlés par le projet.
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search_events(query, vector_store, k=4):
    """Retourne les documents les plus proches sémantiquement."""
    if not query.strip():
        return []

    if k <= 0:
        raise ValueError("k doit être supérieur à 0.")

    return vector_store.similarity_search(
        query,
        k=k,
    )


if __name__ == "__main__":
    from src.data_ingestion import fetch_openagenda_events, process_events

    print("Récupération des événements...")
    events = fetch_openagenda_events()

    if not events:
        print("Aucun événement trouvé.")
        raise SystemExit(0)

    df = process_events(events)

    print(f"Nombre d'événements à indexer : {len(df)}")

    print("Création des chunks...")
    documents = create_chunks(df)

    print(f"Nombre de documents créés : {len(documents)}")

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
