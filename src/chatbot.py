import logging
import os
from datetime import date, datetime
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from src.vector_store import load_vector_store


logger = logging.getLogger(__name__)
PARIS_TZ = ZoneInfo("Europe/Paris")


class QueryFilters(BaseModel):
    """Contraintes extraites de la question utilisateur."""

    search_query: str = Field(
        default="",
        description=(
            "Termes utiles pour la recherche sémantique, sans les dates "
            "relatives. Vide si l'utilisateur demande tous les événements "
            "d'une période sans thème particulier."
        ),
    )
    start_date: date | None = Field(
        default=None,
        description="Début de la période demandée au format YYYY-MM-DD, ou null.",
    )
    end_date: date | None = Field(
        default=None,
        description="Fin de la période demandée au format YYYY-MM-DD, ou null.",
    )
    list_all: bool = Field(
        default=False,
        description=(
            "True si l'utilisateur demande explicitement tous les événements "
            "correspondant aux contraintes données, sans filtre thématique "
            "supplémentaire."
        ),
    )


@dataclass
class RAGChatbot:
    vector_store: FAISS
    query_parser: Any
    answer_chain: Any
    # Cache des documents complets du vector_store, utilisé pour les recherches
    # exhaustives (list_all). None tant qu'il n'a pas été construit.
    _all_documents_cache: list[Document] | None = None


def _today() -> date:
    """Date du jour dans le fuseau horaire français."""
    return datetime.now(PARIS_TZ).date()


def _parse_document_date(value: Any) -> date | None:
    """Convertit une date OpenAgenda en date Python."""
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None

    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        ).date()
    except ValueError:
        pass

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _overlaps(
    document: Document,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    """Vérifie si l'événement chevauche la période demandée."""
    event_start = _parse_document_date(
        document.metadata.get("start_date")
    )
    event_end = _parse_document_date(
        document.metadata.get("end_date")
    )

    if event_start is None and event_end is None:
        return False

    if event_start is None:
        event_start = event_end
    if event_end is None:
        event_end = event_start

    if start_date is not None and event_end < start_date:
        return False
    if end_date is not None and event_start > end_date:
        return False

    return True


def _unique_events(
    documents: list[Document],
    limit: int | None = None,
) -> list[Document]:
    """Garde un seul chunk par événement."""
    result = []
    seen = set()

    for document in documents:
        uid = str(document.metadata.get("uid", "")).strip()
        key = uid or document.page_content[:120]

        if key in seen:
            continue

        seen.add(key)
        result.append(document)

        if limit is not None and len(result) >= limit:
            break

    return result


def _all_documents(chatbot: "RAGChatbot") -> list[Document]:
    """Récupère tous les documents du docstore FAISS, avec mise en cache."""
    if chatbot._all_documents_cache is not None:
        return chatbot._all_documents_cache

    vector_store = chatbot.vector_store
    documents = []

    for docstore_id in vector_store.index_to_docstore_id.values():
        document = vector_store.docstore.search(docstore_id)

        if document is not None:
            documents.append(document)

    chatbot._all_documents_cache = documents
    return documents


def _sort_by_start_date(documents: list[Document]) -> list[Document]:
    """Trie les événements par date de début quand elle est disponible."""

    def sort_key(document: Document):
        event_start = _parse_document_date(
            document.metadata.get("start_date")
        )
        return event_start or date.max

    return sorted(
        documents,
        key=sort_key,
    )


def _build_answer_chain(llm):
    prompt = ChatPromptTemplate.from_template(
        """
Tu es un assistant spécialisé dans les événements culturels à Lille et dans les Hauts-de-France.

Réponds uniquement à partir du contexte fourni.

Règles :
- N'invente aucune information.
- N'utilise pas tes connaissances générales pour compléter le contexte.
- Respecte les noms, dates, horaires, lieux, adresses et descriptions tels qu'ils apparaissent dans le contexte.
- Une question peut porter sur le passé, le présent ou le futur.
- Ne supprime pas un événement simplement parce qu'il est passé : respecte la période demandée.
- Le contexte a déjà été filtré par le programme lorsqu'une période a été identifiée.
- Ne réintroduis jamais un événement absent du contexte.
- Ne recalcule pas et ne modifie pas les dates.
- Si le contexte ne permet pas de répondre, réponds exactement :
  "Je ne dispose pas de cette information."

Réponds en français, de manière claire, concise et factuelle.
Pour plusieurs événements, utilise une liste.
Mets le nom de l'événement en gras.
N'affiche Date, Horaire, Lieu, Adresse ou Description que si l'information est disponible.

Question :
{question}

Date actuelle :
{current_date}

Contexte :
{context}
""".strip()
    )

    document_prompt = PromptTemplate.from_template(
        """
ÉVÉNEMENT
UID : {uid}
TITRE : {title}
LIEU : {location}
ADRESSE : {address}
DÉBUT : {start_date}
FIN : {end_date}

CONTENU :
{page_content}
""".strip()
    )

    return create_stuff_documents_chain(
        llm,
        prompt,
        document_prompt=document_prompt,
    )


def get_chatbot_chain() -> RAGChatbot:
    """Initialise le parseur de question et la chaîne de réponse."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "La variable d'environnement MISTRAL_API_KEY n'est pas définie."
        )

    model_name = os.getenv(
        "MISTRAL_MODEL",
        "mistral-small-latest",
    )

    llm = ChatMistralAI(
        mistral_api_key=api_key,
        model=model_name,
        temperature=0,
    )

    query_parser = llm.with_structured_output(
        QueryFilters,
        include_raw=False,
    )

    vector_store = load_vector_store()
    answer_chain = _build_answer_chain(llm)

    logger.info(
        "Chatbot RAG initialisé avec le modèle '%s'.",
        model_name,
    )

    return RAGChatbot(
        vector_store=vector_store,
        query_parser=query_parser,
        answer_chain=answer_chain,
    )


def _parse_query(
    chatbot: RAGChatbot,
    question: str,
    today: date,
) -> QueryFilters:
    """Laisse Mistral interpréter le langage naturel de la question."""
    prompt = f"""
Analyse cette demande concernant des événements culturels.

Date actuelle : {today.isoformat()}

Extrait uniquement les éléments nécessaires à la recherche :
- search_query : termes importants pour la recherche, sans les dates relatives ;
- start_date : début de la période demandée ;
- end_date : fin de la période demandée.

Interprète naturellement les expressions telles que :
"aujourd'hui", "demain", "ce week-end", "le mois dernier",
"la semaine prochaine", "samedi prochain", "dans deux semaines", etc.

Pour une période, retourne ses deux bornes.
Pour une date précise, utilise la même date comme début et fin.
Pour une question sans contrainte temporelle, retourne null pour les deux dates.
Si la question exprime seulement une borne (par exemple "à partir de novembre"),
retourne uniquement la borne connue.

Ne réponds pas à la question. Retourne uniquement la structure demandée.
""".strip()

    filters = chatbot.query_parser.invoke(
        [
            ("system", prompt),
            ("human", question),
        ]
    )

    if (
        filters.start_date is not None
        and filters.end_date is not None
        and filters.end_date < filters.start_date
    ):
        raise ValueError(
            "La période extraite de la question est invalide."
        )

    return filters


def _retrieve(
    chatbot: RAGChatbot,
    question: str,
    filters: QueryFilters,
) -> list[Document]:
    """Effectue la recherche FAISS, avec filtre de période si nécessaire."""
    k = int(os.getenv("RETRIEVER_K", "10"))
    max_events = int(os.getenv("MAX_CONTEXT_EVENTS", "20"))

    search_query = filters.search_query.strip()

    has_temporal_constraint = (
        filters.start_date is not None
        or filters.end_date is not None
    )

    if has_temporal_constraint:
        # Cas particulier : l'utilisateur demande explicitement
        # TOUS les événements de la période, sans thème particulier.
        # Une recherche vectorielle n'est pas adaptée à cette demande
        # car elle ne garantit pas de récupérer tous les événements.
        if filters.list_all and not search_query:
            documents = _all_documents(chatbot)

            documents = [
                document
                for document in documents
                if _overlaps(
                    document,
                    filters.start_date,
                    filters.end_date,
                )
            ]

            documents = _unique_events(
                documents,
                limit=None,
            )

            documents = _sort_by_start_date(
                documents
            )

            logger.info(
                "Recherche exhaustive : %s -> %s, %d événements retenus.",
                filters.start_date or "-∞",
                filters.end_date or "+∞",
                len(documents),
            )

            return documents

        # Recherche thématique + contrainte temporelle.
        # Le filtre de dates est appliqué avant la sélection finale.
        search_query = search_query or question

        total_chunks = int(chatbot.vector_store.index.ntotal)
        candidate_k = min(
            max(k * 4, 40),
            total_chunks,
        )

        documents = chatbot.vector_store.similarity_search(
            search_query,
            k=candidate_k,
            fetch_k=total_chunks,
            filter=lambda metadata: _overlaps(
                Document(page_content="", metadata=metadata),
                filters.start_date,
                filters.end_date,
            ),
        )

        documents = _unique_events(
            documents,
            max_events,
        )

        logger.info(
            "Recherche temporelle : %s -> %s, %d événements retenus.",
            filters.start_date or "-∞",
            filters.end_date or "+∞",
            len(documents),
        )

        return documents

    # Pas de contrainte temporelle : recherche sémantique classique.
    search_query = search_query or question

    documents = chatbot.vector_store.similarity_search(
        search_query,
        k=k,
    )

    documents = _unique_events(
        documents,
        max_events,
    )

    logger.info(
        "Recherche sémantique : %d événements retenus.",
        len(documents),
    )

    return documents


def ask_chatbot_with_context(
    query: str,
    chatbot: RAGChatbot,
) -> dict:
    """
    Variante de `ask_chatbot` qui renvoie également les documents retrouvés.

    Utile pour l'évaluation Ragas, qui a besoin de la liste des contextes
    utilisés pour générer la réponse.

    Retourne un dict : {"answer": str, "context": list[Document], "filters": QueryFilters}.
    """
    question = query.strip()
    if not question:
        raise ValueError("La question ne peut pas être vide.")

    today = _today()

    # 1. Compréhension du langage naturel.
    filters = _parse_query(chatbot, question, today)

    logger.info(
        "Question analysée : start=%r, end=%r, search=%r, list_all=%r",
        filters.start_date,
        filters.end_date,
        filters.search_query,
        filters.list_all,
    )

    # 2. Recherche dans FAISS.
    documents = _retrieve(chatbot, question, filters)

    if not documents:
        return {
            "answer": "Je ne dispose pas de cette information.",
            "context": [],
            "filters": filters,
        }

    # 3. Génération de la réponse finale.
    answer = chatbot.answer_chain.invoke(
        {
            "question": question,
            "current_date": today.isoformat(),
            "context": documents,
        }
    )

    if not answer:
        raise RuntimeError(
            "La chaîne RAG n'a pas retourné de réponse."
        )

    return {"answer": answer, "context": documents, "filters": filters}


def ask_chatbot(
    query: str,
    chatbot: RAGChatbot,
) -> str:
    """Interprète la question, récupère les documents puis génère la réponse."""
    return ask_chatbot_with_context(query, chatbot)["answer"]