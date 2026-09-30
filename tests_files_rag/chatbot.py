import logging
import os

from dotenv import load_dotenv
try:
    # LangChain récent : ces chaînes sont dans langchain-classic.
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # Compatibilité avec les versions précédentes de LangChain.
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from src.vector_store import load_vector_store


load_dotenv()

logger = logging.getLogger(__name__)


def get_chatbot_chain():
    """
    Initialise et retourne la chaîne RAG.
    """
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable d'environnement MISTRAL_API_KEY n'est pas définie."
        )

    model_name = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    # 1. Chargement du vector store
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))}
    )

    # 2. Initialisation du LLM
    llm = ChatMistralAI(
        mistral_api_key=api_key,
        model=model_name,
        temperature=float(os.getenv("MISTRAL_TEMPERATURE", "0.2")),
    )

    # 3. Prompt du RAG
    system_prompt = """
Vous êtes un assistant spécialisé dans les événements culturels à Lille
et dans les Hauts-de-France.

Répondez uniquement à partir du contexte fourni.

Règles :
- N'inventez aucune information.
- Si le contexte ne permet pas de répondre à la question, répondez :
  "Je ne dispose pas de cette information."
- Conservez les informations précises présentes dans le contexte,
  notamment les dates, lieux, adresses et descriptions.
- Lorsque plusieurs événements répondent à la question, présentez-les
  de manière claire.

Contexte :
{context}
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 4. Chaîne de génération à partir des documents récupérés
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )

    # 5. Chaîne RAG complète
    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain,
    )

    logger.info("Chaîne RAG initialisée avec le modèle '%s'.", model_name)

    return rag_chain


def ask_chatbot(query: str, rag_chain):
    """
    Pose une question au système RAG et retourne uniquement la réponse.
    """
    response = rag_chain.invoke({"input": query})

    answer = response.get("answer")

    if answer is None:
        raise RuntimeError(
            "La chaîne RAG n'a pas retourné de réponse."
        )

    return answer
