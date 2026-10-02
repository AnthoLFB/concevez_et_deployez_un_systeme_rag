from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.chatbot import (
    QueryFilters,
    RAGChatbot,
    _overlaps,
    _unique_events,
    ask_chatbot,
    get_chatbot_chain,
)


@patch("src.chatbot.load_vector_store")
@patch("src.chatbot.ChatMistralAI")
def test_get_chatbot_chain(mock_mistral_cls, mock_load_vs, monkeypatch):
    """get_chatbot_chain assemble vector_store + parser + answer_chain."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

    mock_vs = MagicMock()
    mock_load_vs.return_value = mock_vs

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock(name="parser")
    mock_mistral_cls.return_value = mock_llm

    chatbot = get_chatbot_chain()

    assert isinstance(chatbot, RAGChatbot)
    assert chatbot.vector_store is mock_vs
    mock_load_vs.assert_called_once()
    mock_mistral_cls.assert_called_once()
    mock_llm.with_structured_output.assert_called_once()


def test_get_chatbot_chain_requires_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        get_chatbot_chain()


def test_ask_chatbot_returns_answer():
    """ask_chatbot délègue à la chaîne et renvoie uniquement la réponse texte."""
    chatbot = RAGChatbot(
        vector_store=MagicMock(),
        query_parser=MagicMock(),
        answer_chain=MagicMock(),
    )

    # Pas de contrainte temporelle, pas de search_query -> fallback sur question
    chatbot.query_parser.invoke.return_value = QueryFilters()
    chatbot.vector_store.similarity_search.return_value = [
        Document(page_content="info", metadata={"uid": "1"})
    ]
    chatbot.answer_chain.invoke.return_value = "Réponse générée"

    result = ask_chatbot("Quels événements ?", chatbot)

    assert result == "Réponse générée"
    chatbot.answer_chain.invoke.assert_called_once()


def test_ask_chatbot_empty_question_raises():
    chatbot = RAGChatbot(
        vector_store=MagicMock(),
        query_parser=MagicMock(),
        answer_chain=MagicMock(),
    )
    with pytest.raises(ValueError):
        ask_chatbot("   ", chatbot)


def test_overlaps_handles_missing_dates():
    doc = Document(
        page_content="",
        metadata={"start_date": "2024-05-01", "end_date": "2024-05-03"},
    )
    assert _overlaps(doc, date(2024, 5, 2), date(2024, 5, 2)) is True
    assert _overlaps(doc, date(2024, 6, 1), date(2024, 6, 2)) is False
    # Sans aucune date, l'événement n'est pas gardé
    doc_empty = Document(page_content="", metadata={})
    assert _overlaps(doc_empty, date(2024, 1, 1), None) is False


def test_unique_events_deduplicates_by_uid():
    docs = [
        Document(page_content="a", metadata={"uid": "1"}),
        Document(page_content="b", metadata={"uid": "1"}),
        Document(page_content="c", metadata={"uid": "2"}),
    ]
    result = _unique_events(docs)
    assert len(result) == 2
    assert [d.metadata["uid"] for d in result] == ["1", "2"]
