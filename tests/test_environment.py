import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from mistralai.client import Mistral

# Test de la configuration de base
def test_imports():
    assert faiss is not None
    assert FAISS is not None
    assert HuggingFaceEmbeddings is not None
    assert Mistral is not None