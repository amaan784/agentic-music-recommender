"""ChromaDB vector store setup and retrieval."""

import os
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")


def get_embeddings():
    """Return the configured embedding model."""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_vectorstore(persist_directory: str = CHROMA_DIR) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vectorstore


def retrieve(
    query: str,
    top_k: int = 30,
    persist_directory: str = CHROMA_DIR,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K tracks from ChromaDB matching the query.

    Returns a list of dicts with keys: content, metadata, relevance_score.
    """
    vectorstore = load_vectorstore(persist_directory)
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)

    tracks = []
    for doc, score in results:
        tracks.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": float(score),
        })
    return tracks
