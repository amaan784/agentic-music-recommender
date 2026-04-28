"""Retrieval chain: converts user preferences to queries and fetches candidates."""

from typing import Dict, List, Any

from rag.vectorstore import load_vectorstore


def build_query(user_preferences: Dict[str, Any]) -> str:
    """Convert user preferences into a natural language query string."""
    parts = []

    genres = user_preferences.get("genres", [])
    if genres:
        parts.append(f"{', '.join(genres)} music")

    mood = user_preferences.get("mood", "")
    if mood:
        parts.append(f"{mood} mood")

    energy = user_preferences.get("energy", "")
    if energy:
        parts.append(f"{energy}")

    danceability = user_preferences.get("danceability", "")
    if danceability:
        parts.append(f"{danceability} danceability")

    additional = user_preferences.get("additional", "")
    if additional:
        parts.append(additional)

    if not parts:
        return "recommend popular music tracks"

    return " ".join(parts)


def retrieve_candidates(
    user_preferences: Dict[str, Any],
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    """Build query, retrieve from ChromaDB, and return scored candidates."""
    query = build_query(user_preferences)
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)

    candidates = []
    for doc, score in results:
        candidates.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": float(score),
        })
    return candidates
