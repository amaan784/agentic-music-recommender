"""LangGraph state schema."""

from typing import TypedDict, List, Optional


class RecommenderState(TypedDict):
    user_preferences: dict          # parsed from UI input
    query: str                      # natural language query for retriever
    retrieved_candidates: list      # raw retrieval results
    guardrail_results: list         # what got filtered and why
    scored_recommendations: list    # scored + ranked list
    bias_report: dict               # from bias detector
    confidence_scores: list         # per-song confidence
    critique: Optional[dict]        # LLM critique response
    revision_count: int             # how many loops so far
    final_recommendations: list     # output
    decision_log: list              # every step logged
    error: Optional[str]            # error message if any step fails
