"""LangGraph workflow definition for SoundScout AI."""

import pandas as pd
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from agent.state import RecommenderState
from agent.logger import create_log_entry, StepTimer, save_run_log
from rag.retriever import build_query, retrieve_candidates
from rag.guardrails import apply_all_guardrails
from recommender import score_tracks, adjust_weights
from evaluation.bias_detector import run_all_checks
from confidence.scorer import score_all_recommendations
from confidence.critic import critique_recommendations
from confidence.explainer import explain_recommendations

MAX_REVISIONS = 3
CONFIDENCE_THRESHOLD = 0.6

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLEAN_CSV = os.path.join(DATA_DIR, "tracks_clean.csv")


def _load_catalog_sample() -> list:
    """Load a sample of the catalog for bias detection."""
    try:
        df = pd.read_csv(CLEAN_CSV)
        return df.sample(min(1000, len(df))).to_dict("records")
    except Exception:
        return []


def parse_input(state: RecommenderState) -> dict:
    """Parse and validate user preferences."""
    with StepTimer() as timer:
        prefs = state.get("user_preferences", {})
        prefs.setdefault("genres", [])
        prefs.setdefault("mood", "neutral")
        prefs.setdefault("energy", "moderate")
        prefs.setdefault("danceability", "medium")
        prefs.setdefault("additional", "")

    log = create_log_entry(
        step="parse_input",
        input_summary=f"Raw preferences: {list(prefs.keys())}",
        output_summary=f"Genres: {prefs.get('genres')}, Mood: {prefs.get('mood')}, Energy: {prefs.get('energy')}",
        duration_ms=timer.duration_ms,
    )
    return {
        "user_preferences": prefs,
        "decision_log": state.get("decision_log", []) + [log],
    }


def build_query_node(state: RecommenderState) -> dict:
    """Build a natural language retrieval query from preferences."""
    with StepTimer() as timer:
        query = build_query(state["user_preferences"])

    log = create_log_entry(
        step="build_query",
        input_summary=f"Preferences: {state['user_preferences'].get('genres', [])}",
        output_summary=f"Query: '{query}'",
        duration_ms=timer.duration_ms,
    )
    return {
        "query": query,
        "decision_log": state.get("decision_log", []) + [log],
    }


def retrieve_node(state: RecommenderState) -> dict:
    """Query ChromaDB for candidate tracks."""
    with StepTimer() as timer:
        candidates = retrieve_candidates(state["user_preferences"], top_k=30)

    top_name = ""
    if candidates:
        top_name = candidates[0].get("metadata", {}).get("track_name", "Unknown")
        top_score = candidates[0].get("relevance_score", 0)

    log = create_log_entry(
        step="retrieve",
        input_summary=f"query='{state.get('query', '')}'",
        output_summary=f"Retrieved {len(candidates)} candidates, top: '{top_name}' ({top_score:.2f})" if candidates else "No candidates retrieved",
        duration_ms=timer.duration_ms,
        notes="All candidates passed minimum relevance threshold" if candidates else "Empty retrieval",
    )
    return {
        "retrieved_candidates": candidates,
        "decision_log": state.get("decision_log", []) + [log],
    }


def apply_guardrails_node(state: RecommenderState) -> dict:
    """Filter candidates through quality guardrails."""
    with StepTimer() as timer:
        candidates = state.get("retrieved_candidates", [])
        filtered, guardrail_results = apply_all_guardrails(candidates)

    total_removed = len(candidates) - len(filtered)
    log = create_log_entry(
        step="apply_guardrails",
        input_summary=f"{len(candidates)} candidates",
        output_summary=f"{len(filtered)} passed ({total_removed} removed)",
        duration_ms=timer.duration_ms,
        notes="; ".join(
            gr["result"]["reason"] for gr in guardrail_results if not gr["result"]["passed"]
        ) or "All guardrails passed",
    )
    return {
        "retrieved_candidates": filtered,
        "guardrail_results": guardrail_results,
        "decision_log": state.get("decision_log", []) + [log],
    }


def score_node(state: RecommenderState) -> dict:
    """Rank candidates by audio feature similarity."""
    with StepTimer() as timer:
        candidates = state.get("retrieved_candidates", [])
        scored = score_tracks(candidates, state["user_preferences"], top_n=10)

    top_name = scored[0].get("metadata", {}).get("track_name", "") if scored else ""
    top_score = scored[0].get("score", 0) if scored else 0

    log = create_log_entry(
        step="score",
        input_summary=f"{len(candidates)} candidates to score",
        output_summary=f"Top 10 selected, best: '{top_name}' (score={top_score:.3f})",
        duration_ms=timer.duration_ms,
    )
    return {
        "scored_recommendations": scored,
        "decision_log": state.get("decision_log", []) + [log],
    }


def check_bias_node(state: RecommenderState) -> dict:
    """Run bias checks against a catalog sample."""
    with StepTimer() as timer:
        recs = state.get("scored_recommendations", [])
        catalog = _load_catalog_sample()
        bias_report = run_all_checks(recs, catalog)

    log = create_log_entry(
        step="check_bias",
        input_summary=f"{len(recs)} recommendations",
        output_summary=bias_report.get("summary", "No report"),
        duration_ms=timer.duration_ms,
        notes="Flagged checks: " + ", ".join(
            c["metric"] for c in bias_report.get("bias_checks", []) if c.get("flagged")
        ) or "No bias flags",
    )
    return {
        "bias_report": bias_report,
        "decision_log": state.get("decision_log", []) + [log],
    }


def compute_confidence_node(state: RecommenderState) -> dict:
    """Assign confidence scores to each recommendation."""
    with StepTimer() as timer:
        recs = state.get("scored_recommendations", [])
        scored = score_all_recommendations(recs, state["user_preferences"])
        confidence_scores = [
            r.get("confidence", {}).get("overall_confidence", 0) for r in scored
        ]

    avg_conf = sum(confidence_scores) / max(len(confidence_scores), 1)
    log = create_log_entry(
        step="compute_confidence",
        input_summary=f"{len(recs)} recommendations",
        output_summary=f"Avg confidence: {avg_conf:.3f}, range: [{min(confidence_scores, default=0):.3f}, {max(confidence_scores, default=0):.3f}]",
        duration_ms=timer.duration_ms,
    )
    return {
        "scored_recommendations": scored,
        "confidence_scores": confidence_scores,
        "decision_log": state.get("decision_log", []) + [log],
    }


def critique_node(state: RecommenderState) -> dict:
    """Critique the recommendation list via LLM or rules."""
    with StepTimer() as timer:
        recs = state.get("scored_recommendations", [])
        bias_report = state.get("bias_report", {})
        critique = critique_recommendations(recs, bias_report)

    log = create_log_entry(
        step="critique",
        input_summary=f"{len(recs)} recs, {state.get('bias_report', {}).get('total_flags', 0)} bias flags",
        output_summary=f"Should revise: {critique.get('should_revise', False)}, Poor fits: {len(critique.get('poor_fits', []))}",
        duration_ms=timer.duration_ms,
        notes=critique.get("overall_assessment", "")[:200],
    )
    return {
        "critique": critique,
        "decision_log": state.get("decision_log", []) + [log],
    }


def should_revise(state: RecommenderState) -> str:
    """Decide whether to revise or finalize."""
    critique = state.get("critique", {})
    revision_count = state.get("revision_count", 0)
    avg_confidence = 0.0
    conf_scores = state.get("confidence_scores", [])
    if conf_scores:
        avg_confidence = sum(conf_scores) / len(conf_scores)

    if (
        critique.get("should_revise", False)
        and revision_count < MAX_REVISIONS
        and avg_confidence < CONFIDENCE_THRESHOLD
    ):
        return "revise"
    return "finalize"


def revise_weights_node(state: RecommenderState) -> dict:
    """Adjust scoring weights based on critique feedback."""
    with StepTimer() as timer:
        critique = state.get("critique", {})
        prefs = adjust_weights(critique, state["user_preferences"])
        revision_count = state.get("revision_count", 0) + 1

    log = create_log_entry(
        step="revise_weights",
        input_summary=f"Revision #{revision_count}",
        output_summary=f"Adjusted preferences for next iteration",
        duration_ms=timer.duration_ms,
        notes=f"Adjustments: {len(critique.get('adjustments', []))}",
    )
    return {
        "user_preferences": prefs,
        "revision_count": revision_count,
        "decision_log": state.get("decision_log", []) + [log],
    }


def finalize_node(state: RecommenderState) -> dict:
    """Attach explanations and persist the run log."""
    with StepTimer() as timer:
        recs = state.get("scored_recommendations", [])
        prefs = state.get("user_preferences", {})

        use_llm = os.getenv("USE_LLM_EXPLANATIONS", "false").lower() == "true"
        final = explain_recommendations(recs, prefs, use_llm=use_llm)

    decision_log = state.get("decision_log", [])
    log = create_log_entry(
        step="finalize",
        input_summary=f"{len(recs)} scored recommendations",
        output_summary=f"Final {len(final)} recommendations with explanations",
        duration_ms=timer.duration_ms,
        notes=f"Total revisions: {state.get('revision_count', 0)}",
    )
    final_log = decision_log + [log]

    save_run_log(final_log)

    return {
        "final_recommendations": final,
        "decision_log": final_log,
    }


def build_graph() -> StateGraph:
    """Compile the LangGraph recommendation workflow."""
    workflow = StateGraph(RecommenderState)

    workflow.add_node("parse_input", parse_input)
    workflow.add_node("build_query", build_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("apply_guardrails", apply_guardrails_node)
    workflow.add_node("score", score_node)
    workflow.add_node("check_bias", check_bias_node)
    workflow.add_node("compute_confidence", compute_confidence_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("revise_weights", revise_weights_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "build_query")
    workflow.add_edge("build_query", "retrieve")
    workflow.add_edge("retrieve", "apply_guardrails")
    workflow.add_edge("apply_guardrails", "score")
    workflow.add_edge("score", "check_bias")
    workflow.add_edge("check_bias", "compute_confidence")
    workflow.add_edge("compute_confidence", "critique")

    workflow.add_conditional_edges(
        "critique",
        should_revise,
        {
            "revise": "revise_weights",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("revise_weights", "retrieve")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def run_recommendation_pipeline(user_preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full recommendation pipeline and return final state."""
    graph = build_graph()

    initial_state = {
        "user_preferences": user_preferences,
        "query": "",
        "retrieved_candidates": [],
        "guardrail_results": [],
        "scored_recommendations": [],
        "bias_report": {},
        "confidence_scores": [],
        "critique": None,
        "revision_count": 0,
        "final_recommendations": [],
        "decision_log": [],
        "error": None,
    }

    try:
        result = graph.invoke(initial_state)
        return result
    except Exception as e:
        initial_state["error"] = str(e)
        initial_state["decision_log"].append(
            create_log_entry(
                step="error",
                input_summary="Pipeline execution",
                output_summary=f"Error: {str(e)}",
                notes="Pipeline failed, check logs for details",
            )
        )
        return initial_state
