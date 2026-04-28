"""Per-recommendation confidence scoring."""

from typing import List, Dict, Any
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

AUDIO_FEATURES = ["danceability", "energy", "valence", "tempo", "acousticness"]

WEIGHTS = {
    "feature_match": 0.35,
    "retrieval_relevance": 0.25,
    "margin": 0.20,
    "bias_contribution": 0.20,
}


def compute_feature_match(
    track_meta: Dict[str, Any],
    user_preferences: Dict[str, Any],
) -> float:
    """Cosine similarity between track features and user preferences."""
    track_vec = np.array([float(track_meta.get(f, 0.5)) for f in AUDIO_FEATURES]).reshape(1, -1)
    pref_vec = np.array([float(user_preferences.get(f, 0.5)) for f in AUDIO_FEATURES]).reshape(1, -1)

    sim = cosine_similarity(track_vec, pref_vec)[0][0]
    return float(max(0.0, min(1.0, sim)))


def compute_margin(score: float, all_scores: List[float]) -> float:
    """Normalized gap between this track's score and the median."""
    if not all_scores or len(all_scores) < 2:
        return 0.5

    median = float(np.median(all_scores))
    max_gap = max(all_scores) - min(all_scores)

    if max_gap == 0:
        return 0.5

    margin = (score - median) / max_gap
    return float(max(0.0, min(1.0, (margin + 1) / 2)))


def compute_bias_contribution(
    track_meta: Dict[str, Any],
    genre_counts: Dict[str, int],
    total_recs: int,
) -> float:
    """Score how much this track contributes to genre diversity."""
    genre = track_meta.get("track_genre", "unknown")
    if total_recs == 0:
        return 0.5

    genre_ratio = genre_counts.get(genre, 0) / total_recs
    return float(max(0.0, 1.0 - genre_ratio))


def score_recommendation(
    track: Dict[str, Any],
    user_preferences: Dict[str, Any],
    all_relevance_scores: List[float],
    genre_counts: Dict[str, int],
    total_recs: int,
) -> Dict[str, Any]:
    """Compute overall confidence for one recommendation."""
    meta = track.get("metadata", {})

    feature_match = compute_feature_match(meta, user_preferences)
    retrieval_relevance = float(track.get("relevance_score", 0.5))
    margin = compute_margin(retrieval_relevance, all_relevance_scores)
    bias_contrib = compute_bias_contribution(meta, genre_counts, total_recs)

    overall = (
        WEIGHTS["feature_match"] * feature_match
        + WEIGHTS["retrieval_relevance"] * retrieval_relevance
        + WEIGHTS["margin"] * margin
        + WEIGHTS["bias_contribution"] * bias_contrib
    )
    overall = float(max(0.0, min(1.0, overall)))

    return {
        "overall_confidence": overall,
        "components": {
            "feature_match": round(feature_match, 4),
            "retrieval_relevance": round(retrieval_relevance, 4),
            "margin": round(margin, 4),
            "bias_contribution": round(bias_contrib, 4),
        },
    }


def score_all_recommendations(
    tracks: List[Dict[str, Any]],
    user_preferences: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Score all recommendations and sort by confidence descending."""
    all_relevance = [t.get("relevance_score", 0.5) for t in tracks]
    genre_counts = Counter(
        t.get("metadata", {}).get("track_genre", "unknown") for t in tracks
    )
    total = len(tracks)

    scored_tracks = []
    for track in tracks:
        confidence = score_recommendation(
            track, user_preferences, all_relevance, genre_counts, total
        )
        scored_track = dict(track)
        scored_track["confidence"] = confidence
        scored_tracks.append(scored_track)

    scored_tracks.sort(key=lambda x: x["confidence"]["overall_confidence"], reverse=True)
    return scored_tracks
