"""Core scoring engine using cosine similarity."""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "speechiness", "instrumentalness", "liveness", "loudness",
]


def build_user_vector(user_preferences: Dict[str, Any]) -> np.ndarray:
    """Map user preferences to a numeric feature vector."""
    mood_map = {"melancholic": 0.2, "neutral": 0.5, "upbeat": 0.8}
    energy_map = {"low-energy": 0.2, "moderate": 0.5, "high-energy": 0.8}
    dance_map = {"low": 0.2, "medium": 0.5, "high": 0.8}

    valence = user_preferences.get("valence", None)
    if valence is None:
        mood = user_preferences.get("mood", "neutral")
        valence = mood_map.get(mood, 0.5)

    energy = user_preferences.get("energy_value", None)
    if energy is None:
        energy_str = user_preferences.get("energy", "moderate")
        energy = energy_map.get(energy_str, 0.5)

    danceability = user_preferences.get("danceability_value", None)
    if danceability is None:
        dance_str = user_preferences.get("danceability", "medium")
        danceability = dance_map.get(dance_str, 0.5)

    vector = np.array([
        danceability,
        energy,
        valence,
        user_preferences.get("tempo", 0.5),
        user_preferences.get("acousticness", 0.5),
        user_preferences.get("speechiness", 0.3),
        user_preferences.get("instrumentalness", 0.3),
        user_preferences.get("liveness", 0.3),
        user_preferences.get("loudness", 0.5),
    ])

    return vector.reshape(1, -1)


def build_track_vector(track_meta: Dict[str, Any]) -> np.ndarray:
    """Extract audio feature vector from track metadata."""
    vector = np.array([float(track_meta.get(f, 0.5)) for f in AUDIO_FEATURES])
    return vector.reshape(1, -1)


def score_tracks(
    candidates: List[Dict[str, Any]],
    user_preferences: Dict[str, Any],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Score candidates against user preferences. Returns top_n by score."""
    user_vec = build_user_vector(user_preferences)
    scored = []

    for candidate in candidates:
        meta = candidate.get("metadata", {})
        track_vec = build_track_vector(meta)
        sim = float(cosine_similarity(user_vec, track_vec)[0][0])

        scored_candidate = dict(candidate)
        scored_candidate["score"] = sim
        scored.append(scored_candidate)

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_n]


def adjust_weights(
    critique: Dict[str, Any],
    current_preferences: Dict[str, Any],
) -> Dict[str, Any]:
    """Shift preference weights based on critique feedback."""
    adjusted = dict(current_preferences)
    adjustments = critique.get("adjustments", [])

    for adj in adjustments:
        action = adj.get("action", "")
        if action == "boost_genre":
            genres = adjusted.get("genres", [])
            detail = adj.get("detail", "")
            if "genre" in detail.lower():
                adjusted["_genre_boost"] = True

        elif action == "reweight":
            adjusted["_diversity_boost"] = True
            for feat in ["danceability_value", "energy_value", "valence"]:
                if feat in adjusted:
                    current = adjusted[feat]
                    adjusted[feat] = current * 0.8 + 0.5 * 0.2

    return adjusted
