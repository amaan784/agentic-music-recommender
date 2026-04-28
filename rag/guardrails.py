"""Retrieval quality guardrails."""

from typing import List, Dict, Any, NamedTuple
from collections import Counter


class GuardrailResult(NamedTuple):
    passed: bool
    reason: str
    removed_tracks: List[Dict[str, Any]]


def max_single_genre_ratio(
    tracks: List[Dict[str, Any]],
    max_ratio: float = 0.5,
) -> GuardrailResult:
    """Enforce max genre concentration. Removes excess tracks from over-represented genres."""
    if not tracks:
        return GuardrailResult(passed=True, reason="No tracks to check", removed_tracks=[])

    genre_counts = Counter(t["metadata"].get("track_genre", "unknown") for t in tracks)
    total = len(tracks)
    removed = []

    for genre, count in genre_counts.items():
        if count / total > max_ratio:
            max_allowed = int(total * max_ratio)
            genre_tracks = [t for t in tracks if t["metadata"].get("track_genre") == genre]
            genre_tracks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            excess = genre_tracks[max_allowed:]
            removed.extend(excess)

    if removed:
        return GuardrailResult(
            passed=False,
            reason=f"Genre concentration exceeded {max_ratio:.0%} threshold; removed {len(removed)} tracks",
            removed_tracks=removed,
        )
    return GuardrailResult(passed=True, reason="Genre distribution OK", removed_tracks=[])


def max_single_artist(
    tracks: List[Dict[str, Any]],
    max_count: int = 2,
) -> GuardrailResult:
    """No artist should appear more than max_count times in the result list."""
    if not tracks:
        return GuardrailResult(passed=True, reason="No tracks to check", removed_tracks=[])

    artist_counts = Counter(t["metadata"].get("artists", "unknown") for t in tracks)
    removed = []

    for artist, count in artist_counts.items():
        if count > max_count:
            artist_tracks = [t for t in tracks if t["metadata"].get("artists") == artist]
            artist_tracks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            excess = artist_tracks[max_count:]
            removed.extend(excess)

    if removed:
        return GuardrailResult(
            passed=False,
            reason=f"Artist repetition exceeded {max_count}; removed {len(removed)} tracks",
            removed_tracks=removed,
        )
    return GuardrailResult(passed=True, reason="Artist distribution OK", removed_tracks=[])


def min_relevance_score(
    tracks: List[Dict[str, Any]],
    min_score: float = 0.3,
) -> GuardrailResult:
    """Drop any retrieval result below the minimum relevance score."""
    removed = [t for t in tracks if t.get("relevance_score", 0) < min_score]

    if removed:
        return GuardrailResult(
            passed=False,
            reason=f"{len(removed)} tracks below relevance threshold {min_score}",
            removed_tracks=removed,
        )
    return GuardrailResult(passed=True, reason="All tracks meet relevance threshold", removed_tracks=[])


def metadata_completeness(
    tracks: List[Dict[str, Any]],
    required_fields: List[str] = None,
) -> GuardrailResult:
    """Skip tracks missing key audio features."""
    if required_fields is None:
        required_fields = [
            "danceability", "energy", "valence", "tempo",
            "acousticness", "track_genre"
        ]

    removed = []
    for t in tracks:
        meta = t.get("metadata", {})
        missing = [f for f in required_fields if f not in meta or meta[f] is None]
        if missing:
            removed.append(t)

    if removed:
        return GuardrailResult(
            passed=False,
            reason=f"{len(removed)} tracks missing required metadata fields",
            removed_tracks=removed,
        )
    return GuardrailResult(passed=True, reason="All tracks have complete metadata", removed_tracks=[])


def apply_all_guardrails(
    tracks: List[Dict[str, Any]],
    max_genre_ratio: float = 0.5,
    max_artist_count: int = 2,
    min_relevance: float = 0.3,
) -> tuple:
    """Apply all guardrails in sequence. Returns (filtered_tracks, guardrail_results)."""
    results = []
    current_tracks = list(tracks)

    gr = metadata_completeness(current_tracks)
    results.append({"guardrail": "metadata_completeness", "result": gr._asdict()})
    removed_set = {id(t) for t in gr.removed_tracks}
    current_tracks = [t for t in current_tracks if id(t) not in removed_set]

    gr = min_relevance_score(current_tracks, min_relevance)
    results.append({"guardrail": "min_relevance_score", "result": gr._asdict()})
    removed_set = {id(t) for t in gr.removed_tracks}
    current_tracks = [t for t in current_tracks if id(t) not in removed_set]

    gr = max_single_genre_ratio(current_tracks, max_genre_ratio)
    results.append({"guardrail": "max_single_genre_ratio", "result": gr._asdict()})
    removed_set = {id(t) for t in gr.removed_tracks}
    current_tracks = [t for t in current_tracks if id(t) not in removed_set]

    gr = max_single_artist(current_tracks, max_artist_count)
    results.append({"guardrail": "max_single_artist", "result": gr._asdict()})
    removed_set = {id(t) for t in gr.removed_tracks}
    current_tracks = [t for t in current_tracks if id(t) not in removed_set]

    return current_tracks, results
