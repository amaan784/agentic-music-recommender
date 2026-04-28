"""Tests for the RAG module."""

import pytest
from rag.guardrails import (
    max_single_genre_ratio,
    max_single_artist,
    min_relevance_score,
    metadata_completeness,
    apply_all_guardrails,
)


def _make_track(name, artist, genre, relevance=0.8, **extra_meta):
    """Create a mock track dict."""
    meta = {
        "track_id": f"id_{name}",
        "track_name": name,
        "artists": artist,
        "track_genre": genre,
        "danceability": 0.7,
        "energy": 0.6,
        "valence": 0.5,
        "tempo": 0.5,
        "acousticness": 0.3,
        "popularity": 50,
    }
    meta.update(extra_meta)
    return {
        "content": f"{name} by {artist}",
        "metadata": meta,
        "relevance_score": relevance,
    }


class TestGuardrails:
    def test_genre_ratio_passes_balanced(self):
        tracks = [
            _make_track("A", "Art1", "rock"),
            _make_track("B", "Art2", "pop"),
            _make_track("C", "Art3", "jazz"),
            _make_track("D", "Art4", "rock"),
        ]
        result = max_single_genre_ratio(tracks, max_ratio=0.5)
        assert result.passed

    def test_genre_ratio_fails_concentrated(self):
        tracks = [
            _make_track("A", "Art1", "rock"),
            _make_track("B", "Art2", "rock"),
            _make_track("C", "Art3", "rock"),
            _make_track("D", "Art4", "pop"),
        ]
        result = max_single_genre_ratio(tracks, max_ratio=0.5)
        assert not result.passed
        assert len(result.removed_tracks) > 0

    def test_artist_repetition_passes(self):
        tracks = [
            _make_track("A", "Art1", "rock"),
            _make_track("B", "Art2", "pop"),
            _make_track("C", "Art1", "jazz"),
        ]
        result = max_single_artist(tracks, max_count=2)
        assert result.passed

    def test_artist_repetition_fails(self):
        tracks = [
            _make_track("A", "Art1", "rock"),
            _make_track("B", "Art1", "pop"),
            _make_track("C", "Art1", "jazz"),
        ]
        result = max_single_artist(tracks, max_count=2)
        assert not result.passed

    def test_min_relevance_filters_low(self):
        tracks = [
            _make_track("A", "Art1", "rock", relevance=0.8),
            _make_track("B", "Art2", "pop", relevance=0.1),
        ]
        result = min_relevance_score(tracks, min_score=0.3)
        assert not result.passed
        assert len(result.removed_tracks) == 1

    def test_min_relevance_passes_all_high(self):
        tracks = [
            _make_track("A", "Art1", "rock", relevance=0.8),
            _make_track("B", "Art2", "pop", relevance=0.5),
        ]
        result = min_relevance_score(tracks, min_score=0.3)
        assert result.passed

    def test_metadata_completeness_passes(self):
        tracks = [_make_track("A", "Art1", "rock")]
        result = metadata_completeness(tracks)
        assert result.passed

    def test_metadata_completeness_fails_missing(self):
        track = _make_track("A", "Art1", "rock")
        del track["metadata"]["danceability"]
        result = metadata_completeness([track])
        assert not result.passed

    def test_empty_tracks_pass(self):
        result = max_single_genre_ratio([], max_ratio=0.5)
        assert result.passed

    def test_apply_all_guardrails(self):
        tracks = [
            _make_track("A", "Art1", "rock", relevance=0.8),
            _make_track("B", "Art2", "pop", relevance=0.5),
            _make_track("C", "Art3", "jazz", relevance=0.1),  # below threshold
        ]
        filtered, results = apply_all_guardrails(tracks)
        assert len(filtered) < len(tracks)
        assert len(results) == 4  # 4 guardrails applied
