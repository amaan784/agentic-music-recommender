"""Tests for the core scoring engine."""

import pytest
import numpy as np
from recommender import build_user_vector, build_track_vector, score_tracks, adjust_weights


def _make_candidate(name="Track", genre="rock", relevance=0.8, **features):
    defaults = {
        "danceability": 0.5,
        "energy": 0.5,
        "valence": 0.5,
        "tempo": 0.5,
        "acousticness": 0.5,
        "speechiness": 0.3,
        "instrumentalness": 0.1,
        "liveness": 0.2,
        "loudness": 0.5,
        "track_name": name,
        "artists": "TestArtist",
        "track_genre": genre,
    }
    defaults.update(features)
    return {
        "content": f"{name} by TestArtist",
        "metadata": defaults,
        "relevance_score": relevance,
    }


class TestBuildUserVector:
    def test_default_values(self):
        vec = build_user_vector({})
        assert vec.shape == (1, 9)
        assert all(v >= 0 and v <= 1 for v in vec[0])

    def test_mood_mapping(self):
        vec = build_user_vector({"mood": "upbeat"})
        assert vec[0][2] == 0.8

    def test_explicit_values_override(self):
        vec = build_user_vector({"energy_value": 0.9, "valence": 0.1})
        assert vec[0][1] == 0.9
        assert vec[0][2] == 0.1


class TestBuildTrackVector:
    def test_extracts_all_features(self):
        meta = {
            "danceability": 0.7,
            "energy": 0.8,
            "valence": 0.3,
            "tempo": 0.6,
            "acousticness": 0.2,
            "speechiness": 0.1,
            "instrumentalness": 0.0,
            "liveness": 0.4,
            "loudness": 0.5,
        }
        vec = build_track_vector(meta)
        assert vec.shape == (1, 9)
        assert vec[0][0] == 0.7


class TestScoreTracks:
    def test_returns_top_n(self):
        candidates = [_make_candidate(f"t{i}") for i in range(20)]
        prefs = {"mood": "neutral", "energy": "moderate"}
        result = score_tracks(candidates, prefs, top_n=10)
        assert len(result) == 10

    def test_sorted_by_score_desc(self):
        candidates = [
            _make_candidate("Low", energy=0.1, valence=0.1),
            _make_candidate("High", energy=0.8, valence=0.8),
        ]
        prefs = {"energy_value": 0.8, "mood": "upbeat"}
        result = score_tracks(candidates, prefs, top_n=2)
        assert result[0]["score"] >= result[1]["score"]

    def test_similar_tracks_score_high(self):
        candidates = [
            _make_candidate("Match", energy=0.75, valence=0.75, danceability=0.75),
        ]
        prefs = {"energy_value": 0.75, "valence": 0.75, "danceability_value": 0.75}
        result = score_tracks(candidates, prefs, top_n=1)
        assert result[0]["score"] > 0.8

    def test_empty_candidates(self):
        result = score_tracks([], {}, top_n=10)
        assert result == []


class TestAdjustWeights:
    def test_reweight_shifts_toward_center(self):
        prefs = {"energy_value": 0.9, "danceability_value": 0.1}
        critique = {"adjustments": [{"action": "reweight", "detail": "adjust for diversity"}]}
        adjusted = adjust_weights(critique, prefs)
        assert adjusted["energy_value"] < 0.9
        assert adjusted["danceability_value"] > 0.1

    def test_no_adjustments_returns_same(self):
        prefs = {"energy_value": 0.8}
        critique = {"adjustments": []}
        adjusted = adjust_weights(critique, prefs)
        assert adjusted["energy_value"] == 0.8
