"""Tests for the confidence module."""

import pytest
from confidence.scorer import (
    compute_feature_match,
    compute_margin,
    compute_bias_contribution,
    score_recommendation,
    score_all_recommendations,
)
from confidence.critic import rule_based_critique
from confidence.explainer import template_explanation


def _make_track(name="Track", genre="rock", relevance=0.8, **features):
    defaults = {
        "danceability": 0.5,
        "energy": 0.5,
        "valence": 0.5,
        "tempo": 0.5,
        "acousticness": 0.5,
        "track_name": name,
        "artists": "TestArtist",
        "track_genre": genre,
        "popularity": 50,
        "mood_descriptor": "neutral",
        "energy_descriptor": "moderate",
    }
    defaults.update(features)
    return {
        "content": f"{name} by TestArtist",
        "metadata": defaults,
        "relevance_score": relevance,
    }


class TestScorer:
    def test_perfect_feature_match_high_confidence(self):
        track = _make_track(energy=0.8, valence=0.8, danceability=0.8)
        prefs = {"energy": 0.8, "valence": 0.8, "danceability": 0.8, "tempo": 0.5, "acousticness": 0.5}
        match = compute_feature_match(track["metadata"], prefs)
        assert match > 0.9

    def test_zero_feature_match_low_confidence(self):
        track = _make_track(energy=1.0, valence=1.0, danceability=1.0, tempo=1.0, acousticness=1.0)
        prefs = {"energy": 0.0, "valence": 0.0, "danceability": 0.0, "tempo": 0.0, "acousticness": 0.0}
        match = compute_feature_match(track["metadata"], prefs)
        assert match < 0.3

    def test_margin_high_for_top_scorer(self):
        margin = compute_margin(0.95, [0.3, 0.5, 0.7, 0.95])
        assert margin > 0.5

    def test_margin_low_for_bottom_scorer(self):
        margin = compute_margin(0.3, [0.3, 0.5, 0.7, 0.95])
        assert margin < 0.5

    def test_bias_contribution_high_for_rare_genre(self):
        genre_counts = {"rock": 8, "jazz": 1}
        score = compute_bias_contribution({"track_genre": "jazz"}, genre_counts, 10)
        assert score > 0.8

    def test_bias_contribution_low_for_common_genre(self):
        genre_counts = {"rock": 8, "jazz": 1}
        score = compute_bias_contribution({"track_genre": "rock"}, genre_counts, 10)
        assert score < 0.3

    def test_score_all_returns_sorted(self):
        tracks = [
            _make_track("Low", "pop", relevance=0.2, energy=0.1),
            _make_track("High", "rock", relevance=0.9, energy=0.8),
        ]
        prefs = {"energy_value": 0.8, "mood": "upbeat", "genres": ["rock"]}
        scored = score_all_recommendations(tracks, prefs)
        assert scored[0]["metadata"]["track_name"] == "High"
        assert "confidence" in scored[0]


class TestCritic:
    def test_rule_based_critique_identifies_poor_fits(self):
        tracks = [
            _make_track("Bad", "pop", relevance=0.2),
            _make_track("Good", "rock", relevance=0.9),
        ]
        tracks[0]["confidence"] = {"overall_confidence": 0.2, "components": {}}
        tracks[1]["confidence"] = {"overall_confidence": 0.9, "components": {}}

        bias_report = {"bias_checks": [], "total_flags": 0}
        critique = rule_based_critique(tracks, bias_report)
        assert "Bad" in critique["poor_fits"]

    def test_rule_based_critique_triggers_revision_on_low_avg(self):
        tracks = [
            _make_track(f"t{i}", "pop") for i in range(5)
        ]
        for t in tracks:
            t["confidence"] = {"overall_confidence": 0.3, "components": {}}

        bias_report = {"bias_checks": [], "total_flags": 0}
        critique = rule_based_critique(tracks, bias_report)
        assert critique["should_revise"]

    def test_max_revision_cap_respected(self):
        tracks = [_make_track("t1", "pop")]
        tracks[0]["confidence"] = {"overall_confidence": 0.9, "components": {}}
        bias_report = {"bias_checks": [], "total_flags": 0}
        critique = rule_based_critique(tracks, bias_report)
        assert not critique["should_revise"]


class TestExplainer:
    def test_template_explanation_produces_output(self):
        track = _make_track("Song", "rock", energy=0.8, valence=0.7)
        track["confidence"] = {
            "overall_confidence": 0.85,
            "components": {"feature_match": 0.9, "retrieval_relevance": 0.8, "margin": 0.7, "bias_contribution": 0.8},
        }
        prefs = {"genres": ["rock"], "mood": "upbeat", "energy_value": 0.75}
        explanation = template_explanation(track, prefs)
        assert len(explanation) > 20
        assert "Confidence" in explanation

    def test_template_explanation_handles_no_match(self):
        track = _make_track("Song", "unknown-genre")
        track["confidence"] = {
            "overall_confidence": 0.4,
            "components": {"feature_match": 0.3, "retrieval_relevance": 0.3, "margin": 0.3, "bias_contribution": 0.3},
        }
        prefs = {"genres": ["pop"], "mood": "melancholic"}
        explanation = template_explanation(track, prefs)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
