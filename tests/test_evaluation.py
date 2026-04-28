"""Tests for the evaluation module."""

import pytest
from evaluation.bias_detector import (
    genre_concentration,
    popularity_bias,
    artist_repetition,
    mood_homogeneity,
    demographic_proxy_check,
    run_all_checks,
)
from evaluation.metrics import (
    diversity_score,
    coverage,
    novelty_score,
    fairness_ratio,
    intra_list_similarity,
)


def _make_rec(name, genre, popularity=50, valence=0.5, explicit=False, **extra):
    meta = {
        "track_id": f"id_{name}",
        "track_name": name,
        "artists": f"artist_{name}",
        "track_genre": genre,
        "popularity": popularity,
        "valence": valence,
        "explicit": explicit,
        "danceability": 0.5,
        "energy": 0.5,
        "tempo": 0.5,
        "acousticness": 0.5,
        "speechiness": 0.3,
        "instrumentalness": 0.1,
        "liveness": 0.2,
        "loudness": 0.5,
    }
    meta.update(extra)
    return {"metadata": meta, "relevance_score": 0.8}


def _make_catalog_entry(genre, popularity=50, explicit=False):
    return {"track_genre": genre, "popularity": popularity, "explicit": explicit}


class TestBiasDetector:
    def test_genre_concentration_flags_single_genre_recs(self):
        recs = [_make_rec(f"t{i}", "rock") for i in range(10)]
        catalog = [_make_catalog_entry(g) for g in ["rock", "pop", "jazz", "hip-hop", "classical"] * 20]
        result = genre_concentration(recs, catalog)
        assert result["flagged"]

    def test_popularity_bias_detects_skew(self):
        recs = [_make_rec(f"t{i}", "pop", popularity=90) for i in range(10)]
        catalog = [_make_catalog_entry("pop", popularity=30) for _ in range(100)]
        result = popularity_bias(recs, catalog)
        assert result["flagged"]

    def test_artist_repetition_detects_repeats(self):
        recs = [_make_rec("t1", "rock")] * 5  # same track 5 times
        recs[0]["metadata"]["artists"] = "SameArtist"
        for r in recs:
            r["metadata"]["artists"] = "SameArtist"
        result = artist_repetition(recs)
        assert result["flagged"]

    def test_mood_homogeneity_flags_same_mood(self):
        recs = [_make_rec(f"t{i}", "pop", valence=0.5) for i in range(10)]
        result = mood_homogeneity(recs, min_std=0.1)
        assert result["flagged"]

    def test_mood_homogeneity_passes_diverse(self):
        recs = [_make_rec(f"t{i}", "pop", valence=i * 0.1) for i in range(10)]
        result = mood_homogeneity(recs, min_std=0.1)
        assert not result["flagged"]

    def test_run_all_checks_returns_summary(self):
        recs = [_make_rec(f"t{i}", "rock") for i in range(10)]
        catalog = [_make_catalog_entry("rock") for _ in range(100)]
        report = run_all_checks(recs, catalog)
        assert "bias_checks" in report
        assert "any_flagged" in report
        assert isinstance(report["total_flags"], int)


class TestMetrics:
    def test_diversity_score_zero_for_single_genre(self):
        recs = [_make_rec(f"t{i}", "rock") for i in range(10)]
        assert diversity_score(recs) == 0.0

    def test_diversity_score_high_for_mixed_genres(self):
        genres = ["rock", "pop", "jazz", "hip-hop", "classical", "electronic", "indie", "country", "folk", "blues"]
        recs = [_make_rec(f"t{i}", genres[i]) for i in range(10)]
        score = diversity_score(recs)
        assert score > 2.0

    def test_coverage_counts_unique_tracks(self):
        run1 = [_make_rec("t1", "rock"), _make_rec("t2", "pop")]
        run2 = [_make_rec("t2", "pop"), _make_rec("t3", "jazz")]
        cov = coverage([run1, run2], catalog_size=100)
        assert cov == 3 / 100

    def test_novelty_score_high_for_unpopular(self):
        recs = [_make_rec(f"t{i}", "pop", popularity=5) for i in range(5)]
        catalog = [_make_catalog_entry("pop", popularity=100)]
        score = novelty_score(recs, catalog)
        assert score > 0.9

    def test_fairness_ratio_perfect_for_matching(self):
        recs = [_make_rec("t1", "rock"), _make_rec("t2", "pop")]
        catalog = [_make_catalog_entry("rock"), _make_catalog_entry("pop")]
        ratios = fairness_ratio(recs, catalog)
        assert abs(ratios.get("rock", 0) - 1.0) < 0.01
        assert abs(ratios.get("pop", 0) - 1.0) < 0.01

    def test_intra_list_similarity_zero_for_single(self):
        recs = [_make_rec("t1", "rock")]
        assert intra_list_similarity(recs) == 0.0

    def test_intra_list_similarity_returns_value(self):
        recs = [_make_rec(f"t{i}", "rock") for i in range(5)]
        sim = intra_list_similarity(recs)
        assert 0.0 <= sim <= 1.0
