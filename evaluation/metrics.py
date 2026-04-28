"""Evaluation metrics: diversity, coverage, novelty, fairness, intra-list similarity."""

from typing import List, Dict, Any
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


AUDIO_FEATURES = ["danceability", "energy", "valence", "tempo", "acousticness", "speechiness", "instrumentalness", "liveness", "loudness"]


def diversity_score(recs: List[Dict[str, Any]]) -> float:
    """Shannon entropy across genres in the recommendation list."""
    genres = [r.get("metadata", {}).get("track_genre", "unknown") for r in recs]
    if not genres:
        return 0.0

    counts = Counter(genres)
    total = len(genres)
    probs = [c / total for c in counts.values()]

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return float(entropy)


def coverage(all_runs: List[List[Dict[str, Any]]], catalog_size: int) -> float:
    """
    What percentage of catalog has been recommended across N user profiles.
    """
    if catalog_size == 0:
        return 0.0

    recommended_ids = set()
    for run in all_runs:
        for r in run:
            track_id = r.get("metadata", {}).get("track_id", "")
            if track_id:
                recommended_ids.add(track_id)

    return len(recommended_ids) / catalog_size


def novelty_score(recs: List[Dict[str, Any]], catalog: List[Dict[str, Any]]) -> float:
    """
    Average inverse popularity of recommended tracks.
    Higher = more novel (recommending less popular tracks).
    """
    if not recs:
        return 0.0

    max_pop = max(
        (c.get("popularity", 0) for c in catalog), default=100
    )
    if max_pop == 0:
        max_pop = 100

    scores = []
    for r in recs:
        pop = r.get("metadata", {}).get("popularity", 0)
        scores.append(1.0 - (pop / max_pop))

    return float(np.mean(scores))


def fairness_ratio(
    recs: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Per-genre: (% in recs) / (% in catalog). Perfect fairness = 1.0 for all genres.
    """
    rec_genres = [r.get("metadata", {}).get("track_genre", "unknown") for r in recs]
    cat_genres = [c.get("track_genre", "unknown") for c in catalog]

    rec_counts = Counter(rec_genres)
    cat_counts = Counter(cat_genres)

    rec_total = max(len(rec_genres), 1)
    cat_total = max(len(cat_genres), 1)

    ratios = {}
    for genre in set(list(rec_counts.keys()) + list(cat_counts.keys())):
        rec_pct = rec_counts.get(genre, 0) / rec_total
        cat_pct = cat_counts.get(genre, 0) / cat_total
        if cat_pct > 0:
            ratios[genre] = round(rec_pct / cat_pct, 4)
        else:
            ratios[genre] = float("inf") if rec_pct > 0 else 0.0

    return ratios


def intra_list_similarity(recs: List[Dict[str, Any]]) -> float:
    """Average pairwise cosine similarity of audio features across recs."""
    if len(recs) < 2:
        return 0.0

    vectors = []
    for r in recs:
        meta = r.get("metadata", {})
        vec = [float(meta.get(f, 0.0)) for f in AUDIO_FEATURES]
        vectors.append(vec)

    matrix = np.array(vectors)
    sim_matrix = cosine_similarity(matrix)

    n = len(recs)
    upper_indices = np.triu_indices(n, k=1)
    avg_sim = float(np.mean(sim_matrix[upper_indices]))

    return avg_sim


def compute_all_metrics(
    recs: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
    all_runs: List[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics and return a summary dict."""
    if all_runs is None:
        all_runs = [recs]

    catalog_size = len(catalog) if catalog else 1

    return {
        "diversity_score": diversity_score(recs),
        "coverage": coverage(all_runs, catalog_size),
        "novelty_score": novelty_score(recs, catalog),
        "fairness_ratio": fairness_ratio(recs, catalog),
        "intra_list_similarity": intra_list_similarity(recs),
    }
