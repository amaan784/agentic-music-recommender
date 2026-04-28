"""Bias detection for recommendation outputs."""

from typing import List, Dict, Any
from collections import Counter
import numpy as np


def genre_concentration(
    recs: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """KL divergence between rec and catalog genre distributions."""
    rec_genres = [r.get("metadata", {}).get("track_genre", "unknown") for r in recs]
    cat_genres = [c.get("track_genre", "unknown") for c in catalog]

    all_genres = set(rec_genres + cat_genres)

    rec_counts = Counter(rec_genres)
    cat_counts = Counter(cat_genres)

    rec_total = max(len(rec_genres), 1)
    cat_total = max(len(cat_genres), 1)

    kl_div = 0.0
    epsilon = 1e-10
    for genre in all_genres:
        p = rec_counts.get(genre, 0) / rec_total + epsilon
        q = cat_counts.get(genre, 0) / cat_total + epsilon
        kl_div += p * np.log(p / q)

    flagged = kl_div > threshold
    return {
        "metric": "genre_concentration",
        "kl_divergence": float(kl_div),
        "threshold": threshold,
        "flagged": flagged,
        "rec_distribution": dict(rec_counts),
        "detail": f"KL divergence = {kl_div:.4f} ({'FLAGGED' if flagged else 'OK'})",
    }


def popularity_bias(
    recs: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare rec vs catalog popularity and compute Gini coefficient."""
    rec_pop = [r.get("metadata", {}).get("popularity", 0) for r in recs]
    cat_pop = [c.get("popularity", 0) for c in catalog]

    rec_mean = float(np.mean(rec_pop)) if rec_pop else 0.0
    cat_mean = float(np.mean(cat_pop)) if cat_pop else 0.0

    gini = _gini_coefficient(rec_pop)

    flagged = abs(rec_mean - cat_mean) > 20

    return {
        "metric": "popularity_bias",
        "rec_mean_popularity": rec_mean,
        "catalog_mean_popularity": cat_mean,
        "popularity_gap": rec_mean - cat_mean,
        "gini_coefficient": gini,
        "flagged": flagged,
        "detail": f"Rec mean={rec_mean:.1f}, Catalog mean={cat_mean:.1f}, Gini={gini:.3f}",
    }


def _gini_coefficient(values: List[float]) -> float:
    """Compute the Gini coefficient for a list of values."""
    if not values or len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    return float((2 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * np.sum(sorted_vals))) - (n + 1) / n)


def artist_repetition(
    recs: List[Dict[str, Any]],
    max_allowed: int = 2,
) -> Dict[str, Any]:
    """Flag if any artist appears more than max_allowed times in a top-10 list."""
    artists = [r.get("metadata", {}).get("artists", "unknown") for r in recs]
    counts = Counter(artists)
    repeated = {a: c for a, c in counts.items() if c > max_allowed}

    return {
        "metric": "artist_repetition",
        "repeated_artists": repeated,
        "flagged": len(repeated) > 0,
        "detail": f"{len(repeated)} artists repeated >{max_allowed} times" if repeated else "No excessive repetition",
    }


def mood_homogeneity(
    recs: List[Dict[str, Any]],
    min_std: float = 0.1,
) -> Dict[str, Any]:
    """Flag low valence variance (all same mood)."""
    valences = [r.get("metadata", {}).get("valence", 0.5) for r in recs]
    std = float(np.std(valences)) if valences else 0.0

    return {
        "metric": "mood_homogeneity",
        "valence_std": std,
        "min_std_threshold": min_std,
        "flagged": std < min_std,
        "detail": f"Valence std={std:.4f} ({'FLAGGED: too homogeneous' if std < min_std else 'OK'})",
    }


def demographic_proxy_check(
    recs: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
    tolerance: float = 0.15,
) -> Dict[str, Any]:
    """
    Check if explicit content ratio in recs significantly differs from catalog baseline.
    """
    rec_explicit = [r.get("metadata", {}).get("explicit", False) for r in recs]
    cat_explicit = [c.get("explicit", False) for c in catalog]

    rec_ratio = sum(rec_explicit) / max(len(rec_explicit), 1)
    cat_ratio = sum(cat_explicit) / max(len(cat_explicit), 1)
    diff = abs(rec_ratio - cat_ratio)

    return {
        "metric": "demographic_proxy_check",
        "rec_explicit_ratio": float(rec_ratio),
        "catalog_explicit_ratio": float(cat_ratio),
        "difference": float(diff),
        "flagged": diff > tolerance,
        "detail": f"Explicit ratio: recs={rec_ratio:.2%}, catalog={cat_ratio:.2%}, diff={diff:.2%}",
    }


def run_all_checks(
    recs: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all bias detection checks and return a consolidated report."""
    checks = [
        genre_concentration(recs, catalog),
        popularity_bias(recs, catalog),
        artist_repetition(recs),
        mood_homogeneity(recs),
        demographic_proxy_check(recs, catalog),
    ]

    any_flagged = any(c["flagged"] for c in checks)

    return {
        "bias_checks": checks,
        "any_flagged": any_flagged,
        "total_flags": sum(1 for c in checks if c["flagged"]),
        "summary": f"{sum(1 for c in checks if c['flagged'])}/{len(checks)} bias checks flagged",
    }
