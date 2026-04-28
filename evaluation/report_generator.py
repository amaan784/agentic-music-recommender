"""Evaluation report generator."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List

from evaluation.bias_detector import run_all_checks
from evaluation.metrics import compute_all_metrics

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def generate_report(
    recs: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
    all_runs: List[List[Dict[str, Any]]] = None,
    run_id: str = None,
) -> Dict[str, Any]:
    """
    Generate a full evaluation report combining bias checks and metrics.
    """
    bias_report = run_all_checks(recs, catalog)
    metrics = compute_all_metrics(recs, catalog, all_runs)

    report = {
        "run_id": run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_recommendations": len(recs),
        "bias_report": bias_report,
        "metrics": metrics,
        "overall_quality": _assess_quality(bias_report, metrics),
    }

    return report


def _assess_quality(bias_report: Dict, metrics: Dict) -> str:
    """Simple quality assessment based on bias flags and diversity."""
    flags = bias_report.get("total_flags", 0)
    diversity = metrics.get("diversity_score", 0)

    if flags == 0 and diversity > 2.0:
        return "excellent"
    elif flags <= 1 and diversity > 1.5:
        return "good"
    elif flags <= 2:
        return "fair"
    else:
        return "poor"


def save_report(report: Dict[str, Any], logs_dir: str = LOGS_DIR) -> str:
    """Save evaluation report to logs directory as JSON. Returns file path."""
    os.makedirs(logs_dir, exist_ok=True)
    filename = f"eval_{report.get('run_id', 'unknown')}.json"
    filepath = os.path.join(logs_dir, filename)

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return filepath


def format_report_for_display(report: Dict[str, Any]) -> Dict[str, Any]:
    """Format report data for Streamlit rendering."""
    bias = report.get("bias_report", {})
    metrics = report.get("metrics", {})

    display = {
        "summary": {
            "Overall Quality": report.get("overall_quality", "unknown").title(),
            "Bias Flags": f"{bias.get('total_flags', 0)}/5",
            "Diversity Score": f"{metrics.get('diversity_score', 0):.2f}",
            "Novelty Score": f"{metrics.get('novelty_score', 0):.2f}",
            "Intra-list Similarity": f"{metrics.get('intra_list_similarity', 0):.2f}",
        },
        "bias_details": [
            {
                "Check": check.get("metric", ""),
                "Status": "FLAGGED" if check.get("flagged") else "OK",
                "Detail": check.get("detail", ""),
            }
            for check in bias.get("bias_checks", [])
        ],
        "fairness_ratio": metrics.get("fairness_ratio", {}),
    }

    return display
