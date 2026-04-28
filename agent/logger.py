"""Structured decision logging for the recommendation agent."""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def create_log_entry(
    step: str,
    input_summary: str,
    output_summary: str,
    notes: str = "",
    duration_ms: float = 0,
) -> Dict[str, Any]:
    """Create a structured log entry for an agent step."""
    return {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(duration_ms, 2),
        "input_summary": input_summary,
        "output_summary": output_summary,
        "notes": notes,
    }


class StepTimer:
    """Context manager to time agent steps."""

    def __init__(self):
        self.start_time = None
        self.duration_ms = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.duration_ms = (time.time() - self.start_time) * 1000


def save_run_log(decision_log: List[Dict[str, Any]], run_id: str = None) -> str:
    """Save the full decision log to a JSON file. Returns file path."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    run_id = run_id or str(uuid.uuid4())[:8]
    filename = f"run_{run_id}.json"
    filepath = os.path.join(LOGS_DIR, filename)

    log_data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_steps": len(decision_log),
        "steps": decision_log,
    }

    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    return filepath


def format_log_for_display(decision_log: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Format log entries for Streamlit display."""
    display_entries = []
    for entry in decision_log:
        display_entries.append({
            "Step": entry.get("step", ""),
            "Duration": f"{entry.get('duration_ms', 0):.0f}ms",
            "Input": entry.get("input_summary", ""),
            "Output": entry.get("output_summary", ""),
            "Notes": entry.get("notes", ""),
        })
    return display_entries
