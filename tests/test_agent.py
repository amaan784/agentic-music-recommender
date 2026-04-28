"""Tests for the agent module."""

import pytest
from unittest.mock import patch, MagicMock
from agent.state import RecommenderState
from agent.logger import create_log_entry, StepTimer, save_run_log, format_log_for_display
from agent.graph import should_revise, MAX_REVISIONS


class TestLogger:
    def test_create_log_entry_has_required_fields(self):
        entry = create_log_entry(
            step="test_step",
            input_summary="input",
            output_summary="output",
        )
        assert entry["step"] == "test_step"
        assert "timestamp" in entry
        assert "duration_ms" in entry

    def test_step_timer_measures_time(self):
        import time
        with StepTimer() as timer:
            time.sleep(0.01)
        assert timer.duration_ms > 5  # at least ~10ms

    def test_format_log_for_display(self):
        log = [create_log_entry("step1", "in", "out")]
        display = format_log_for_display(log)
        assert len(display) == 1
        assert display[0]["Step"] == "step1"

    def test_save_run_log_creates_file(self, tmp_path):
        log = [create_log_entry("step1", "in", "out")]
        path = save_run_log(log, run_id="test123")
        assert "test123" in path


class TestGraphConditional:
    def test_should_revise_returns_finalize_when_good(self):
        state = {
            "critique": {"should_revise": False},
            "revision_count": 0,
            "confidence_scores": [0.9, 0.8, 0.85],
        }
        assert should_revise(state) == "finalize"

    def test_should_revise_returns_revise_when_low_confidence(self):
        state = {
            "critique": {"should_revise": True},
            "revision_count": 0,
            "confidence_scores": [0.3, 0.2, 0.4],
        }
        assert should_revise(state) == "revise"

    def test_should_revise_respects_max_cap(self):
        state = {
            "critique": {"should_revise": True},
            "revision_count": MAX_REVISIONS,
            "confidence_scores": [0.3, 0.2, 0.4],
        }
        assert should_revise(state) == "finalize"

    def test_should_revise_finalize_high_confidence_despite_critique(self):
        state = {
            "critique": {"should_revise": True},
            "revision_count": 0,
            "confidence_scores": [0.9, 0.85, 0.88],
        }
        assert should_revise(state) == "finalize"


class TestState:
    def test_recommender_state_can_be_created(self):
        state: RecommenderState = {
            "user_preferences": {"genres": ["rock"]},
            "query": "",
            "retrieved_candidates": [],
            "guardrail_results": [],
            "scored_recommendations": [],
            "bias_report": {},
            "confidence_scores": [],
            "critique": None,
            "revision_count": 0,
            "final_recommendations": [],
            "decision_log": [],
            "error": None,
        }
        assert state["user_preferences"]["genres"] == ["rock"]
        assert state["revision_count"] == 0
