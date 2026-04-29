"""Self-critique loop using LLM or rule-based fallback."""

import os
import json
import time
from typing import Dict, Any, List, Optional

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

CRITIQUE_PROMPT = """You are a music recommendation critic. Given this list of recommended songs
with their confidence scores and a bias analysis, identify:
1. Any songs that seem like poor fits (confidence < 0.4)
2. Any systematic bias (genre/mood/popularity skew)
3. Suggest specific adjustments (boost underrepresented genres, swap low-confidence songs)

Recommendations:
{recommendations}

Confidence Scores:
{confidence_scores}

Bias Report:
{bias_report}

Return your analysis as JSON with the following structure:
{{
    "poor_fits": [list of track names that should be replaced],
    "bias_issues": [list of identified bias issues],
    "adjustments": [
        {{
            "action": "boost_genre" | "swap_track" | "reweight",
            "detail": "description of adjustment"
        }}
    ],
    "should_revise": true/false,
    "overall_assessment": "brief assessment"
}}"""


PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "mistral": "mistral-large-latest",
    "gemini": "gemini-2.0-flash",
}


LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))


def get_llm():
    """Return the configured LLM for critique."""
    model = LLM_MODEL or PROVIDER_DEFAULTS.get(LLM_PROVIDER, "gpt-4o-mini")

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0.1, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    elif LLM_PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model, temperature=0.1, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0.1, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.1, request_timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)


def format_recommendations_for_critique(tracks: List[Dict[str, Any]]) -> str:
    """Format tracks into a readable string for the LLM prompt."""
    lines = []
    for i, t in enumerate(tracks, 1):
        meta = t.get("metadata", {})
        conf = t.get("confidence", {}).get("overall_confidence", 0)
        lines.append(
            f"{i}. {meta.get('track_name', 'Unknown')} by {meta.get('artists', 'Unknown')} "
            f"[{meta.get('track_genre', 'unknown')}] confidence: {conf:.2f}"
        )
    return "\n".join(lines)


def format_confidence_for_critique(tracks: List[Dict[str, Any]]) -> str:
    """Format confidence breakdowns for the LLM prompt."""
    lines = []
    for t in tracks:
        meta = t.get("metadata", {})
        conf = t.get("confidence", {})
        components = conf.get("components", {})
        lines.append(
            f"- {meta.get('track_name', 'Unknown')}: "
            f"feature={components.get('feature_match', 0):.2f}, "
            f"relevance={components.get('retrieval_relevance', 0):.2f}, "
            f"margin={components.get('margin', 0):.2f}, "
            f"bias={components.get('bias_contribution', 0):.2f}"
        )
    return "\n".join(lines)


def critique_recommendations(
    tracks: List[Dict[str, Any]],
    bias_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Critique via LLM with rule-based fallback."""
    try:
        llm = get_llm()
        prompt = CRITIQUE_PROMPT.format(
            recommendations=format_recommendations_for_critique(tracks),
            confidence_scores=format_confidence_for_critique(tracks),
            bias_report=json.dumps(bias_report, indent=2, default=str),
        )
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = llm.invoke(prompt)
                break
            except Exception as retry_err:
                if "rate" in str(retry_err).lower() and attempt < LLM_MAX_RETRIES:
                    time.sleep(2 ** attempt)
                    continue
                raise
        content = response.content if hasattr(response, "content") else str(response)

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            critique = json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            critique = {
                "poor_fits": [],
                "bias_issues": ["Could not parse LLM response"],
                "adjustments": [],
                "should_revise": False,
                "overall_assessment": content[:500],
                "parse_error": True,
            }
    except Exception as e:
        critique = rule_based_critique(tracks, bias_report)
        critique["llm_error"] = str(e)

    return critique


def rule_based_critique(
    tracks: List[Dict[str, Any]],
    bias_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Rule-based fallback critique when LLM is unavailable.
    """
    poor_fits = []
    for t in tracks:
        conf = t.get("confidence", {}).get("overall_confidence", 0)
        if conf < 0.4:
            poor_fits.append(t.get("metadata", {}).get("track_name", "Unknown"))

    bias_issues = []
    for check in bias_report.get("bias_checks", []):
        if check.get("flagged"):
            bias_issues.append(check.get("detail", check.get("metric", "")))

    adjustments = []
    if poor_fits:
        adjustments.append({
            "action": "swap_track",
            "detail": f"Replace low-confidence tracks: {', '.join(poor_fits[:3])}",
        })
    if bias_issues:
        adjustments.append({
            "action": "reweight",
            "detail": "Adjust scoring weights to address identified bias",
        })

    avg_conf = 0.0
    if tracks:
        avg_conf = sum(t.get("confidence", {}).get("overall_confidence", 0) for t in tracks) / len(tracks)

    should_revise = len(poor_fits) > 2 or avg_conf < 0.6

    return {
        "poor_fits": poor_fits,
        "bias_issues": bias_issues,
        "adjustments": adjustments,
        "should_revise": should_revise,
        "overall_assessment": f"Rule-based: {len(poor_fits)} poor fits, {len(bias_issues)} bias issues, avg confidence {avg_conf:.2f}",
        "is_fallback": True,
    }
