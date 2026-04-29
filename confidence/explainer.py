"""Per-recommendation explanation generator."""

import os
import json
import time
from typing import Dict, Any, List

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

EXPLANATION_PROMPT = """Generate a brief, friendly explanation for why this song was recommended.

Song: {track_name} by {artists}
Genre: {genre}
Audio Features: energy={energy:.2f}, valence={valence:.2f}, danceability={danceability:.2f}, tempo={tempo:.2f}
User Preferences: {user_prefs}
Confidence Score: {confidence:.2f}
Confidence Breakdown: feature_match={feature_match:.2f}, retrieval_relevance={retrieval_relevance:.2f}

Write 1-2 sentences explaining the match. Be specific about which features align."""


PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "mistral": "mistral-large-latest",
    "gemini": "gemini-2.0-flash",
}


LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_CALL_DELAY = float(os.getenv("LLM_CALL_DELAY", "0.5"))


def get_llm():
    """Return the configured LLM for explanations."""
    model = LLM_MODEL or PROVIDER_DEFAULTS.get(LLM_PROVIDER, "gpt-4o-mini")

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0.3, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    elif LLM_PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model, temperature=0.3, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0.3, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.3, request_timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)


def template_explanation(
    track: Dict[str, Any],
    user_preferences: Dict[str, Any],
) -> str:
    """Template-based explanation (no LLM needed)."""
    meta = track.get("metadata", {})
    conf = track.get("confidence", {})
    components = conf.get("components", {})
    overall = conf.get("overall_confidence", 0)

    track_name = meta.get("track_name", "This track")
    artists = meta.get("artists", "Unknown Artist")
    genre = meta.get("track_genre", "unknown")
    energy = meta.get("energy", 0.5)
    valence = meta.get("valence", 0.5)
    danceability = meta.get("danceability", 0.5)

    reasons = []

    preferred_genres = user_preferences.get("genres", [])
    if genre.lower() in [g.lower() for g in preferred_genres]:
        reasons.append(f"matches your preference for {genre}")

    pref_energy = user_preferences.get("energy_value", 0.5)
    if abs(energy - pref_energy) < 0.2:
        reasons.append(f"similar energy level ({energy:.2f} vs your {pref_energy:.2f})")

    mood = meta.get("mood_descriptor", "neutral")
    pref_mood = user_preferences.get("mood", "")
    if mood and pref_mood and mood.lower() in pref_mood.lower():
        reasons.append(f"{mood} mood aligns with your preference")
    elif valence > 0.6:
        reasons.append("upbeat mood")
    elif valence < 0.3:
        reasons.append("melancholic mood")

    if danceability > 0.7:
        reasons.append("high danceability")

    if not reasons:
        reasons.append("strong overall feature match")

    if overall >= 0.8:
        conf_label = "high"
    elif overall >= 0.6:
        conf_label = "moderate"
    else:
        conf_label = "low"

    reason_str = ", ".join(reasons)
    explanation = (
        f"Recommended because: {reason_str}. "
        f"Confidence: {overall:.2f} ({conf_label})."
    )

    bias_contrib = components.get("bias_contribution", 0.5)
    if bias_contrib > 0.7:
        explanation += " This pick also improves genre diversity in your list."

    return explanation


def llm_explanation(
    track: Dict[str, Any],
    user_preferences: Dict[str, Any],
) -> str:
    """Generate an LLM-powered explanation for a recommendation."""
    meta = track.get("metadata", {})
    conf = track.get("confidence", {})
    components = conf.get("components", {})

    prompt = EXPLANATION_PROMPT.format(
        track_name=meta.get("track_name", "Unknown"),
        artists=meta.get("artists", "Unknown"),
        genre=meta.get("track_genre", "unknown"),
        energy=meta.get("energy", 0.5),
        valence=meta.get("valence", 0.5),
        danceability=meta.get("danceability", 0.5),
        tempo=meta.get("tempo", 0.5),
        user_prefs=json.dumps(user_preferences, default=str),
        confidence=conf.get("overall_confidence", 0),
        feature_match=components.get("feature_match", 0),
        retrieval_relevance=components.get("retrieval_relevance", 0),
    )

    try:
        llm = get_llm()
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = llm.invoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
            except Exception as retry_err:
                if "rate" in str(retry_err).lower() and attempt < LLM_MAX_RETRIES:
                    time.sleep(2 ** attempt)
                    continue
                raise
    except Exception:
        return template_explanation(track, user_preferences)


def explain_recommendations(
    tracks: List[Dict[str, Any]],
    user_preferences: Dict[str, Any],
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Attach explanations to all recommendations."""
    explained = []
    for idx, track in enumerate(tracks):
        t = dict(track)
        if use_llm:
            if idx > 0:
                time.sleep(LLM_CALL_DELAY)
            t["explanation"] = llm_explanation(track, user_preferences)
        else:
            t["explanation"] = template_explanation(track, user_preferences)
        explained.append(t)
    return explained
