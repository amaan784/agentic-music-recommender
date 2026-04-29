"""Streamlit UI for SoundScout AI."""

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from agent.graph import run_recommendation_pipeline
from agent.logger import format_log_for_display
from evaluation.report_generator import generate_report, format_report_for_display

st.set_page_config(
    page_title="SoundScout AI",
    page_icon="music",
    layout="wide",
)

st.markdown(
    """
    <style>
    .song-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border-left: 4px solid #e94560;
        color: #eee;
    }
    .song-title { font-size: 1.2em; font-weight: 700; color: #fff; margin-bottom: 4px; }
    .song-artist { font-size: 1em; color: #a0a0b0; margin-bottom: 8px; }
    .song-meta { font-size: 0.85em; color: #8888a0; }
    .conf-high { color: #4ade80; font-weight: 700; }
    .conf-mid { color: #facc15; font-weight: 700; }
    .conf-low { color: #f87171; font-weight: 700; }
    .explanation-box {
        background: #0f3460;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.9em;
        color: #ccd;
        line-height: 1.5;
    }
    .feature-bar-label { display: inline-block; width: 110px; font-size: 0.85em; color: #aaa; }
    .bias-ok { color: #4ade80; }
    .bias-flag { color: #f87171; }
    .bias-card {
        background: #16213e;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .bias-card-icon {
        font-size: 1.4em;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .bias-card-icon-ok { background: #0f3d1f; color: #4ade80; }
    .bias-card-icon-flag { background: #3d0f0f; color: #f87171; }
    .bias-card-name { font-weight: 600; color: #ddd; font-size: 0.95em; }
    .bias-card-detail { color: #8888a0; font-size: 0.85em; margin-top: 2px; }
    .step-card {
        background: #16213e;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border-left: 3px solid #e94560;
    }
    .step-header { display: flex; justify-content: space-between; align-items: center; }
    .step-name { font-weight: 600; color: #ddd; font-size: 0.95em; }
    .step-time { color: #8888a0; font-size: 0.8em; }
    .step-output { color: #aab; font-size: 0.85em; margin-top: 6px; }
    .step-notes { color: #667; font-size: 0.8em; margin-top: 4px; font-style: italic; }
    .stat-card {
        background: #16213e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stat-number { font-size: 2em; font-weight: 700; color: #e94560; }
    .stat-label { font-size: 0.9em; color: #8888a0; margin-top: 4px; }
    .llm-banner {
        background: linear-gradient(90deg, #0f3460 0%, #1a1a2e 100%);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .llm-banner-left { display: flex; align-items: center; gap: 12px; }
    .llm-dot-on { width: 10px; height: 10px; border-radius: 50%; background: #4ade80; display: inline-block; animation: pulse 1.5s infinite; }
    .llm-dot-off { width: 10px; height: 10px; border-radius: 50%; background: #666; display: inline-block; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
    .llm-label { color: #ddd; font-weight: 600; font-size: 0.95em; }
    .llm-detail { color: #8888a0; font-size: 0.85em; }
    .llm-mode { color: #e94560; font-weight: 700; font-size: 0.9em; }
    .pipeline-container {
        background: #0a0a15;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 18px;
        background: #16213e;
        border-radius: 10px;
        margin-bottom: 8px;
        border-left: 3px solid #333;
        transition: all 0.3s ease;
    }
    .pipeline-step.active { border-left-color: #e94560; background: #1a2744; }
    .pipeline-step.completed { border-left-color: #4ade80; }
    .pipeline-step-number {
        width: 28px; height: 28px;
        border-radius: 50%;
        background: #0f3460;
        color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85em; font-weight: 700;
    }
    .pipeline-step.completed .pipeline-step-number { background: #0f3d1f; color: #4ade80; }
    .pipeline-step.active .pipeline-step-number { background: #e94560; }
    .pipeline-step-content { flex: 1; }
    .pipeline-step-name { font-weight: 600; color: #ddd; font-size: 0.95em; }
    .pipeline-step-detail { color: #8888a0; font-size: 0.8em; margin-top: 2px; }
    .pipeline-arrow {
        text-align: center;
        color: #444;
        font-size: 1.2em;
        margin: -4px 0;
        padding-left: 40px;
    }
    .intermediate-results {
        background: #0f1a2e;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        border: 1px solid #1e3a5f;
    }
    .intermediate-title {
        color: #e94560;
        font-weight: 600;
        font-size: 1em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .query-box {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: monospace;
        color: #4ade80;
        font-size: 0.9em;
        border-left: 3px solid #4ade80;
    }
    .candidate-list {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }
    .candidate-chip {
        background: #1e3a5f;
        color: #aab;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8em;
    }
    .guardrail-result {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: #1a1a2e;
        border-radius: 6px;
        margin-bottom: 6px;
    }
    .revision-badge {
        background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# SoundScout AI")
st.markdown("Find your next favorite tracks from 114K real Spotify songs using AI-powered recommendations with bias detection and self-critique.")

llm_provider = os.getenv("LLM_PROVIDER", "openai")
llm_model = os.getenv("LLM_MODEL", "") or {
    "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-20250514",
    "mistral": "mistral-large-latest", "gemini": "gemini-2.0-flash",
}.get(llm_provider, "gpt-4o-mini")
has_api_key = bool(
    os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("ANTHROPIC_API_KEY", "").strip()
    or os.getenv("MISTRAL_API_KEY", "").strip()
    or os.getenv("GOOGLE_API_KEY", "").strip()
)

with st.sidebar:
    st.markdown("""
        <div style="text-align:center; margin-bottom:20px;">
            <div style="font-size:2em; margin-bottom:5px;">🎵</div>
            <div style="font-size:1.3em; font-weight:700; color:#e94560;">SoundScout AI</div>
            <div style="font-size:0.85em; color:#8888a0; margin-top:4px;">Your Music Discovery Agent</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🤖 AI Configuration")

    llm_status_col1, llm_status_col2 = st.columns([1, 2])
    with llm_status_col1:
        if has_api_key:
            st.markdown("""<div style="background:#0f3d1f; border-radius:20px; padding:8px 12px; text-align:center;">
                <span style="color:#4ade80; font-size:0.85em; font-weight:600;">● LLM Ready</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#3d0f0f; border-radius:20px; padding:8px 12px; text-align:center;">
                <span style="color:#f87171; font-size:0.85em; font-weight:600;">○ No API Key</span>
            </div>""", unsafe_allow_html=True)

    with llm_status_col2:
        if has_api_key:
            st.caption(f"Using **{llm_provider.title()}**")
        else:
            st.caption("Rule-based mode only")

    if has_api_key:
        llm_mode = st.segmented_control(
            "LLM Mode",
            options=["Full AI", "Critique Only", "Rule-Based"],
            default="Full AI",
        )
        use_llm_explanations = llm_mode == "Full AI"
        skip_llm = llm_mode == "Rule-Based"
    else:
        st.info("💡 Add an API key to `.env` for AI-powered critique and explanations")
        use_llm_explanations = False
        skip_llm = True

    st.markdown("---")

    st.markdown("### 🎧 What do you want to hear?")

    genres = st.multiselect(
        "🎼 Genres",
        options=[
            "pop", "rock", "hip-hop", "r-n-b", "jazz", "classical", "electronic",
            "indie", "country", "latin", "metal", "folk", "blues", "soul",
            "reggae", "punk", "alternative", "ambient", "dance", "disco",
        ],
        default=["pop", "rock"],
        help="Select multiple genres to explore",
    )

    st.markdown("#### Vibe & Energy")

    vibe_cols = st.columns(3)
    with vibe_cols[0]:
        mood = st.selectbox(
            "😊 Mood",
            options=["melancholic", "neutral", "upbeat"],
            index=1,
        )
    with vibe_cols[1]:
        energy = st.selectbox(
            "⚡ Energy",
            options=["low-energy", "moderate", "high-energy"],
            index=1,
        )
    with vibe_cols[2]:
        danceability = st.selectbox(
            "💃 Danceability",
            options=["low", "medium", "high"],
            index=1,
        )

    st.markdown("#### 📝 Describe your vibe")
    additional = st.text_area(
        "Free-form description",
        placeholder="e.g., songs like Radiohead, atmospheric, late-night listening, workout pump-up...",
        height=80,
        help="Add any specific artists, moods, or scenarios you're looking for",
        label_visibility="collapsed",
    )

    with st.expander("🔧 Fine-tune Audio Features", expanded=False):
        st.caption("Adjust these for more precise control")

        feat_cols = st.columns(2)
        with feat_cols[0]:
            tempo = st.slider("🥁 Tempo", 0.0, 1.0, 0.5, 0.05,
                help="Slow (0) to Fast (1)")
        with feat_cols[1]:
            acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.5, 0.05,
                help="Electronic (0) to Acoustic (1)")

    st.markdown("---")

    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "tracks_clean.csv")
        df = pd.read_csv(data_path)
        stat_cols = st.columns(3)
        stat_cols[0].metric("🎵 Tracks", f"{len(df):,}", delta=None)
        stat_cols[1].metric("🎼 Genres", df["track_genre"].nunique(), delta=None)
        stat_cols[2].metric("🎤 Artists", f"{df['artists'].nunique():,}", delta=None)
    except Exception:
        pass

    st.markdown("---")

    st.markdown("""
        <div style="background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border-radius:12px; padding:15px; margin-bottom:15px;
                    border-left:4px solid #e94560;">
            <div style="color:#8888a0; font-size:0.85em; margin-bottom:8px;">
                Ready to discover your next favorite tracks?
            </div>
        </div>
    """, unsafe_allow_html=True)

    run_button = st.button(
        "🚀 Get Recommendations",
        type="primary",
        use_container_width=True,
    )


def _conf_class(score):
    if score >= 0.8:
        return "conf-high"
    elif score >= 0.6:
        return "conf-mid"
    return "conf-low"


def _conf_label(score):
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    return "Low"


def _feature_bar(label, value):
    pct = int(value * 100)
    return f'<span class="feature-bar-label">{label}</span> <progress value="{pct}" max="100" style="width:60%;height:8px;"></progress> {pct}%'


if run_button:
    user_preferences = {
        "genres": genres,
        "mood": mood,
        "energy": energy,
        "danceability": danceability,
        "tempo": tempo,
        "acousticness": acousticness,
        "additional": additional,
    }

    active_llm = not skip_llm and has_api_key
    use_llm_exp = use_llm_explanations and active_llm

    if active_llm:
        dot_cls = "llm-dot-on"
        mode_text = "Full AI Mode"
        detail = f"{llm_provider.title()} / {llm_model}"
        if use_llm_exp:
            detail += " (critique + explanations)"
        else:
            detail += " (critique only, template explanations)"
    else:
        dot_cls = "llm-dot-off"
        mode_text = "Rule-Based Mode"
        detail = "No LLM calls. Using scoring rules and template explanations."

    st.markdown(
        f'<div class="llm-banner">'
        f'<div class="llm-banner-left">'
        f'<span class="{dot_cls}"></span>'
        f'<span class="llm-label">AI Status</span>'
        f'<span class="llm-detail">{detail}</span>'
        f'</div>'
        f'<span class="llm-mode">{mode_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    os.environ["USE_LLM_EXPLANATIONS"] = str(use_llm_exp).lower()
    if skip_llm:
        os.environ["LLM_PROVIDER"] = "disabled"

    pipeline_container = st.empty()
    intermediate_container = st.container()

    with pipeline_container.container():
        st.markdown("### 🤖 Agent Pipeline Execution")
        st.caption("Watch the agent work through each step of the recommendation process")

        # Visual pipeline steps
        steps_html = """
        <div class="pipeline-container">
            <div class="pipeline-step active" id="step-1">
                <div class="pipeline-step-number">1</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">Parse Input</div>
                    <div class="pipeline-step-detail">Validating preferences...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-2">
                <div class="pipeline-step-number">2</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">Build Query</div>
                    <div class="pipeline-step-detail">Converting to search query...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-3">
                <div class="pipeline-step-number">3</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">RAG Retrieval</div>
                    <div class="pipeline-step-detail">Searching ChromaDB vector store...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-4">
                <div class="pipeline-step-number">4</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">Apply Guardrails</div>
                    <div class="pipeline-step-detail">Filtering low-quality tracks...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-5">
                <div class="pipeline-step-number">5</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">Score & Rank</div>
                    <div class="pipeline-step-detail">Computing audio feature similarity...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-6">
                <div class="pipeline-step-number">6</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">Bias Detection</div>
                    <div class="pipeline-step-detail">Running fairness checks...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-7">
                <div class="pipeline-step-number">7</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">Compute Confidence</div>
                    <div class="pipeline-step-detail">Scoring prediction quality...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-8">
                <div class="pipeline-step-number">8</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">LLM Critique</div>
                    <div class="pipeline-step-detail">Evaluating recommendations...</div>
                </div>
            </div>
            <div class="pipeline-arrow">↓</div>
            <div class="pipeline-step" id="step-9">
                <div class="pipeline-step-number">✓</div>
                <div class="pipeline-step-content">
                    <div class="pipeline-step-name">Finalize</div>
                    <div class="pipeline-step-detail">Generating explanations...</div>
                </div>
            </div>
        </div>
        """
        st.markdown(steps_html, unsafe_allow_html=True)

    # Run pipeline
    result = run_recommendation_pipeline(user_preferences)

    if skip_llm:
        os.environ["LLM_PROVIDER"] = llm_provider

    # Extract intermediate data from decision log
    decision_log = result.get("decision_log", [])
    step_data = {entry.get("step", ""): entry for entry in decision_log}

    with intermediate_container:
        st.markdown("---")
        st.markdown("### 🔍 Open Process: What Happened Behind the Scenes")

        # Build Query section
        with st.expander("📋 1. Query Construction", expanded=True):
            query_step = step_data.get("build_query", {})
            query_text = query_step.get("output_summary", "").replace("Query: '", "").replace("'", "")
            st.markdown("**Your preferences were converted to this search query:**")
            st.markdown(f'<div class="query-box">{query_text}</div>', unsafe_allow_html=True)
            st.caption(f"Time: {query_step.get('duration_ms', 0):.0f}ms")

        # Retrieval section
        with st.expander("🔎 2. RAG Vector Retrieval", expanded=True):
            retrieve_step = step_data.get("retrieve", {})
            st.markdown("**Vector search in ChromaDB:**")
            retrieve_summary = retrieve_step.get("output_summary", "")
            st.info(retrieve_summary)

            # Show sample candidates if available
            if "candidates" in str(retrieve_summary).lower():
                num_candidates = int(''.join(filter(str.isdigit, retrieve_summary.split("candidates")[1].split(",")[0])) if "candidates" in retrieve_summary else 30
                st.markdown(f"**Retrieved {num_candidates} candidate tracks** from 114K song catalog")
                st.caption(f"Search time: {retrieve_step.get('duration_ms', 0):.0f}ms")

        # Guardrails section
        with st.expander("🛡️ 3. Quality Guardrails", expanded=True):
            guardrail_step = step_data.get("apply_guardrails", {})
            guardrail_summary = guardrail_step.get("output_summary", "")
            st.markdown("**Filtering applied:**")
            st.markdown(f'<div class="guardrail-result">✓ {guardrail_summary}</div>', unsafe_allow_html=True)
            notes = guardrail_step.get("notes", "")
            if notes and notes != "All guardrails passed":
                st.warning(f"⚠️ {notes}")
            st.caption(f"Filter time: {guardrail_step.get('duration_ms', 0):.0f}ms")

        # Scoring section
        with st.expander("📊 4. Audio Feature Scoring", expanded=True):
            score_step = step_data.get("score", {})
            st.markdown("**Scoring method:** Cosine similarity between user preference vector and track audio features")
            st.markdown(f"**{score_step.get('output_summary', '')}**")
            st.caption(f"Scoring time: {score_step.get('duration_ms', 0):.0f}ms")

        # Bias detection section
        with st.expander("⚖️ 5. Bias Detection Analysis", expanded=True):
            bias_step = step_data.get("check_bias", {})
            bias_summary = bias_step.get("output_summary", "")
            bias_notes = bias_step.get("notes", "")
            st.markdown(f"**{bias_summary}**")
            if bias_notes:
                if "No bias" in bias_notes or "No flags" in bias_notes:
                    st.success(f"✓ {bias_notes}")
                else:
                    st.warning(f"⚠️ {bias_notes}")
            st.caption(f"Analysis time: {bias_step.get('duration_ms', 0):.0f}ms")

        # Confidence section
        with st.expander("🎯 6. Confidence Scoring", expanded=True):
            conf_step = step_data.get("compute_confidence", {})
            st.markdown(f"**{conf_step.get('output_summary', '')}**")
            st.markdown("Signals used:")
            st.markdown("- Feature match: Audio alignment with preferences")
            st.markdown("- Retrieval relevance: Vector search similarity")
            st.markdown("- Score margin: Gap from next-best tracks")
            st.markdown("- Diversity contribution: Uniqueness in set")
            st.caption(f"Confidence computation: {conf_step.get('duration_ms', 0):.0f}ms")

        # Critique section
        with st.expander("🤖 7. LLM Self-Critique", expanded=True):
            critique_step = step_data.get("critique", {})
            critique_summary = critique_step.get("output_summary", "")
            st.markdown(f"**{critique_summary}**")

            critique_notes = critique_step.get("notes", "")
            if critique_notes:
                st.markdown(f"*Assessment: {critique_notes}*")

            if "should_revise: True" in critique_summary or "Should revise: true" in critique_summary:
                st.warning("🔄 Agent decided to revise recommendations based on critique")
            else:
                st.success("✓ Agent approved recommendations for finalization")

            st.caption(f"Critique time: {critique_step.get('duration_ms', 0):.0f}ms")

        # Revisions section (if any)
        revision_count = result.get("revision_count", 0)
        if revision_count > 0:
            with st.expander("🔄 Revision Loop", expanded=True):
                st.markdown(f"<span class='revision-badge'>🔄 {revision_count} Revision(s) Made</span>", unsafe_allow_html=True)
                st.markdown("The agent detected issues and re-ran the pipeline with adjusted weights:")

                # Show revision steps
                revision_steps = [entry for entry in decision_log if entry.get("step") == "revise_weights"]
                for i, rev_step in enumerate(revision_steps, 1):
                    st.markdown(f"**Revision #{i}:** {rev_step.get('output_summary', '')}")
                    st.caption(f"Adjustment time: {rev_step.get('duration_ms', 0):.0f}ms")

        st.markdown("---")

    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
    else:
        final_recs = result.get("final_recommendations", [])
        revision_count = result.get("revision_count", 0)

        if revision_count > 0:
            st.info(f"The agent self-critiqued and revised its recommendations {revision_count} time(s) using {llm_provider.title()} {llm_model}.")

        st.markdown(f"### Your Top {len(final_recs)} Tracks")

        for i, rec in enumerate(final_recs, 1):
            meta = rec.get("metadata", {})
            conf = rec.get("confidence", {})
            overall_conf = conf.get("overall_confidence", 0)
            components = conf.get("components", {})
            explanation = rec.get("explanation", "")

            conf_cls = _conf_class(overall_conf)
            conf_lbl = _conf_label(overall_conf)

            card_html = f"""
            <div class="song-card">
                <div class="song-title">{i}. {meta.get('track_name', 'Unknown')}</div>
                <div class="song-artist">{meta.get('artists', 'Unknown')}</div>
                <div class="song-meta">
                    {meta.get('track_genre', 'unknown').title()} &middot;
                    {meta.get('album_name', '')} &middot;
                    Popularity: {meta.get('popularity', 'N/A')} &middot;
                    Confidence: <span class="{conf_cls}">{overall_conf:.0%} ({conf_lbl})</span>
                </div>
            """
            if explanation:
                card_html += f'<div class="explanation-box">{explanation}</div>'
            card_html += "</div>"

            st.markdown(card_html, unsafe_allow_html=True)

            with st.expander(f"Details for {meta.get('track_name', 'Track')}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Audio Profile**")
                    for feat_name, feat_key in [("Energy", "energy"), ("Valence (Happiness)", "valence"), ("Danceability", "danceability"), ("Tempo", "tempo"), ("Acousticness", "acousticness")]:
                        val = float(meta.get(feat_key, 0))
                        st.markdown(_feature_bar(feat_name, val), unsafe_allow_html=True)

                with col_b:
                    st.markdown("**Why this confidence score?**")
                    for comp_name, comp_key in [("Feature Match", "feature_match"), ("Retrieval Relevance", "retrieval_relevance"), ("Score Margin", "margin"), ("Diversity Boost", "bias_contribution")]:
                        val = components.get(comp_key, 0)
                        st.markdown(_feature_bar(comp_name, val), unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### How Fair Are These Results?")
        bias_report = result.get("bias_report", {})

        if bias_report:
            try:
                catalog_df = pd.read_csv(
                    os.path.join(os.path.dirname(__file__), "data", "tracks_clean.csv")
                )
                catalog = catalog_df.sample(min(1000, len(catalog_df))).to_dict("records")
            except Exception:
                catalog = []

            report = generate_report(final_recs, catalog)
            display = format_report_for_display(report)

            quality = display["summary"].get("Overall Quality", "Unknown")
            quality_colors = {"Excellent": "#4ade80", "Good": "#86efac", "Fair": "#facc15", "Poor": "#f87171"}
            q_color = quality_colors.get(quality, "#aaa")

            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{q_color}">{quality}</div><div class="stat-label">Overall Quality</div></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="stat-card"><div class="stat-number">{display["summary"].get("Diversity Score", "0")}</div><div class="stat-label">Genre Diversity</div></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="stat-card"><div class="stat-number">{display["summary"].get("Novelty Score", "0")}</div><div class="stat-label">Novelty</div></div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="stat-card"><div class="stat-number">{display["summary"].get("Bias Flags", "0")}</div><div class="stat-label">Bias Flags</div></div>', unsafe_allow_html=True)

            st.markdown("")
            for check in display.get("bias_details", []):
                status = check.get("Status", "")
                is_ok = status == "OK"
                icon_cls = "bias-card-icon-ok" if is_ok else "bias-card-icon-flag"
                icon_text = "OK" if is_ok else "!!"
                name = check.get("Check", "").replace("_", " ").title()
                detail = check.get("Detail", "")
                st.markdown(
                    f'<div class="bias-card">'
                    f'<div class="bias-card-icon {icon_cls}">{icon_text}</div>'
                    f'<div><div class="bias-card-name">{name}</div>'
                    f'<div class="bias-card-detail">{detail}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        st.markdown("### Agent Decision Log")
        decision_log = result.get("decision_log", [])

        if decision_log:
            st.caption(f"{len(decision_log)} steps | {revision_count} revision(s)")

            step_icons = {
                "parse_input": "1", "build_query": "2", "retrieve": "3",
                "apply_guardrails": "4", "score": "5", "check_bias": "6",
                "compute_confidence": "7", "critique": "8",
                "revise_weights": "R", "finalize": "F", "error": "X",
            }
            for entry in decision_log:
                step = entry.get("step", "")
                dur = entry.get("duration_ms", 0)
                out = entry.get("output_summary", "")
                notes = entry.get("notes", "")
                icon = step_icons.get(step, "?")
                step_label = step.replace("_", " ").title()
                notes_html = f'<div class="step-notes">{notes}</div>' if notes else ""
                st.markdown(
                    f'<div class="step-card">'
                    f'<div class="step-header">'
                    f'<span class="step-name">Step {icon}: {step_label}</span>'
                    f'<span class="step-time">{dur:.0f}ms</span>'
                    f'</div>'
                    f'<div class="step-output">{out}</div>'
                    f'{notes_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

else:
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "tracks_clean.csv")
        df = pd.read_csv(data_path)

        st.markdown("")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="stat-card"><div class="stat-number">{len(df):,}</div><div class="stat-label">Tracks in Catalog</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="stat-card"><div class="stat-number">{df["track_genre"].nunique()}</div><div class="stat-label">Genres</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="stat-card"><div class="stat-number">{df["artists"].nunique():,}</div><div class="stat-label">Artists</div></div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown("Pick your genres, mood, and energy in the sidebar, then hit **Get Recommendations**.")
    except Exception:
        st.info("Run `python data/prepare_data.py` first to build the track catalog.")
