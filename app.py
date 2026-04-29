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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# SoundScout AI")
st.markdown("Find your next favorite tracks from 114K real Spotify songs using AI-powered recommendations with bias detection and self-critique.")

with st.sidebar:
    st.markdown("### What do you want to hear?")

    genres = st.multiselect(
        "Genres",
        options=[
            "pop", "rock", "hip-hop", "r-n-b", "jazz", "classical", "electronic",
            "indie", "country", "latin", "metal", "folk", "blues", "soul",
            "reggae", "punk", "alternative", "ambient", "dance", "disco",
        ],
        default=["pop", "rock"],
    )

    mood = st.select_slider(
        "Mood",
        options=["melancholic", "neutral", "upbeat"],
        value="neutral",
    )

    energy = st.select_slider(
        "Energy",
        options=["low-energy", "moderate", "high-energy"],
        value="moderate",
    )

    danceability = st.select_slider(
        "Danceability",
        options=["low", "medium", "high"],
        value="medium",
    )

    with st.expander("Advanced settings"):
        tempo = st.slider("Tempo", 0.0, 1.0, 0.5, 0.05)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.05)
        additional = st.text_area(
            "Describe what you're looking for",
            placeholder="e.g., songs like Radiohead, atmospheric, good for late-night listening",
        )
        use_llm_explanations = st.checkbox("Use LLM for explanations (slower)", value=False)

    run_button = st.button("Get Recommendations", type="primary", width="stretch")


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

    with st.spinner("Running recommendation pipeline..."):
        os.environ["USE_LLM_EXPLANATIONS"] = str(use_llm_explanations).lower()
        result = run_recommendation_pipeline(user_preferences)

    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
    else:
        final_recs = result.get("final_recommendations", [])
        revision_count = result.get("revision_count", 0)

        if revision_count > 0:
            st.warning(f"The agent revised its recommendations {revision_count} time(s) after self-critique.")

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
