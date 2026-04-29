"""Streamlit UI for SoundScout AI."""

import os
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

st.title("SoundScout AI")
st.markdown("**Agentic Music Recommendation System** powered by RAG, LangGraph, bias detection, and self-critique.")

with st.sidebar:
    st.header("Your Music Preferences")

    genres = st.multiselect(
        "Preferred Genres",
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
        "Energy Level",
        options=["low-energy", "moderate", "high-energy"],
        value="moderate",
    )

    danceability = st.select_slider(
        "Danceability",
        options=["low", "medium", "high"],
        value="medium",
    )

    st.divider()
    st.subheader("Fine-tune (optional)")

    tempo = st.slider("Tempo Preference", 0.0, 1.0, 0.5, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.05)

    additional = st.text_area(
        "Additional preferences (free text)",
        placeholder="e.g., 'songs like Radiohead, atmospheric, good for late-night listening'",
    )

    use_llm_explanations = st.checkbox("Use LLM for explanations (slower)", value=False)
    run_button = st.button("Get Recommendations", type="primary", width="stretch")

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

        st.header(f"Top {len(final_recs)} Recommendations")

        for i, rec in enumerate(final_recs, 1):
            meta = rec.get("metadata", {})
            conf = rec.get("confidence", {})
            overall_conf = conf.get("overall_confidence", 0)

            with st.container():
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.markdown(f"**{i}. {meta.get('track_name', 'Unknown')}** by {meta.get('artists', 'Unknown')}")
                    st.caption(f"Genre: {meta.get('track_genre', 'unknown')} | Album: {meta.get('album_name', 'N/A')}")
                with col2:
                    st.metric("Confidence", f"{overall_conf:.2f}")
                with col3:
                    st.metric("Popularity", meta.get("popularity", "N/A"))

                explanation = rec.get("explanation", "")
                if explanation:
                    st.info(explanation)

                with st.expander("Audio Features"):
                    features = {
                        "Energy": f"{meta.get('energy', 0):.2f}",
                        "Valence": f"{meta.get('valence', 0):.2f}",
                        "Danceability": f"{meta.get('danceability', 0):.2f}",
                        "Tempo": f"{meta.get('tempo', 0):.2f}",
                        "Acousticness": f"{meta.get('acousticness', 0):.2f}",
                    }
                    st.json(features)

                    components = conf.get("components", {})
                    if components:
                        st.markdown("**Confidence Breakdown:**")
                        st.json(components)

                st.divider()

        st.header("Evaluation Report")
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

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Summary")
                for key, val in display["summary"].items():
                    st.metric(key, val)

            with col2:
                st.subheader("Bias Checks")
                bias_df = pd.DataFrame(display["bias_details"])
                if not bias_df.empty:
                    st.dataframe(bias_df, width="stretch")

        st.header("Agent Decision Log")
        decision_log = result.get("decision_log", [])

        if decision_log:
            with st.expander("View Full Decision Trace", expanded=False):
                log_display = format_log_for_display(decision_log)
                log_df = pd.DataFrame(log_display)
                st.dataframe(log_df, width="stretch")

            st.caption(
                f"Total steps: {len(decision_log)} | "
                f"Revisions: {result.get('revision_count', 0)}"
            )

else:
    st.info("Configure your preferences in the sidebar and click **Get Recommendations** to start.")

    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "tracks_clean.csv")
        df = pd.read_csv(data_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tracks", f"{len(df):,}")
        col2.metric("Genres", df["track_genre"].nunique())
        col3.metric("Artists", df["artists"].nunique())
    except Exception:
        st.caption("Data not yet loaded. Run `python data/prepare_data.py` first.")
