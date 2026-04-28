# Model Card: SoundScout AI

## 1. Model Name

SoundScout AI 1.0

## 2. Intended Use

This system suggests up to 10 songs from a 114K-track Spotify catalog based on a user's preferred genres, mood, energy level, danceability, tempo, and acousticness. It is a portfolio project built for the CodePath AI course. It is not intended for production use without further evaluation.

## 3. How It Works

The user picks genres, mood, energy, and other preferences in a Streamlit form. The system converts those into a text query and searches a ChromaDB vector store containing 114K embedded track descriptions. It pulls 30 candidates, filters them through four guardrails (genre cap, artist cap, relevance floor, metadata check), then ranks the survivors by cosine similarity between their audio features and the user's preference vector. Five bias checks compare the recommendation set against a catalog sample. Each track gets a confidence score from four signals: feature match, retrieval relevance, margin over median, and diversity contribution. An LLM (or rule-based fallback) critiques the list and can trigger up to three revision cycles if average confidence is below 0.6. The final list comes with per-song explanations, a bias report, and a full decision log.

## 4. Data

The dataset is the Kaggle Spotify Tracks Dataset with about 114,000 tracks across 125 genres. Each track has 20 features including danceability, energy, valence, tempo, acousticness, speechiness, instrumentalness, liveness, loudness, and popularity. The data skews toward popular Western music. Genres like K-pop, Afrobeats, and regional music are underrepresented. The dataset is a static snapshot and does not reflect new releases.

## 5. Strengths

The system works well when the user has clear genre and mood preferences. The guardrails prevent obvious failure modes like returning 10 songs by the same artist or all from one genre. The self-critique loop catches low-confidence lists that a single pass would miss. The rule-based fallback means the system still works without an LLM API key. Every agent decision is logged, so a reviewer can trace exactly why each song was recommended.

## 6. Limitations and Bias

**Limitations:**
- Content-based only, no collaborative filtering from listening history
- Cold start: no way to learn from user feedback within a session
- LLM critique quality varies by provider and model
- Audio features are Spotify's precomputed values, not independently verified
- Static dataset, no live catalog updates

**Detected biases:**
- Popularity bias: the system tends to recommend higher-popularity tracks because they appear more often in the vector store
- Genre concentration: even with the 50% guardrail, strong genre preferences still dominate results
- Mood homogeneity: without the critique loop, all 10 recommendations can have nearly identical valence
- Western music bias inherited from the dataset

**Mitigations:**
- 4 guardrails filter the worst cases before scoring
- 5 bias checks run on every recommendation set and are shown to the user
- The critique loop can trigger re-scoring with adjusted weights
- Diversity contribution makes up 20% of each track's confidence score

## 7. Evaluation

50 tests across 5 test files. 45 pass without external dependencies. The remaining 5 test the LangGraph agent and require `langgraph` to be installed.

| Test File | Tests | Pass |
|-----------|-------|------|
| test_rag.py | 10 | 10 |
| test_evaluation.py | 12 | 12 |
| test_confidence.py | 12 | 12 |
| test_recommender.py | 9 | 9 |
| test_agent.py | 7 | 7 (with langgraph) |

Tests cover: guardrail filtering (genre cap, artist cap, relevance floor, metadata), bias detection (single-genre flagging, popularity skew, mood homogeneity), metric correctness (diversity entropy, fairness ratio, novelty), confidence score bounds, critique trigger conditions, graph conditional routing, and the scoring engine.

## 8. Future Work

- Add collaborative filtering if user listening history becomes available
- Build an A/B testing harness to compare scoring strategies
- Add streaming UI updates during the agent loop
- Explore music-specific embeddings instead of general-purpose sentence transformers
- Expand the dataset to include non-Western catalogs

## 9. Personal Reflection

**What surprised me during testing:**
The confidence scorer bug was the most instructive. Two of my tests were passing despite using wrong dictionary keys (`energy_value` instead of `energy`). The function silently fell back to default values (0.5) for those features, which happened to produce passing cosine similarities by coincidence. I only caught this when a third test with more extreme values failed. It showed me that passing tests do not mean correct tests, and that default-value fallbacks can mask real bugs.

**AI collaboration during this project:**
I used AI assistance throughout development. One helpful instance: when structuring the LangGraph workflow, the AI suggested using a `should_revise` conditional edge that checks both the critique's `should_revise` flag and the average confidence threshold together. This was better than my initial plan of only checking the critique flag, because it prevents unnecessary revision cycles when confidence is already high.

One flawed instance: the AI initially generated the `get_llm()` function using `from langchain_community.chat_models import ChatAnthropic`, which is a deprecated import path. The correct import is `from langchain_anthropic import ChatAnthropic`. I caught this during dependency validation and fixed it. It reminded me to verify import paths against current package documentation rather than trusting generated code.

## 10. Potential Misuse and Prevention

This system could be misused to create filter bubbles by always reinforcing a user's existing preferences, never exposing them to new genres or artists. The bias detection and diversity scoring are designed to counteract this, but a bad actor could disable the guardrails and critique loop to maximize engagement over discovery.

The recommendation explanations could also be misleading if the LLM generates plausible-sounding but incorrect reasoning about why a song was picked. The template-based fallback mitigates this by only stating facts (feature values, genre match) rather than generating free-form text.

The system does not collect or store user data. All processing happens locally. The decision logs contain only track metadata and preference inputs, not personally identifiable information.
