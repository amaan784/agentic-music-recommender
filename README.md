# SoundScout AI

An end-to-end agentic RAG system that retrieves from 114K real Spotify tracks, orchestrates a multi-step recommendation pipeline with LangGraph, detects and corrects for popularity and genre bias, and self-critiques its own outputs with confidence scoring, all with full observability logging.

## Base Project

This project extends the [Music Recommender Simulation](https://github.com/amaan784/ai110-module3show-musicrecommendersimulation-starter) from Module 3. That starter project was a small content-based recommender with a hardcoded scoring rule (genre match +2, mood match +1, energy similarity bonus), a tiny CSV catalog of about 20 songs, and a few basic pytest tests. It had no retrieval, no agent loop, no bias detection, and no LLM integration.

SoundScout AI replaces every component: the small catalog becomes 114K real Spotify tracks in a ChromaDB vector store, the hardcoded scoring becomes cosine similarity over 9 audio features, and the single-pass pipeline becomes a LangGraph agent with conditional self-revision, bias detection, confidence scoring, and LLM-powered critique.

## Architecture

See the full diagram with data flow and testing coverage in [assets/architecture.md](assets/architecture.md).

```
STREAMLIT UI -> LANGGRAPH AGENT -> DATA LAYER
                  |
                  |-- Parse Input
                  |-- Build Query
                  |-- Retrieve (RAG / ChromaDB, 30 candidates)
                  |-- Apply Guardrails (4 filters)
                  |-- Score Candidates (cosine similarity, top 10)
                  |-- Bias Detection (5 checks against catalog)
                  |-- Confidence Scoring (4 weighted signals)
                  |-- LLM Critique (or rule-based fallback)
                  |-- [Conditional] Revise & Re-retrieve (max 3 loops)
                  |-- Finalize with Explanations
```

The user submits preferences through the Streamlit sidebar. The LangGraph agent runs 10 nodes in sequence, with a conditional loop between critique and retrieval. Each node logs its inputs, outputs, and duration. The final output includes ranked recommendations with per-song explanations, a bias report, evaluation metrics, and the full decision trace.

## Tech Stack

| Layer | Tool |
|-------|------|
| Agent Orchestration | LangGraph |
| RAG / Retrieval | LangChain + ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (default) or OpenAI |
| LLM (critique, explain) | GPT-4o-mini, Claude Sonnet, or Mistral Large |
| Data | Kaggle Spotify Tracks Dataset (114K tracks) |
| UI | Streamlit |
| Testing | pytest (50 tests) |

## Quick Start

### 1. Create a virtual environment (pick one)

**Option A: pip + venv**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\Activate           # Windows
pip install -r requirements.txt
```

**Option B: conda**
```bash
conda create -n soundscout python=3.12 -y
conda activate soundscout
pip install -r requirements.txt
```

**Option C: pipenv**
```bash
pip install pipenv
pipenv install -r requirements.txt
pipenv shell
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env with your API keys (optional, HuggingFace embeddings are free)
```

### 3. Download data
Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place `dataset.csv` as `data/spotify_tracks.csv`.

### 4. Prepare data & build vector store
```bash
python data/prepare_data.py
```

### 5. Run the app
```bash
python -m streamlit run app.py
```

### 6. Run tests
```bash
pytest tests/ -v
```

## Configuration

Set `LLM_PROVIDER` in `.env` to switch between LLM backends:

| Provider | Env Var | Default Model |
|----------|---------|---------------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| Mistral | `MISTRAL_API_KEY` | mistral-large-latest |

Set `LLM_MODEL` to override the default model for any provider. Set `EMBEDDING_PROVIDER` to `openai` or `huggingface` (default, free, no key needed).

## Sample Interactions

**Example 1: Upbeat indie rock fan**

Input: genres=["indie", "rock"], mood="upbeat", energy="high-energy", danceability="high"

Output (top 3 of 10):

| # | Track | Artist | Genre | Confidence |
|---|-------|--------|-------|------------|
| 1 | Mr. Brightside | The Killers | indie-rock | 0.91 |
| 2 | Take Me Out | Franz Ferdinand | indie-rock | 0.87 |
| 3 | Somebody Told Me | The Killers | indie-rock | 0.84 |

Explanation for #1: "Recommended because: matches your preference for indie-rock, similar energy level (0.82 vs your 0.80), upbeat mood. Confidence: 0.91 (high). This pick also improves genre diversity in your list."

Bias report: 1/5 checks flagged (genre concentration), triggering one revision cycle that swapped in tracks from adjacent genres.

**Example 2: Chill jazz listener**

Input: genres=["jazz"], mood="melancholic", energy="low-energy", danceability="low", acousticness=0.8

Output (top 3 of 10):

| # | Track | Artist | Genre | Confidence |
|---|-------|--------|-------|------------|
| 1 | Blue in Green | Miles Davis | jazz | 0.89 |
| 2 | Flamenco Sketches | Miles Davis | jazz | 0.85 |
| 3 | In a Sentimental Mood | Duke Ellington | jazz | 0.82 |

Explanation for #1: "Recommended because: matches your preference for jazz, similar energy level (0.18 vs your 0.20), melancholic mood aligns with your preference. Confidence: 0.89 (high)."

Bias report: 2/5 flagged (genre concentration, artist repetition). Critique triggered a revision that replaced one duplicate Miles Davis track with a Chet Baker track.

**Example 3: Mixed genre explorer**

Input: genres=["electronic", "classical", "hip-hop"], mood="neutral", energy="moderate", additional="good for studying"

Output: 10 tracks spread across all three genres with no single genre exceeding 40%. Average confidence 0.74. No bias flags. No revision needed. Decision log shows 10 steps completed in under 2 seconds.

## Design Decisions

Key trade-offs and rationale are documented in [reflection.md](reflection.md). The short version:

- **LangGraph over simpler chains** because the conditional revision loop requires cyclic state.
- **ChromaDB over cloud vector stores** for zero-cost, fully local reproducibility.
- **Rule-based fallback for critique** so the system works without an API key.
- **20% diversity weight in confidence scoring** to prevent the system from always recommending the same genre even when the user asks for it.

## Testing Summary

50 tests across 5 test files. 45 of 45 runnable tests pass (the remaining 5 in test_agent.py require `langgraph` to be installed). Tests cover guardrail filtering, bias detection flagging, evaluation metric correctness, confidence score ranges, critique logic, graph conditional routing, and the scoring engine.

The confidence scorer was the hardest to test because cosine similarity between feature vectors depends on which dict keys the function reads. Early tests used mismatched keys (`energy_value` vs `energy`) and passed by coincidence. Fixing the key mapping caught a real bug where the scorer was silently falling back to defaults for two out of five features.

## Reflection

See [reflection.md](reflection.md) for design decisions, challenges, and learnings. See [model_card.md](model_card.md) for bias analysis, limitations, evaluation, misuse considerations, and AI collaboration disclosure.

## Project Structure

```
soundscout-ai/
  assets/
    architecture.md     # Mermaid system diagram with data flow
  agent/
    state.py            # LangGraph state schema (RecommenderState)
    graph.py            # LangGraph workflow with 10 nodes + conditional revision
    logger.py           # Structured JSON logging per agent step
  confidence/
    scorer.py           # Multi-signal confidence scoring (feature, relevance, margin, bias)
    critic.py           # LLM self-critique with rule-based fallback
    explainer.py        # Per-song natural language explanations
  data/
    prepare_data.py     # CSV cleaning, normalization, ChromaDB ingestion
  evaluation/
    bias_detector.py    # Genre, popularity, artist, mood, demographic bias checks
    metrics.py          # Diversity, coverage, novelty, fairness, intra-list similarity
    report_generator.py # Evaluation report builder + JSON export
  rag/
    vectorstore.py      # ChromaDB setup, load, retrieve
    retriever.py        # Preference-to-query conversion + candidate retrieval
    guardrails.py       # Genre ratio, artist cap, relevance floor, metadata checks
  tests/
    test_rag.py         # 10 guardrail tests
    test_evaluation.py  # 12 bias + metrics tests
    test_confidence.py  # 12 scorer, critic, explainer tests
    test_agent.py       # 7 logger, graph conditional, state tests
    test_recommender.py # 9 scoring engine tests
  app.py                # Streamlit UI
  recommender.py        # Core cosine-similarity scoring engine
  model_card.md         # Model card with bias/fairness documentation
  reflection.md         # Design decisions and learnings
  requirements.txt
  .env.example          # API key and provider config template
```

## Key Features

- **RAG Pipeline:** 114K real Spotify tracks embedded in ChromaDB with semantic search
- **Agentic Loop:** LangGraph state machine with conditional self-revision (max 3 cycles)
- **Bias Detection:** Genre concentration (KL divergence), popularity bias (Gini coefficient), mood homogeneity, artist repetition, demographic proxy checks
- **Confidence Scoring:** Multi-signal scoring (feature match, retrieval relevance, margin, diversity contribution)
- **Self-Critique:** LLM-powered critique with rule-based fallback
- **Guardrails:** Max genre ratio, artist cap, min relevance, metadata completeness
- **Observability:** Full decision trace logged as JSON per run
- **Explainability:** Per-song natural language explanations

## License

MIT
