# SoundScout AI

> An end-to-end agentic RAG system that retrieves from 114K real Spotify tracks, orchestrates a multi-step recommendation pipeline with LangGraph, detects and corrects for popularity and genre bias, and self-critiques its own outputs with confidence scoring, all with full observability logging.

## Architecture

```
STREAMLIT UI -> LANGGRAPH AGENT -> DATA LAYER
                  |
                  |-- Parse Input
                  |-- Build Query
                  |-- Retrieve (RAG / ChromaDB)
                  |-- Apply Guardrails
                  |-- Score Candidates
                  |-- Bias Detection
                  |-- Confidence Scoring
                  |-- LLM Critique
                  |-- [Conditional] Revise & Re-retrieve (max 3 loops)
                  |-- Finalize with Explanations
```

## Tech Stack

| Layer | Tool |
|-------|------|
| Agent Orchestration | LangGraph |
| RAG / Retrieval | LangChain + ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (default) or OpenAI |
| LLM (critique, explain) | GPT-4o-mini, Claude Sonnet, or Mistral Large |
| Data | Kaggle Spotify Tracks Dataset (114K tracks) |
| UI | Streamlit |
| Testing | pytest |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
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
streamlit run app.py
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

## Project Structure

```
soundscout-ai/
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
