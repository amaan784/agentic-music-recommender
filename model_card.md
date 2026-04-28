# Model Card: SoundScout AI

## Model Details

- **Name:** SoundScout AI Recommendation Agent
- **Type:** Agentic RAG-based music recommendation system
- **Version:** 1.0.0
- **Architecture:** LangGraph state machine with retrieve, score, critique, revise loop
- **Embedding Model:** HuggingFace `all-MiniLM-L6-v2` (default) or OpenAI `text-embedding-3-small`
- **LLM (critique and explain):** GPT-4o-mini or Claude Sonnet
- **Vector Store:** ChromaDB (local persistence)

## Intended Use

- Personalized music recommendations based on user mood, energy, genre preferences, and audio features
- Portfolio demonstration of agentic AI, RAG, bias detection, and self-critique patterns
- **Not intended** for production use without further evaluation and safety testing

## Training Data

- **Kaggle Spotify Tracks Dataset:** ~114,000 tracks across 125 genres
- 20 audio features per track: danceability, energy, valence, tempo, acousticness, etc.
- No user data is collected or used for model training

## Evaluation Metrics

_To be populated after evaluation runs:_

| Metric | Value | Notes |
|--------|-------|-------|
| Diversity Score (Shannon Entropy) | _TBD_ | Higher = more genre diversity |
| Coverage (% catalog recommended) | _TBD_ | Across N user profiles |
| Novelty Score | _TBD_ | Avg inverse popularity |
| Intra-list Similarity | _TBD_ | Lower = more diverse recs |
| Avg Confidence Score | _TBD_ | Weighted multi-signal score |
| Bias Flags (avg per run) | _TBD_ | Out of 5 checks |

## Bias & Fairness

### Detected Biases
- **Popularity bias:** Recommender may favor high-popularity tracks (tested via popularity_bias detector)
- **Genre concentration:** Risk of over-representing user's stated genres at the expense of discovery
- **Mood homogeneity:** Without critique loop, recommendations may cluster around similar valence

### Mitigation Strategies
1. **Guardrails:** Max 50% single-genre ratio, max 2 tracks per artist, min relevance threshold
2. **Bias detection:** 5 automated checks run on every recommendation set
3. **Self-critique loop:** LLM critiques biased lists and triggers re-scoring (up to 3 revisions)
4. **Confidence scoring:** Diversity contribution is 20% of each track's confidence score

## Limitations

- Recommendations are based on audio features only (no collaborative filtering)
- Cold start: no user history, relies entirely on stated preferences
- LLM critique quality depends on model availability and API costs
- Audio features are pre-computed by Spotify, no custom audio analysis
- Dataset may not reflect current music trends (static snapshot)

## Ethical Considerations

- No user data is collected or stored
- Explicit content ratio is monitored as a demographic proxy
- System is designed with transparency: full decision trace logged per run
- All bias checks and their results are visible to the user
