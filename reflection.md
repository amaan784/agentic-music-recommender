# Reflection: SoundScout AI

## What I Built

SoundScout AI is an agentic music recommendation system that combines RAG (Retrieval-Augmented Generation), a LangGraph-powered agent loop, automated bias detection, confidence scoring, and LLM-powered self-critique to produce explainable, fair, and reliable music recommendations from 114K real Spotify tracks.

## Key Design Decisions

### Why LangGraph over simpler approaches?
LangGraph provides a stateful, cyclic graph model that naturally supports the retrieve, score, critique, revise loop. This was essential for implementing the self-critique pattern where low-confidence recommendations trigger re-scoring with adjusted weights, up to a maximum of 3 revision cycles.

### Why ChromaDB for the vector store?
ChromaDB runs locally with no infrastructure requirements, making the project fully reproducible. Combined with LangChain's document loaders and the HuggingFace `all-MiniLM-L6-v2` embedding model, this provides a zero-cost RAG pipeline.

### Why both rule-based and LLM critique?
The rule-based critique acts as a reliable fallback when the LLM API is unavailable. This ensures the system degrades gracefully, a production engineering consideration that demonstrates reliability thinking.

## Challenges & Learnings

1. **Guardrail design:** Determining the right thresholds (50% genre cap, min 0.3 relevance) required iterative testing against the full dataset
2. **Confidence scoring weights:** The 35/25/20/20 split (feature match / retrieval relevance / margin / bias contribution) was calibrated through experimentation
3. **Bias detection trade-offs:** Some bias is intentional. If a user requests "upbeat pop," the system should concentrate on that genre. The bias detector flags concentration but doesn't automatically penalize user-aligned skew

## Testing Results

50 tests across 5 files. 45 of 45 runnable tests pass without the LangGraph dependency installed. The full 50 pass once `langgraph` is available. Tests cover guardrail filtering, bias detection, evaluation metrics, confidence scoring, critique logic, graph conditionals, logging, and the core scoring engine.

The most valuable tests were the bias detection ones. Writing a test that creates 10 identical-genre recommendations and asserts the detector flags it forced me to think about what "too concentrated" actually means numerically. The KL divergence threshold of 0.5 came directly from running that test against different distributions.

## What Surprised Me

The confidence scorer had a subtle bug where two out of five audio features were silently falling back to default values (0.5) because the test passed preference keys like `energy_value` while the function looked up `energy`. Two tests passed by coincidence because the default values happened to produce acceptable cosine similarities. I only found it when a third test with extreme values (all 0.0 vs all 1.0) failed. This taught me that default-value fallbacks are dangerous in scoring functions and that passing tests can give false confidence.

The bias detector also surprised me. When I first ran it on the indie-rock example, it flagged genre concentration even though the user specifically asked for indie and rock. This is technically correct behavior (the recommendations are concentrated), but it felt wrong to penalize a system for doing what the user asked. I resolved this by having the critique node consider the bias flags but not automatically reject the list, only revise if confidence is also low.

## AI Collaboration

I used AI assistance throughout this project.

**Helpful:** When structuring the LangGraph workflow, the AI suggested using a `should_revise` conditional edge that checks both the critique's `should_revise` flag and the average confidence threshold together, rather than just the critique flag alone. This was a better design because it prevents unnecessary revision cycles when confidence is already high, even if the critique identifies minor issues.

**Flawed:** The AI initially generated the Anthropic LLM integration using `from langchain_community.chat_models import ChatAnthropic`, which is a deprecated import path from an older version of the langchain ecosystem. The correct import is `from langchain_anthropic import ChatAnthropic`. I caught this during dependency validation when the import failed. It was a reminder to verify generated code against current package documentation.

## What I'd Do Differently

- Add collaborative filtering signals if user history were available
- Implement A/B testing infrastructure for comparing different scoring strategies
- Add streaming support for real-time UI updates during the agent loop
- Explore fine-tuned embeddings on music-specific data for better retrieval

## Skills Demonstrated

- **RAG:** Real 114K-track vector store with LangChain + ChromaDB
- **Agentic AI:** LangGraph state machine with conditional revision loops
- **Responsible AI:** Bias detection, fairness metrics, model card with real data
- **Evaluation:** Diversity, coverage, novelty, and fairness scoring
- **Self-critique:** LLM-powered critique loop with confidence thresholds
- **Observability:** Full decision trace logged per run
- **Software Engineering:** Modular architecture, tests for every module, clean repo structure
