# System Architecture Diagram

```mermaid
flowchart TD
    subgraph UI["Streamlit UI"]
        A[User Preference Form] --> B[Get Recommendations Button]
        R[Results + Explanations]
        BR[Bias Report + Metrics]
        DL[Decision Log Panel]
    end

    subgraph AGENT["LangGraph Agent (10 nodes, stateful)"]
        C[Parse Input] --> D[Build Query]
        D --> E[Retrieve from ChromaDB]
        E --> F[Apply Guardrails]
        F --> G[Score Candidates]
        G --> H[Check Bias]
        H --> I[Compute Confidence]
        I --> J[LLM Critique]
        J --> K{Revise?}
        K -- "avg confidence < 0.6\nand revision < 3" --> L[Adjust Weights]
        L --> E
        K -- "good enough\nor max loops hit" --> M[Finalize + Explain]
    end

    subgraph DATA["Data Layer"]
        DB[(ChromaDB\n114K tracks)]
        CSV[tracks_clean.csv]
        LOGS[logs/\nrun_*.json\neval_*.json]
    end

    subgraph TESTING["Testing and Evaluation"]
        T1[test_rag.py\n10 guardrail tests]
        T2[test_evaluation.py\n12 bias + metric tests]
        T3[test_confidence.py\n12 scorer + critic tests]
        T4[test_agent.py\n7 graph + logger tests]
        T5[test_recommender.py\n9 scoring tests]
    end

    B --> C
    E --> DB
    G --> CSV
    H --> CSV
    M --> R
    M --> BR
    M --> DL
    M --> LOGS

    TESTING -.->|validates| F
    TESTING -.->|validates| G
    TESTING -.->|validates| H
    TESTING -.->|validates| I
    TESTING -.->|validates| J

    style UI fill:#f0f4ff,stroke:#4a6fa5
    style AGENT fill:#fff8f0,stroke:#c97a2e
    style DATA fill:#f0fff4,stroke:#4a8c5c
    style TESTING fill:#fff0f0,stroke:#a54a4a
```

## Data Flow

1. User enters preferences in Streamlit (genres, mood, energy, danceability, tempo, acousticness, free text)
2. LangGraph agent parses input and builds a natural language query
3. ChromaDB returns 30 candidate tracks via semantic similarity search
4. Guardrails filter out low-relevance, over-represented genre, repeated artist, and incomplete tracks
5. Remaining candidates are ranked by cosine similarity against the user's preference vector
6. Bias detector compares recommendation distribution against a 1000-track catalog sample
7. Confidence scorer assigns each track a 0-to-1 score from four weighted signals
8. LLM (or rule-based fallback) critiques the list and decides whether to revise
9. If confidence is low and revisions remain, the agent adjusts weights and loops back to step 3
10. Final list gets per-song explanations and is returned to the UI with the full decision log

## Where Humans and Tests Fit In

The user reviews recommendations, explanations, bias flags, and the decision log in the Streamlit UI. Every guardrail, bias check, scoring function, critique rule, and graph conditional is covered by automated tests (50 total). Run logs and evaluation reports are persisted as JSON for offline review.
