"""Data preprocessing and ChromaDB ingestion for SoundScout AI."""

import os
import pandas as pd
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(DATA_DIR, "spotify_tracks.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "tracks_clean.csv")
CHROMA_DIR = os.path.join(os.path.dirname(DATA_DIR), "chroma_db")


def load_raw_data(path: str = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, drop nulls, normalize, and add descriptors."""
    initial = len(df)

    df = df.drop_duplicates(subset=["track_id"])
    print(f"After dedup: {len(df)} rows (dropped {initial - len(df)})")

    df = df.dropna(subset=NUMERIC_FEATURES)
    print(f"After dropping nulls: {len(df)} rows")

    for col in NUMERIC_FEATURES:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0

    df["mood_descriptor"] = pd.cut(
        df["valence"],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["melancholic", "neutral", "upbeat"]
    )

    df["energy_descriptor"] = pd.cut(
        df["energy"],
        bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["low-energy", "moderate", "high-energy"]
    )

    return df


def save_clean_data(df: pd.DataFrame, path: str = CLEAN_CSV):
    """Write cleaned data to CSV."""
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} cleaned tracks to {path}")


def build_documents(df: pd.DataFrame) -> list:
    """Convert track rows into LangChain Documents."""
    documents = []
    for _, row in df.iterrows():
        text = (
            f"{row['track_name']} by {row['artists']}. "
            f"Genre: {row['track_genre']}. "
            f"Mood: {row.get('mood_descriptor', 'unknown')}. "
            f"Energy: {row.get('energy_descriptor', 'unknown')}. "
            f"Danceability: {row['danceability']:.2f}. "
            f"Valence: {row['valence']:.2f}. "
            f"Tempo: {row['tempo']:.2f}."
        )
        metadata = {
            "track_id": str(row["track_id"]),
            "track_name": str(row["track_name"]),
            "artists": str(row["artists"]),
            "album_name": str(row.get("album_name", "")),
            "track_genre": str(row["track_genre"]),
            "popularity": int(row.get("popularity", 0)),
            "danceability": float(row["danceability"]),
            "energy": float(row["energy"]),
            "valence": float(row["valence"]),
            "tempo": float(row["tempo"]),
            "acousticness": float(row["acousticness"]),
            "speechiness": float(row["speechiness"]),
            "instrumentalness": float(row["instrumentalness"]),
            "liveness": float(row["liveness"]),
            "loudness": float(row["loudness"]),
            "mood_descriptor": str(row.get("mood_descriptor", "unknown")),
            "energy_descriptor": str(row.get("energy_descriptor", "unknown")),
            "explicit": bool(row.get("explicit", False)),
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def get_embeddings():
    """Return the configured embedding model."""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu", "batch_size": 32}
        )


def build_vectorstore(documents: list, persist_directory: str = CHROMA_DIR):
    """Build and persist a ChromaDB collection."""
    embeddings = get_embeddings()
    print(f"Building ChromaDB with {len(documents)} documents...")

    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)

    batch_size = 5000
    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                batch, embeddings, persist_directory=persist_directory
            )
        else:
            vectorstore.add_documents(batch)
        print(f"  Indexed {min(i + batch_size, len(documents))}/{len(documents)} documents")

    # Explicit persist for newer Chroma versions
    if hasattr(vectorstore, 'persist'):
        vectorstore.persist()

    print(f"ChromaDB persisted to {persist_directory}")
    return vectorstore


def main():
    """Run the full data preparation pipeline."""
    df = load_raw_data()
    df = clean_data(df)
    save_clean_data(df)
    documents = build_documents(df)
    build_vectorstore(documents)
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
