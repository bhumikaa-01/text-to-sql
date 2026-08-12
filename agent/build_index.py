#!/usr/bin/env python3
"""
build_index.py — One-time script to embed the semantic schema into ChromaDB.

Run:
python -m agent.build_index
"""

import os
import logging

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from agent.semantic_layer import SEMANTIC_SCHEMA

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

COLLECTION_NAME = "schema_index"


def serialize_table(table: dict) -> str:
    """Convert a table schema into text for embedding."""

    lines = [
        f"Table: {table['table_name']}",
        f"Description: {table['description']}",
        "Columns:",
    ]

    for col in table["columns"]:
        lines.append(
            f"  - {col['name']}: {col['description']}"
        )

    return "\n".join(lines)


def build_index() -> None:
    """Build and persist schema embeddings."""

    persist_dir = os.getenv(
        "CHROMA_PERSIST_DIR",
        "./chroma_store",
    )

    logger.info(
        "Initializing ChromaDB client at %s",
        persist_dir,
    )

    client = chromadb.PersistentClient(
        path=persist_dir
    )

    embedding_fn = (
        SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    try:
        client.delete_collection(
            COLLECTION_NAME
        )
        logger.info(
            "Deleted existing collection '%s'",
            COLLECTION_NAME,
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []

    for table in SEMANTIC_SCHEMA:
        text = serialize_table(table)

        documents.append(text)
        metadatas.append(
            {
                "table_name": table["table_name"]
            }
        )
        ids.append(table["table_name"])

        logger.info(
            "Prepared embedding for table: %s",
            table["table_name"],
        )

    logger.info(
        "Upserting %d documents into ChromaDB...",
        len(documents),
    )

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(
        "Index build complete. %d tables indexed.",
        len(documents),
    )


if __name__ == "__main__":
    build_index()