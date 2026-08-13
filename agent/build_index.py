#!/usr/bin/env python3
"""
build_index.py — One-time script to embed the semantic schema into ChromaDB.

Run:
python -m agent.build_index
"""

import os
import time
import logging

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from agent.semantic_layer import SEMANTIC_SCHEMA

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO")
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "schema_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def serialize_table(table: dict) -> str:
    """
    Convert a table schema into text for embedding.
    """

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
    """
    Build and persist schema embeddings.
    """

    start_time = time.time()

    persist_dir = os.getenv(
        "CHROMA_PERSIST_DIR",
        "./chroma_store",
    )

    logger.info(
        "Initializing ChromaDB at %s",
        persist_dir,
    )

    client = chromadb.PersistentClient(
        path=persist_dir
    )

    embedding_fn = (
        SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    )

    # Delete old collection if it exists
    try:
        existing = [
            c.name
            for c in client.list_collections()
        ]

        if COLLECTION_NAME in existing:
            client.delete_collection(
                COLLECTION_NAME
            )

            logger.info(
                "Deleted existing collection '%s'",
                COLLECTION_NAME,
            )

    except Exception as exc:
        logger.warning(
            "Could not delete collection: %s",
            exc,
        )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    documents = []
    metadatas = []
    ids = []

    for table in SEMANTIC_SCHEMA:

        table_name = table["table_name"]

        documents.append(
            serialize_table(table)
        )

        metadatas.append(
            {
                "table_name": table_name
            }
        )

        ids.append(table_name)

        logger.info(
            "Prepared schema: %s",
            table_name,
        )

    logger.info(
        "Adding %d schema documents...",
        len(documents),
    )

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    final_count = collection.count()

    elapsed = round(
        time.time() - start_time,
        2
    )

    logger.info(
        "Index build complete."
    )

    logger.info(
        "Documents indexed: %d",
        final_count,
    )

    logger.info(
        "Embedding model: %s",
        EMBEDDING_MODEL,
    )

    logger.info(
        "Build time: %.2f seconds",
        elapsed,
    )


if __name__ == "__main__":
    build_index()