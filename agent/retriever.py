"""
retriever.py — RAG-based schema retrieval from ChromaDB.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "schema_index"

_client: Optional[chromadb.PersistentClient] = None
_collection = None

# Load embedding model once at startup
_embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


@lru_cache(maxsize=256)

def cached_embedding(text: str):
    return [float(x) for x in _embedding_model.encode(text)]


def get_embedding(text: str):
    return cached_embedding(text)


def _get_collection():
    """
    Lazily initialise and cache Chroma collection.
    """

    global _client, _collection

    if _collection is not None:
        return _collection

    persist_dir = os.getenv(
        "CHROMA_PERSIST_DIR",
        "./chroma_store"
    )

    logger.info(
        "Loading Chroma collection from %s",
        persist_dir
    )

    _client = chromadb.PersistentClient(
        path=persist_dir
    )

    _collection = _client.get_collection(
        name=COLLECTION_NAME
    )

    return _collection


def get_relevant_schema(
    query: str,
    k: int = 2
) -> str:

    """
    Retrieve the most relevant schema documents
    for a user question.
    """

    try:
        collection = _get_collection()

        logger.info("COLLECTION LOADED")

        collection_count = collection.count()

        logger.info(
            "Collection count: %d",
            collection_count
        )


        if collection_count == 0:
            logger.warning(
                "Schema collection is empty."
            )
            return ""

        query_embedding = get_embedding(
            query
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=min(
                k,
                collection_count
            ),
        )

        if (
            not results
            or "documents" not in results
            or not results["documents"]
        ):
            logger.warning(
                "No schema documents retrieved."
            )
            return ""

        documents = results["documents"][0]

        logger.info(
            "DOCUMENTS FOUND: %d",
            len(documents)
        )

        logger.info(
            "Retrieved %d schema chunks.",
            len(documents)
        )

        return "\n\n---\n\n".join(
            documents
        )

    except Exception as exc:
        print("!!!!!!!! RETRIEVER ERROR !!!!!!!!")
        print(type(exc))
        print(exc)
        raise