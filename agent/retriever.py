"""
retriever.py — RAG-based schema retrieval from ChromaDB.
"""

import os
import logging
from typing import Optional

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "schema_index"

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_embedding(text: str):
    """
    Generate embeddings using Gemini.
    """
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )

    return response.embeddings[0].values


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

    _client = chromadb.PersistentClient(
        path=persist_dir
    )

    _collection = _client.get_collection(
        name=COLLECTION_NAME
    )

    return _collection


def get_relevant_schema(
    query: str,
    k: int = 3
) -> str:
    """
    Retrieve most relevant schema documents.
    """

    try:
        collection = _get_collection()

        collection_count = collection.count()

        if collection_count == 0:
            logger.warning(
                "Schema collection is empty."
            )
            return ""

        query_embedding = get_embedding(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection_count),
        )

        documents = results["documents"][0]

        return "\n\n---\n\n".join(documents)

    except Exception as exc:
        logger.warning(
            "Schema retrieval failed: %s",
            exc
        )
        return ""