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

RAG_DISTANCE_THRESHOLD = 0.75

_client: Optional[chromadb.PersistentClient] = None
_collection = None

# Load embedding model once at startup
_embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


@lru_cache(maxsize=256)
def cached_embedding(text: str):
    return [
        float(x)
        for x in _embedding_model.encode(text)
    ]


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


def _expand_query(query: str) -> str:
    """
    Add database-domain terminology to natural-language
    questions when useful for semantic retrieval.

    This improves recall for paraphrased questions such as:

        "How much money did the company make?"

    which should retrieve revenue-related schema even though
    the word "revenue" is not present in the original question.
    """

    query_lower = query.lower()

    expansions = []

    # Revenue / sales / money questions
    revenue_terms = (
        "revenue",
        "sales",
        "money",
        "made",
        "income",
        "earnings",
        "turnover",
        "gmv",
        "profit",
        "earned",
    )

    if any(
        term in query_lower
        for term in revenue_terms
    ):
        expansions.extend(
            [
                "revenue",
                "total sales",
                "money made",
                "GMV",
                "order total",
            ]
        )

    # Order questions
    order_terms = (
        "order",
        "orders",
        "purchase",
        "purchases",
    )

    if any(
        term in query_lower
        for term in order_terms
    ):
        expansions.extend(
            [
                "orders",
                "order status",
                "order total",
            ]
        )

    # Customer questions
    customer_terms = (
        "customer",
        "customers",
        "buyer",
        "buyers",
        "user",
        "users",
    )

    if any(
        term in query_lower
        for term in customer_terms
    ):
        expansions.extend(
            [
                "customers",
                "users",
                "customer count",
            ]
        )

    # Product questions
    product_terms = (
        "product",
        "products",
        "category",
        "categories",
        "item",
        "items",
    )

    if any(
        term in query_lower
        for term in product_terms
    ):
        expansions.extend(
            [
                "products",
                "product category",
                "category performance",
            ]
        )

    # Seller questions
    seller_terms = (
        "seller",
        "sellers",
        "merchant",
        "merchants",
    )

    if any(
        term in query_lower
        for term in seller_terms
    ):
        expansions.extend(
            [
                "sellers",
                "seller performance",
            ]
        )

    # Review questions
    review_terms = (
        "review",
        "reviews",
        "rating",
        "ratings",
        "satisfaction",
        "score",
    )

    if any(
        term in query_lower
        for term in review_terms
    ):
        expansions.extend(
            [
                "reviews",
                "review score",
                "customer satisfaction",
            ]
        )

    if not expansions:
        return query

    expanded_query = (
        query
        + " "
        + " ".join(dict.fromkeys(expansions))
    )

    logger.info(
        "RAG query expanded: '%s' -> '%s'",
        query,
        expanded_query,
    )

    return expanded_query


def get_relevant_schema(
    query: str,
    k: int = 2
) -> str:
    """
    Retrieve the most relevant schema documents
    for a user question.

    A semantic distance threshold prevents unrelated
    questions from being sent to the SQL-generating LLM.
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

        # ------------------------------------------------
        # Expand natural-language queries before embedding
        # ------------------------------------------------

        retrieval_query = _expand_query(query)

        query_embedding = get_embedding(
            retrieval_query
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

        distances = (
            results.get("distances", [[]])[0]
        )

        logger.info(
            "DOCUMENTS FOUND: %d",
            len(documents)
        )

        logger.info(
            "Retrieved %d schema chunks.",
            len(documents)
        )

        # ------------------------------------------------
        # RAG relevance guard
        # ------------------------------------------------

        if not distances:
            logger.warning(
                "No distances returned by Chroma."
            )
            return ""

        best_distance = min(distances)

        if best_distance > RAG_DISTANCE_THRESHOLD:

            logger.warning(
                "RAG relevance check FAILED: "
                "best_distance=%.4f exceeds threshold=%.4f",
                best_distance,
                RAG_DISTANCE_THRESHOLD,
            )

            logger.warning(
                "No relevant schema found for question: %s",
                query,
            )

            return ""

        logger.info(
            "RAG relevance check PASSED: "
            "best_distance=%.4f threshold=%.4f",
            best_distance,
            RAG_DISTANCE_THRESHOLD,
        )

        return "\n\n---\n\n".join(
            documents
        )

    except Exception as exc:

        logger.exception(
            "Retriever error: %s",
            exc
        )

        raise