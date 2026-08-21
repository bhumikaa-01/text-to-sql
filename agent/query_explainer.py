"""
Query explanation utilities.

Uses deterministic SQL analysis to create grounded metadata,
then uses the configured LLM to phrase that metadata into a
concise natural-language explanation.

The deterministic layer provides factual grounding.
The LLM is responsible only for natural-language framing.
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate


logger = logging.getLogger(__name__)


# ============================================================
# PUBLIC EXPLANATION FUNCTION
# ============================================================


async def explain_query(
    question: str,
    sql: str,
    tables_used: list[str],
    llm: Any,
) -> dict:
    """
    Generate a grounded natural-language explanation of SQL.

    The SQL is first analyzed deterministically to extract
    operations. The LLM then converts those grounded facts
    into a concise explanation.

    If the LLM fails, a deterministic fallback explanation
    is returned.
    """

    metadata = _analyze_sql(
        sql=sql,
        tables_used=tables_used,
    )

    try:

        prompt = _build_explanation_prompt(
            question=question,
            sql=sql,
            metadata=metadata,
        )

        response = await llm.ainvoke(
            prompt
        )

        explanation = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        explanation = (
            explanation
            .strip()
            .strip('"')
            .strip()
        )

        if not explanation:
            raise ValueError(
                "LLM returned an empty explanation"
        )

        if not _validate_explanation_grounding(
            explanation,
            metadata,
        ):
            logger.warning(
                "LLM explanation failed grounding validation; "
                "using deterministic fallback"
        )

            return {
                "summary": _build_fallback_explanation(
                    metadata
                ),
                "tables_used": tables_used,
                "operation_count": metadata[
                    "operation_count"
                ],
            }

        return {
            "summary": explanation,
            "tables_used": tables_used,
            "operation_count": metadata[
                "operation_count"
            ],
        }

    except Exception as exc:

        logger.warning(
            "LLM query explanation failed; "
            "using deterministic fallback: %s",
            exc,
        )

        return {
            "summary": _build_fallback_explanation(
                metadata
            ),
            "tables_used": tables_used,
            "operation_count": metadata[
                "operation_count"
            ],
        }


# ============================================================
# DETERMINISTIC SQL ANALYSIS
# ============================================================


def _analyze_sql(
    sql: str,
    tables_used: list[str],
) -> dict[str, Any]:
    """
    Extract factual SQL metadata.

    This function does NOT attempt to produce polished
    natural language. Its purpose is grounding.
    """

    sql_upper = sql.upper()

    operations: list[str] = []

    aggregation = _detect_aggregation(
        sql
    )

    if aggregation:
        operations.append(
            aggregation
        )

    filter_description = _detect_filter(
        sql
    )

    if filter_description:
        operations.append(
            filter_description
        )

    group_description = _detect_grouping(
        sql
    )

    if group_description:
        operations.append(
            group_description
        )

    join_description = _detect_join(
        sql
    )

    if join_description:
        operations.append(
            join_description
        )

    order_description = _detect_ordering(
        sql
    )

    if order_description:
        operations.append(
            order_description
        )

    limit_description = _detect_limit(
        sql
    )

    if limit_description:
        operations.append(
            limit_description
        )

    if not operations:
        operations.append(
            "retrieves the requested data"
        )

    return {
        "tables_used": tables_used,
        "aggregation": aggregation,
        "filter": filter_description,
        "grouping": group_description,
        "join": join_description,
        "ordering": order_description,
        "limit": limit_description,
        "operations": operations,
        "operation_count": len(
            operations
        ),
    }


# ============================================================
# AGGREGATION DETECTION
# ============================================================


def _detect_aggregation(
    sql: str,
) -> str:
    """
    Detect aggregation type.

    Returns factual metadata rather than polished prose.
    """

    sql_upper = sql.upper()

    if "COUNT(" in sql_upper:

        if (
            "DISTINCT" in sql_upper
            and "ORDER_ID" in sql_upper
        ):
            return (
                "COUNT DISTINCT on order_id"
            )

        return "COUNT aggregation"

    if "SUM(" in sql_upper:

        if "ORDER_TOTAL_USD" in sql_upper:
            return (
                "SUM aggregation on order_total_usd"
            )

        return "SUM aggregation"

    if "AVG(" in sql_upper:
        return "AVG aggregation"

    if "MIN(" in sql_upper:
        return "MIN aggregation"

    if "MAX(" in sql_upper:
        return "MAX aggregation"

    return ""


# ============================================================
# FILTER DETECTION
# ============================================================

def _detect_filter(
    sql: str,
) -> str:
    """Detect factual WHERE conditions."""

    import re

    sql_upper = sql.upper()

    if "WHERE" not in sql_upper:
        return ""

    # --------------------------------------------------
    # Detect order status filters
    # --------------------------------------------------

    match = re.search(
        r"ORDER_STATUS\s*=\s*['\"]([^'\"]+)['\"]",
        sql,
        flags=re.IGNORECASE,
    )

    if match:
        status = match.group(1)

        return (
            f"WHERE filter: order_status = {status}"
        )

    # --------------------------------------------------
    # Generic WHERE fallback
    # --------------------------------------------------

    return "WHERE filtering is applied"

# ============================================================
# GROUPING DETECTION
# ============================================================


def _detect_grouping(
    sql: str,
) -> str:
    """Detect GROUP BY operations."""

    sql_upper = sql.upper()

    if "GROUP BY" not in sql_upper:
        return ""

    if "CATEGORY_NAME" in sql_upper:
        return (
            "GROUP BY category_name"
        )

    return "GROUP BY is used"


# ============================================================
# JOIN DETECTION
# ============================================================


def _detect_join(
    sql: str,
) -> str:
    """Detect JOIN operations without assuming business meaning."""

    sql_upper = sql.upper()

    if "JOIN DIM_PRODUCTS" in sql_upper:
        return (
            "JOIN between fact_orders and dim_products"
        )

    if "JOIN" in sql_upper:
        return "JOIN between related tables"

    return ""


# ============================================================
# ORDERING DETECTION
# ============================================================


def _detect_ordering(
    sql: str,
) -> str:
    """Detect ORDER BY direction."""

    sql_upper = sql.upper()

    if "ORDER BY" not in sql_upper:
        return ""

    if "DESC" in sql_upper:
        return (
            "ORDER BY descending"
        )

    return "ORDER BY ascending"


# ============================================================
# LIMIT DETECTION
# ============================================================


def _detect_limit(
    sql: str,
) -> str:
    """Detect LIMIT value."""

    import re

    match = re.search(
        r"\bLIMIT\s+(\d+)",
        sql,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    limit_value = match.group(1)

    return (
        f"LIMIT {limit_value}"
    )


# ============================================================
# LLM PROMPT
# ============================================================


def _build_explanation_prompt(
    question: str,
    sql: str,
    metadata: dict[str, Any],
) -> ChatPromptTemplate:
    """
    Build a grounded prompt for the explanation LLM.
    """

    system_prompt = """
You are a SQL query explanation assistant.

Your job is to explain what a SQL query does
in simple language for a non-technical user.

IMPORTANT RULES:

1. Use ONLY facts explicitly supported by the SQL
   and the grounded metadata.

2. Do NOT invent business meaning.

3. Do NOT claim the query calculates revenue unless
   the SQL actually contains a revenue calculation.

4. Do NOT claim the query counts orders unless
   the SQL actually counts order IDs.

5. Explain the actual operations present in the SQL.

6. Mention important filtering, grouping, ordering,
   and LIMIT operations when they affect the user's
   question.

7. Keep the explanation concise:
   preferably one sentence, maximum two sentences.

8. Do not mention SQL syntax such as SELECT, JOIN,
   GROUP BY, or WHERE unless necessary for clarity.

9. Do not provide the SQL itself.

10. Do not discuss confidence, safety, validation,
    implementation details, or the LLM.

11. Do not invent insights from the returned data.
"""

    user_prompt = """
User question:
{question}

SQL query:
{sql}

Grounded SQL metadata:
{metadata}

Write a concise explanation of what the query does.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                user_prompt,
            ),
        ]
    )

    return prompt.invoke(
        {
            "question": question,
            "sql": sql,
            "metadata": metadata,
        }
    )


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================
def _validate_explanation_grounding(
    explanation: str,
    metadata: dict[str, Any],
) -> bool:
    """
    Validate that the LLM explanation does not introduce
    unsupported metrics, filters, or result constraints.

    The validator intentionally focuses on claims that can
    materially change the meaning of the query.
    """

    explanation_lower = explanation.lower()

    aggregation = (
        metadata.get("aggregation") or ""
    ).lower()

    filter_metadata = (
        metadata.get("filter") or ""
    ).lower()

    limit_metadata = (
        metadata.get("limit") or ""
    ).lower()

    # --------------------------------------------------
    # Aggregation grounding
    # --------------------------------------------------

    has_sum = "sum" in aggregation
    has_avg = "avg" in aggregation
    has_count = "count" in aggregation
    has_distinct_orders = (
        "count distinct on order_id"
        in aggregation
    )
    has_min = "min" in aggregation
    has_max = "max" in aggregation

    # Revenue is only supported when the SQL explicitly
    # aggregates order_total_usd.
    if "revenue" in explanation_lower:
        if "order_total_usd" not in aggregation:
            return False

    # "average" must correspond to AVG.
    if "average" in explanation_lower:
        if not has_avg:
            return False

    # "total" should only be treated as a metric claim
    # when SUM is actually present.
    if "total revenue" in explanation_lower:
        if not has_sum:
            return False

    # "minimum" / "maximum" must match SQL aggregation.
    if "minimum" in explanation_lower:
        if not has_min:
            return False

    if "maximum" in explanation_lower:
        if not has_max:
            return False

    # --------------------------------------------------
    # COUNT grounding
    # --------------------------------------------------

    # Only validate explicit counting claims.
    #
    # We deliberately DO NOT reject the word "orders"
    # by itself because orders may be referenced as a
    # source/entity rather than as the calculated metric.

    counting_phrases = (
        "counts the number of orders",
        "counts orders",
        "counts the number of unique orders",
        "counts distinct orders",
        "number of unique orders",
        "number of distinct orders",
    )

    if any(
        phrase in explanation_lower
        for phrase in counting_phrases
    ):
        if not has_count:
            return False

    # --------------------------------------------------
    # LIMIT grounding
    # --------------------------------------------------

    if "top " in explanation_lower:
        if not limit_metadata:
            return False

    # --------------------------------------------------
    # Filter grounding
    # --------------------------------------------------

    if "delivered" in explanation_lower:
        if "delivered" not in filter_metadata:
            return False

    if "canceled" in explanation_lower:
        if "canceled" not in filter_metadata:
            return False

    if "cancelled" in explanation_lower:
        if (
            "canceled" not in filter_metadata
            and "cancelled" not in filter_metadata
        ):
            return False

    return True

def _build_fallback_explanation(
    metadata: dict[str, Any],
) -> str:
    """
    Build a safe explanation if the LLM is unavailable.
    """

    operations = metadata.get(
        "operations",
        [],
    )

    if not operations:
        return (
            "The query retrieves the requested data."
        )

    phrases: list[str] = []

    for operation in operations:

        if operation.startswith(
            "COUNT DISTINCT"
        ):
            phrases.append(
                "counts distinct orders"
            )

        elif operation.startswith(
            "COUNT"
        ):
            phrases.append(
                "counts the matching records"
            )

        elif operation.startswith(
            "SUM"
        ):
            phrases.append(
                "calculates a total"
            )

        elif operation.startswith(
            "AVG"
        ):
            phrases.append(
                "calculates an average"
            )

        elif operation.startswith(
            "MIN"
        ):
            phrases.append(
                "finds the minimum value"
            )

        elif operation.startswith(
            "MAX"
        ):
            phrases.append(
                "finds the maximum value"
            )

        elif operation.startswith(
            "WHERE filter"
        ):
            phrases.append(
                "applies the requested filter"
            )

        elif operation.startswith(
            "WHERE"
        ):
            phrases.append(
                "filters the data"
            )

        elif operation.startswith(
            "GROUP BY category_name"
        ):
            phrases.append(
                "groups the results by product category"
            )

        elif operation.startswith(
            "GROUP BY"
        ):
            phrases.append(
                "groups the results"
            )

        elif operation.startswith(
            "JOIN"
        ):
            phrases.append(
                "combines related table information"
            )

        elif operation.startswith(
            "ORDER BY descending"
        ):
            phrases.append(
                "sorts the results from highest to lowest"
            )

        elif operation.startswith(
            "ORDER BY"
        ):
            phrases.append(
                "sorts the results"
            )

        elif operation.startswith(
            "LIMIT"
        ):
            limit_value = operation.split()[-1]

            phrases.append(
                f"returns the top {limit_value} results"
            )

    if len(phrases) == 1:
        return (
            f"The query {phrases[0]}."
        )

    if len(phrases) == 2:
        return (
            f"The query {phrases[0]} "
            f"and {phrases[1]}."
        )

    return (
        "The query "
        + ", ".join(phrases[:-1])
        + ", and "
        + phrases[-1]
        + "."
    )