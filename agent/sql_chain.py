"""
sql_chain.py — LangChain LCEL pipeline: question → SQL → results.

# INTERN NOTE: LCEL chain composition explained
# LangChain Expression Language (LCEL) lets you compose chains using the pipe
# operator (|). Each component receives the output of the previous one as input.
# Our pipeline:
#   1. Retrieve relevant schema (RAG)     → inject into prompt context
#   2. Load few-shot YAML examples        → inject into prompt for in-context learning
#   3. Build ChatPromptTemplate           → structured system + user messages
#   4. Call LLM (temperature=0)           → deterministic, no creativity in SQL
#   5. Parse SQL from response            → extract the raw SQL string
#   6. HITL check                         → flag writes for human approval
#   7. Execute if approved (SELECT only)  → run against SQLite/PostgreSQL
#   8. Log to query_log                   → observability + debugging
#
# temperature=0 is critical: we want the most deterministic SQL possible.
# Any creativity in SQL generation leads to incorrect queries.

#prompt = f"""
#You are a Text-to-SQL assistant.

#Relevant Schema:
#{schema_context}

#Question:
#{question}

#Generate SQL only.
"""
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text

from agent.hitl_guard import check_sql
from agent.input_guard import check_user_input
from agent.retriever import get_relevant_schema
from model.database import get_engine, get_session
from model.schema import Base, QueryLog
from agent.schema_validator import validate_sql_schema
from agent.query_guard import check_query_resources
from agent.confidence import calculate_confidence
from agent.table_correctness import check_table_correctness
from agent.semantic_evaluator import evaluate_semantics
from agent.query_explainer import explain_query
from agent.telemetry_store import QueryEvent
from agent.telemetry_store import record_event
from agent.query_cache import (
    get_cached_response,
    set_cached_response,
)
from agent.sql_retry import (
    MAX_SQL_RETRIES,
    build_correction_prompt,
    is_retryable_error,
)
from agent.result_visualizer import (
    recommend_visualization,
)
from agent.chart_renderer import (
    render_chart,
)

load_dotenv()

logger = logging.getLogger(__name__)

_YAML_PATH = os.path.join(os.path.dirname(__file__), "few_shot_examples.yaml")

# Module-level LLM instance — reuses the underlying HTTP connection pool
# across all requests instead of creating a new client per query.
_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
    return _llm

SYSTEM_PROMPT = """You are an expert SQL analyst for the Olist Brazilian E-Commerce database.

Your task: Given a natural-language question, write a single, correct SQL SELECT query.

Rules:
1. Output ONLY the SQL query — no explanation, no markdown fences, no commentary.
2. Use only tables and columns present in the schema context below.
3. Use table aliases (e.g. fo for fact_orders, dp for dim_products).
4. Always qualify column names with table aliases.
5. For SQLite date operations use strftime(); never use DATE_TRUNC or EXTRACT.
6. Prefer NULLIF(x, 0) to avoid division-by-zero.
7. Limit results to 1000 rows unless the question asks for all rows.

8. For monetary values and revenue calculations:
   - Always use ROUND(value, 2).
   - Use ROUND(SUM(...), 2) for revenue aggregations.
   - Use ROUND(AVG(...), 2) for averages.
   - Never return floating-point values with excessive precision.

9. Never generate INSERT, UPDATE, DELETE, or DROP statements.
Examples:

Question: What is the total revenue?
SQL:
SELECT ROUND(SUM(fo.order_total_usd), 2) AS total_revenue
FROM fact_orders fo;

Question: What is the average order value?
SQL:
SELECT ROUND(AVG(fo.order_total_usd), 2) AS avg_order_value
FROM fact_orders fo;

--- RELEVANT SCHEMA ---
{schema}

--- FEW-SHOT EXAMPLES ---
{examples}
"""

USER_PROMPT = "Question: {question}\n\nSQL:"


def _load_few_shot_examples() -> str:
    """Load and format few-shot examples from the YAML file."""
    try:
        with open(_YAML_PATH, "r") as fh:
            data = yaml.safe_load(fh)
        lines: list[str] = []
        for ex in data.get("examples", []):
            lines.append(f"Q: {ex['question']}")
            lines.append(f"SQL: {ex['sql'].strip()}")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Failed to load few-shot examples: %s", exc)
        return ""


def _extract_sql(raw_response: str) -> str:
    """Strip markdown code fences and whitespace from the LLM response."""
    # Remove ```sql ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"```(?:sql)?", "", raw_response, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    # Remove any leading "SQL:" label the model might add
    cleaned = re.sub(r"^SQL:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned

async def _correct_sql(
    *,
    question: str,
    previous_sql: str,
    validation_error: str,
    schema_context: str,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """
    Ask the LLM to correct a previously generated SQL query.

    This helper is responsible only for generating the corrected SQL.

    It does NOT:
        - execute SQL
        - validate schema
        - check SQL safety
        - check query resources
        - retry automatically

    Those responsibilities remain with the main pipeline.
    """

    correction_prompt = build_correction_prompt(
        question=question,
        previous_sql=previous_sql,
        validation_error=validation_error,
        schema_context=schema_context,
    )

    response = await llm.ainvoke(
        correction_prompt
    )

    corrected_raw = (
        response.content
        if hasattr(
            response,
            "content",
        )
        else str(
            response
        )
    )

    corrected_sql = _extract_sql(
        corrected_raw
    ).strip()

    if not corrected_sql:

        raise ValueError(
            "SQL correction model returned empty SQL."
        )

    logger.info(
        "SQL correction generated successfully."
    )

    return corrected_sql


def _extract_table_names(sql: str) -> list[str]:
    """Heuristically extract table names referenced in a SQL query."""
    known_tables = {
        "fact_orders",
        "dim_users",
        "dim_products",
        "dim_sellers",
        "dim_geography",
        "dim_reviews",
    }
    sql_upper = sql.upper()
    found = [t for t in known_tables if t.upper() in sql_upper]
    return found


def _ensure_schema_exists() -> None:
    """Create all tables (including query_log) if they don't exist yet."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def _log_query(
    question: str,
    generated_sql: str,
    latency_ms: int,
    tables_used: list[str],
    error: str | None,
) -> None:
    """Persist a query execution record to the query_log table."""
    try:
        _ensure_schema_exists()
        with get_session() as session:
            log_entry = QueryLog(
                question=question,
                generated_sql=generated_sql,
                latency_ms=latency_ms,
                tables_used=",".join(tables_used),
                error=error,
                created_at=datetime.utcnow(),
            )
            session.add(log_entry)
            session.commit()
    except Exception as exc:
        logger.error("Failed to write to query_log: %s", exc)


def _execute_sql(sql: str) -> list[dict[str, Any]]:
    """Run a SQL query and return results as a list of dicts.

    Only single-statement SELECT or WITH (CTE) queries are permitted.
    Any other statement type, or a multi-statement payload, raises ValueError
    so callers receive a clear error instead of executing unexpected SQL.

    A LIMIT clause is injected for SELECT queries so that the response stays
    manageable.
    """
    _MAX_ROWS = 1000
    normalised = sql.strip()

    # Reject multi-statement payloads: strip one trailing semicolon then check
    # for any remaining semicolons which would indicate a second statement.
    if ";" in normalised.rstrip(";"):
        raise ValueError("Multi-statement SQL is not allowed")

    # Allowlist: only SELECT and WITH (CTEs that start WITH … SELECT) are safe
    # read-only operations.  Everything else (PRAGMA, ATTACH, INSERT, …) is
    # rejected here regardless of what upstream guards may have passed.
    first_token = normalised.split()[0].upper() if normalised.split() else ""
    if first_token not in {"SELECT", "WITH"}:
        raise ValueError(
            f"Only SELECT/WITH queries are permitted; got: {first_token!r}"
        )

    normalised_upper = normalised.upper()
    if first_token == "SELECT" and "LIMIT" not in normalised_upper:
        sql = normalised.rstrip(";").rstrip() + f" LIMIT {_MAX_ROWS}"
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

async def run_query(
    question: str,
) -> dict[str, Any]:
    """
    Full Text-to-SQL pipeline.

    Pipeline:
        Natural language question
            ↓
        Schema retrieval
            ↓
        Few-shot examples
            ↓
        LLM SQL generation
            ↓
        SQL extraction
            ↓
        Empty SQL guard
            ↓
        SQL safety guard
            ↓
        Schema validation
            ↓
        Query resource guard
            ↓
        HITL approval guard
            ↓
        SQL execution
            ↓
        Confidence calculation
            ↓
        Structured response
    """
    start_time = time.monotonic()

    # ====================================================
    # STEP 0 — USER INPUT SECURITY GUARD
    # ====================================================

    input_guard = check_user_input(question)

    logger.info(
        "User input security check: allowed=%s risk=%s operation=%s",
        input_guard["allowed"],
        input_guard["risk_level"],
        input_guard["operation"],
    )

    if not input_guard["allowed"]:

        logger.warning(
            "User input blocked by security guard: %s",
            input_guard["reason"],
        )

        latency_ms = int(
            (
                time.monotonic()
                - start_time
            )
            * 1000
        )

        telemetry_event = QueryEvent(
            question=question,
            status="BLOCKED",
            latency_ms=latency_ms,

            sql_generated=False,
            sql_safe=False,

            sql_correction_attempted=False,
            sql_correction_count=0,
            sql_correction_applied=False,

            cache_hit=False,

            resource_decision="BLOCK",

            semantic_correct=None,
            semantic_score=0.0,

            confidence_score=0.0,
            confidence_level="LOW",

            tables_used=[],

            error=input_guard["reason"],
        )

        record_event(telemetry_event)

        logger.info(
            "Blocked telemetry event recorded: request_id=%s",
            telemetry_event.request_id,
        )

        return {
        "sql": "",
        "results": [],
        "tables_used": [],

        "requires_approval": False,
        "approval_reason": "",

        "resource_guard": {
            "decision": "BLOCK",
            "risk_level": input_guard["risk_level"],
            "violations": [
                "USER_INPUT_SECURITY",
                *input_guard.get("violations", []),
            ],
            "reason": input_guard["reason"],
        },

        "semantic_evaluation": {},

        "explanation": {},

        "visualization": {},

        "confidence": {
            "score": 0,
            "level": "LOW",
            "factors": {
                "sql_safety": 0,
                "schema_validity": 0,
                "resource_safety": 0,
                "execution": 0,
                "result_quality": 0,
                "table_correctness": 0,
            },
        },

        "cache": {
            "hit": False,
        },

        "latency_ms": latency_ms,

        "error": (
            "Request blocked by the security guard. "
            "The input contains a potentially dangerous "
            "SQL operation or injection pattern."
        ),
    }

    # ====================================================
    # STEP — Query cache lookup
    # ====================================================

    cached_response = get_cached_response(question)

    if cached_response is not None:

        latency_ms = int(
        (
            time.monotonic()
            - start_time
        )
        * 1000
        )

        logger.info(
            "Query cache HIT for question: %s",
            question,
        )

        cached_response["cache"] = {
            "hit": True,
        }

        cached_response["latency_ms"] = latency_ms

        return cached_response

    logger.info(
        "Query cache MISS for question: %s",
        question,
    )

    generated_sql = ""
    tables_used: list[str] = []

    try:

        # ====================================================
        # STEP 1 — Retrieve relevant schema
        # ====================================================

        schema_context = get_relevant_schema(
            question,
            k=3,
        )

        logger.info(
            "Schema context length: %d",
            len(schema_context),
        )

        # ----------------------------------------------------
        # RAG relevance guard
        # ----------------------------------------------------
        # If the retriever cannot find schema relevant enough
        # to the user's question, do not send the question to
        # the LLM. This prevents unrelated questions such as
        # "What is the capital of France?" from being converted
        # into meaningless SQL against an unrelated table.
        # ----------------------------------------------------

        if not schema_context.strip():

            latency_ms = int(
                (
                    time.monotonic()
                    - start_time
                )
                * 1000
            )

            logger.warning(
                "No relevant schema found for question: %s",
                question,
            )

            confidence = calculate_confidence(
                sql_safe=False,
                schema_valid=False,
                resource_decision="BLOCK",
                execution_success=False,
                result_quality=0,
                table_correct=None,
            )

            logger.info(
                "Confidence score: %.2f (%s)",
                confidence["score"],
                confidence["level"],
            )

            return {
                "sql": "",
                "results": [],
                "tables_used": [],

                "requires_approval": False,
                "approval_reason": "",

                "resource_guard": {
                    "decision": "BLOCK",
                    "risk_level": "HIGH",
                    "violations": [
                        "RAG_RELEVANCE",
                    ],
                    "reason": (
                        "No sufficiently relevant database "
                        "schema was found for the user's question."
                    ),
                },

                "semantic_evaluation": {
                    "is_correct": False,
                    "score": 0.0,
                    "reason": (
                        "Semantic evaluation was skipped because "
                        "no relevant schema was retrieved."
                    ),
                    "issues": [
                        "RAG_RELEVANCE",
                    ],
                },

                "explanation": {
                    "summary": (
                        "The question does not appear to match "
                        "the available database schema."
                    ),
                    "tables_used": [],
                    "operation_count": 0,
                },

                "visualization": {
                    "recommended": False,
                    "chart_type": None,
                    "x_axis": None,
                    "y_axis": None,
                    "reason": (
                        "No visualization is generated because "
                        "no SQL was executed."
                    ),
                    "chart": {
                        "rendered": False,
                        "chart_type": None,
                    },
                },

                "confidence": confidence,

                "cache": {
                    "hit": False,
                },

                "latency_ms": latency_ms,

                "error": (
                    "I couldn't find information relevant to "
                    "your question in the available database. "
                    "Please ask a question about the available "
                    "orders, customers, products, sellers, "
                    "reviews, or revenue data."
                ),
            }

        # ====================================================
        # STEP 2 — Load few-shot examples
        # ====================================================

        few_shot = _load_few_shot_examples()
    
        # ====================================================
        # STEP 2 — Load few-shot examples
        # ====================================================

        few_shot = _load_few_shot_examples()

        # ====================================================
        # STEP 3 — Build prompt
        # ====================================================

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                (
                    "human",
                    USER_PROMPT,
                ),
            ]
        )

        # ====================================================
        # STEP 4 — Initialize LLM
        # ====================================================

        llm = _get_llm()

        # ====================================================
        # STEP 5 — Build LCEL chain
        # ====================================================

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # ====================================================
        # STEP 6 — Generate SQL
        # ====================================================

        raw_response: str = await chain.ainvoke(
            {
                "schema": schema_context,
                "examples": few_shot,
                "question": question,
            }
        )

        logger.info(
            "Raw LLM response:\n%s",
            raw_response,
        )

        # ====================================================
        # STEP 7 — Extract SQL
        # ====================================================

        generated_sql = _extract_sql(
            raw_response
        ).strip()

        logger.info(
            "Extracted SQL:\n%s",
            generated_sql,
        )

        # ====================================================
        # STEP 8 — Empty SQL guard
        # ====================================================

        if not generated_sql:

            latency_ms = int(
                (
                    time.monotonic()
                    - start_time
                )
                * 1000
            )

            logger.warning(
                "Model returned empty SQL for question: %s",
                question,
            )

            confidence = calculate_confidence(
                sql_safe=False,
                schema_valid=False,
                resource_decision="BLOCK",
                execution_success=False,
                result_quality=0,
                table_correct=None,
            )

            logger.info(
                "Confidence score: %.2f (%s)",
                confidence["score"],
                confidence["level"],
            )

            return {
                "sql": "",
                "results": [],
                "tables_used": [],

                "requires_approval": False,
                "approval_reason": "",

                "resource_guard": {
                    "decision": "BLOCK",
                    "risk_level": "HIGH",
                    "violations": [
                        "EMPTY_SQL",
                    ],
                    "reason": (
                        "The model did not generate a valid SQL query."
                    ),
                },

                "semantic_evaluation": {
                    "is_correct": False,
                    "score": 0.0,
                    "reason": (
                        "Semantic evaluation was skipped because "
                        "the model returned empty SQL."
                    ),
                    "issues": [
                        "EMPTY_SQL",
                    ],
                },

                "explanation": {
                    "summary": (
                        "No executable SQL was generated for "
                        "the requested question."
                    ),
                    "tables_used": [],
                    "operation_count": 0,
                },

                "visualization": {
                    "recommended": False,
                    "chart_type": None,
                    "x_axis": None,
                    "y_axis": None,
                    "reason": (
                        "No visualization is generated because "
                        "no SQL was executed."
                    ),
                    "chart": {
                        "rendered": False,
                        "chart_type": None,
                    },
                },

                "confidence": confidence,

                "cache": {
                    "hit": False,
                },

                "latency_ms": latency_ms,

                "error": (
                    "I could not find the requested information "
                    "in the available database schema. Try asking "
                    "about orders, customers, products, sellers, "
                    "reviews, or revenue."
                ),
            }

        # ====================================================
        # STEP 9 — Extract referenced tables
        # ====================================================

        tables_used = _extract_table_names(
            generated_sql
        )

        # ====================================================
        # STEP 10 — SQL safety guard
        # ====================================================

        safety_result = check_sql(
            generated_sql
        )

        logger.info(
            "SQL safety check: "
            "allowed=%s risk=%s operation=%s",
            safety_result.get("allowed"),
            safety_result.get("risk_level"),
            safety_result.get("operation"),
        )

        # ----------------------------------------------------
        # Safety BLOCK
        # ----------------------------------------------------

        if not safety_result.get(
            "allowed",
            False,
        ):

            latency_ms = int(
                (
                    time.monotonic()
                    - start_time
                )
                * 1000
            )

            safety_reason = safety_result.get(
                "reason",
                "SQL failed the safety policy.",
            )

            logger.warning(
                "SQL blocked by safety guard: %s",
                safety_reason,
            )

            _log_query(
                question,
                generated_sql,
                latency_ms,
                tables_used,
                error=safety_reason,
            )

            confidence = calculate_confidence(
                sql_safe=False,
                schema_valid=True,
                resource_decision="BLOCK",
                execution_success=False,
                result_quality=0,
                table_correct=False,
            )

            return {
                "sql": generated_sql,
                "results": [],
                "tables_used": tables_used,

                "requires_approval": False,
                "approval_reason": "",

                "resource_guard": {
                    "decision": "BLOCK",
                    "risk_level": safety_result.get(
                        "risk_level",
                        "CRITICAL",
                    ),
                    "violations": [
                        "SQL_SAFETY",
                    ],
                    "reason": safety_reason,
                },

                "confidence": confidence,

                "latency_ms": latency_ms,
                "error": safety_reason,
            }

        # ====================================================
        # STEP 11 — Schema validation + automatic correction
        # ====================================================

        retry_count = 0

        while True:

            # ------------------------------------------------
            # Validate generated SQL against database schema.
            # ------------------------------------------------

            is_valid, schema_error = validate_sql_schema(
                generated_sql
            )

            if is_valid:
                schema_valid = True
                break

            schema_reason = (
                schema_error
                or "Generated SQL failed schema validation."
            )

            logger.warning(
                "Schema validation failed "
                "(attempt %d/%d): %s",
                retry_count + 1,
                MAX_SQL_RETRIES + 1,
                schema_reason,
            )

            # ------------------------------------------------
            # Check whether this failure can be corrected.
            # ------------------------------------------------

            if (
                not is_retryable_error(schema_reason)
                or retry_count >= MAX_SQL_RETRIES
            ):

                latency_ms = int(
                    (
                        time.monotonic()
                        - start_time
                    )
                    * 1000
                )

                _log_query(
                    question,
                    generated_sql,
                    latency_ms,
                    tables_used,
                    error=schema_reason,
                )

                confidence = calculate_confidence(
                    sql_safe=True,
                    schema_valid=False,
                    resource_decision="BLOCK",
                    execution_success=False,
                    result_quality=0,
                    table_correct=False,
                )

                return {
                    "sql": generated_sql,
                    "results": [],
                    "tables_used": tables_used,

                    "requires_approval": False,
                    "approval_reason": "",

                    "resource_guard": {
                        "decision": "BLOCK",
                        "risk_level": "HIGH",
                        "violations": [
                            "SCHEMA_VALIDATION",
                        ],
                        "reason": schema_reason,
                    },

                    "sql_retry": {
                        "attempted": retry_count > 0,
                        "retry_count": retry_count,
                        "max_retries": MAX_SQL_RETRIES,
                        "corrected": retry_count > 0,
                    },

                    "confidence": confidence,

                    "latency_ms": latency_ms,
                    "error": schema_reason,
                }

            # ------------------------------------------------
            # Automatic SQL correction
            # ------------------------------------------------

            retry_count += 1

            logger.info(
                "Attempting automatic SQL correction "
                "(retry %d/%d)",
                retry_count,
                MAX_SQL_RETRIES,
            )

            try:

                generated_sql = await _correct_sql(
                    question=question,
                    previous_sql=generated_sql,
                    validation_error=schema_reason,
                    schema_context=schema_context,
                    llm=llm,
                )

            except Exception as exc:

                logger.warning(
                    "Automatic SQL correction failed: %s",
                    exc,
                )

                latency_ms = int(
                    (
                    time.monotonic()
                    - start_time
                    )
                * 1000
                )

                _log_query(
                    question,
                    generated_sql,
                    latency_ms,
                    tables_used,
                    error=str(exc),
                )

                confidence = calculate_confidence(
                    sql_safe=True,
                    schema_valid=False,
                    resource_decision="BLOCK",
                    execution_success=False,
                    result_quality=0,
                    table_correct=False,
                )

                return {
                    "sql": generated_sql,
                    "results": [],
                    "tables_used": tables_used,

                    "requires_approval": False,
                    "approval_reason": "",

                    "resource_guard": {
                        "decision": "BLOCK",
                        "risk_level": "HIGH",
                        "violations": [
                            "SQL_CORRECTION_FAILED",
                        ],
                        "reason": str(exc),
                    },

                    "confidence": confidence,

                    "latency_ms": latency_ms,
                    "error": str(exc),
                }

            # ------------------------------------------------
            # Re-run SQL safety after correction.
            # ------------------------------------------------

            corrected_safety = check_sql(
                generated_sql
            )

            if not corrected_safety.get(
                "allowed",
                False,
            ):

                safety_reason = corrected_safety.get(
                    "reason",
                    "Corrected SQL failed the safety policy.",
                )

                logger.warning(
                    "Corrected SQL blocked by safety guard: %s",
                    safety_reason,
                )

                latency_ms = int(
                    (
                        time.monotonic()
                        - start_time
                    )
                    * 1000
                )

                return {
                    "sql": generated_sql,
                    "results": [],
                    "tables_used": _extract_table_names(
                        generated_sql
                    ),

                    "requires_approval": False,
                    "approval_reason": "",

                    "resource_guard": {
                        "decision": "BLOCK",
                        "risk_level": corrected_safety.get(
                            "risk_level",
                            "CRITICAL",
                        ),
                        "violations": [
                            "SQL_SAFETY",
                        ],
                        "reason": safety_reason,
                    },

                    "confidence": calculate_confidence(
                        sql_safe=False,
                        schema_valid=False,
                        resource_decision="BLOCK",
                        execution_success=False,
                        result_quality=0,
                        table_correct=False,
                    ),

                    "latency_ms": latency_ms,
                    "error": safety_reason,
                }

            # ------------------------------------------------
            # Update referenced tables after correction.
            # ------------------------------------------------

            tables_used = _extract_table_names(
                generated_sql
            )

        # ====================================================
        # STEP 12 — Query resource guard
        # ====================================================

        resource_guard = check_query_resources(
            generated_sql
        )

        resource_decision = resource_guard.get(
            "decision",
            "BLOCK",
        )

        resource_risk_level = resource_guard.get(
            "risk_level",
            "HIGH",
        )

        resource_violations = resource_guard.get(
            "violations",
            [],
        )

        resource_reason = resource_guard.get(
            "reason",
            "",
        )

        logger.info(
            "Query resource check: "
            "decision=%s risk=%s violations=%s",
            resource_decision,
            resource_risk_level,
            resource_violations,
        )

        # ----------------------------------------------------
        # Resource BLOCK
        # ----------------------------------------------------

        if resource_decision == "BLOCK":

            latency_ms = int(
                (
                    time.monotonic()
                    - start_time
                )
                * 1000
            )

            logger.warning(
                "Query blocked by resource guard: "
                "risk=%s violations=%s reason=%s",
                resource_risk_level,
                resource_violations,
                resource_reason,
            )

            _log_query(
                question,
                generated_sql,
                latency_ms,
                tables_used,
                error=resource_reason,
            )

            confidence = calculate_confidence(
                sql_safe=True,
                schema_valid=True,
                resource_decision="BLOCK",
                execution_success=False,
                result_quality=0,
                table_correct=False,
            )

            return {
                "sql": generated_sql,
                "results": [],
                "tables_used": tables_used,

                "requires_approval": False,
                "approval_reason": "",

                "resource_guard": {
                    "decision": resource_decision,
                    "risk_level": resource_risk_level,
                    "violations": resource_violations,
                    "reason": resource_reason,
                },

                "confidence": confidence,

                "latency_ms": latency_ms,
                "error": resource_reason,
            }

        # ----------------------------------------------------
        # Resource WARN
        # ----------------------------------------------------

        if resource_decision == "WARN":

            logger.warning(
                "Query resource warning: "
                "risk=%s violations=%s reason=%s",
                resource_risk_level,
                resource_violations,
                resource_reason,
            )

        # ====================================================
        # STEP 13 — HITL guard
        # ====================================================

        hitl_result = check_sql(
            generated_sql
        )

        requires_approval = hitl_result.get(
            "requires_approval",
            False,
        )

        approval_reason = hitl_result.get(
            "reason",
            "",
        )

        # ----------------------------------------------------
        # HITL approval required
        # ----------------------------------------------------

        if requires_approval:

            latency_ms = int(
                (
                    time.monotonic()
                    - start_time
                )
                * 1000
            )

            logger.warning(
                "HITL approval required: %s",
                approval_reason,
            )

            _log_query(
                question,
                generated_sql,
                latency_ms,
                tables_used,
                error=None,
            )

            confidence = calculate_confidence(
                sql_safe=False,
                schema_valid=True,
                resource_decision=resource_decision,
                execution_success=False,
                result_quality=0,
                table_correct=False,
            )

            return {
                "sql": generated_sql,
                "results": [],
                "tables_used": tables_used,

                "requires_approval": False,
                "approval_reason": "",

                "resource_guard": {
                    "decision": resource_decision,
                    "risk_level": resource_risk_level,
                    "violations": resource_violations,
                    "reason": resource_reason,
                },

                "semantic_evaluation": {
                    "is_correct": False,
                    "score": 0.0,
                    "reason": (
                        "Semantic evaluation was skipped because "
                        "the query was blocked by the resource guard."
                    ),
                    "issues": [
                        "RESOURCE_GUARD_BLOCKED",
                    ],
                },

                "explanation": {
                    "summary": (
                        "The query was blocked by the resource "
                        "governance guard before execution."
                    ),
                    "tables_used": tables_used,
                    "operation_count": 0,
                },

                "visualization": {
                    "recommended": False,
                    "chart_type": None,
                    "x_axis": None,
                    "y_axis": None,
                    "reason": (
                        "No visualization is generated for "
                        "a blocked query."
                    ),
                    "chart": {
                        "rendered": False,
                        "chart_type": None,
                    },
                },

                "confidence": confidence,

                "cache": {
                    "hit": False,
                },

                "latency_ms": latency_ms,
                "error": resource_reason,
            }

        # ====================================================
        # STEP 14 — Execute SQL + Automatic Correction Retry
        # ====================================================

        execution_success = False
        results: list[dict[str, Any]] = []

        execution_retry_count = 0
        sql_correction_attempted = False
        sql_correction_applied = False

        while True:

            # ------------------------------------------------
            # Execute current SQL candidate
            # ------------------------------------------------

            try:

                results = await asyncio.to_thread(
                    _execute_sql,
                    generated_sql,
                )

                execution_success = True

                logger.info(
                    "SQL execution successful: rows=%d "
                    "attempt=%d",
                    len(results),
                    execution_retry_count + 1,
                )

                break

            except Exception as exc:

                execution_success = False

                execution_error = str(exc)

                logger.warning(
                    "SQL execution failed "
                    "(attempt %d/%d): %s",
                    execution_retry_count + 1,
                    MAX_SQL_RETRIES + 1,
                    execution_error,
                )

                # ------------------------------------------------
                # Determine whether execution error is retryable
                # ------------------------------------------------

                retryable = is_retryable_error(
                    execution_error
                )

                if (
                    not retryable
                    or execution_retry_count >= MAX_SQL_RETRIES
                ):

                    latency_ms = int(
                        (
                            time.monotonic()
                            - start_time
                        )
                        * 1000
                    )

                    _log_query(
                        question,
                        generated_sql,
                        latency_ms,
                        tables_used,
                        error=execution_error,
                    )

                    confidence = calculate_confidence(
                        sql_safe=True,
                        schema_valid=True,
                        resource_decision=resource_decision,
                        execution_success=False,
                        result_quality=0,
                        table_correct=False,
                    )

                    logger.warning(
                        "SQL execution retry stopped. "
                        "retryable=%s retries=%d/%d",
                        retryable,
                        execution_retry_count,
                        MAX_SQL_RETRIES,
                    )

                    return {
                        "sql": generated_sql,
                        "results": [],
                        "tables_used": tables_used,

                        "semantic_evaluation": {},

                        "explanation": {},

                        "visualization": {},

                        "cache": {
                            "hit": False,
                        },

                        "requires_approval": False,
                        "approval_reason": "",

                        "resource_guard": {
                            "decision": resource_decision,
                            "risk_level": "HIGH",
                            "violations": [
                                "SQL_EXECUTION",
                            ],
                            "reason": execution_error,
                        },

                        "sql_retry": {
                            "attempted": sql_correction_attempted,
                            "retry_count": execution_retry_count,
                            "max_retries": MAX_SQL_RETRIES,
                            "corrected": sql_correction_applied,
                        },

                        "confidence": confidence,

                        "latency_ms": latency_ms,

                        "error": execution_error,
                    }
                
                # ------------------------------------------------
                # Automatic SQL correction
                # ------------------------------------------------

                execution_retry_count += 1
                sql_correction_attempted = True

                logger.info(
                    "Attempting automatic SQL correction "
                    "after execution failure "
                    "(retry %d/%d)",
                    execution_retry_count,
                    MAX_SQL_RETRIES,
                )

                try:

                    generated_sql = await _correct_sql(
                        question=question,
                        previous_sql=generated_sql,
                        validation_error=execution_error,
                        schema_context=schema_context,
                        llm=llm,
                    )

                except Exception as exc:

                    latency_ms = int(
                        (
                            time.monotonic()
                            - start_time
                        )
                        * 1000
                    )

                    error_msg = (
                        "Automatic SQL correction failed: "
                        f"{exc}"
                    )

                    logger.warning(
                        error_msg
                    )

                    _log_query(
                        question,
                        generated_sql,
                        latency_ms,
                        tables_used,
                        error=error_msg,
                    )

                    confidence = calculate_confidence(
                        sql_safe=True,
                        schema_valid=True,
                        resource_decision=resource_decision,
                        execution_success=False,
                        result_quality=0,
                        table_correct=False,
                    )

                    return {
                        "sql": generated_sql,
                        "results": [],
                        "tables_used": tables_used,

                        "semantic_evaluation": {},

                        "explanation": {},

                        "visualization": {},

                        "cache": {
                            "hit": False,
                        },

                        "requires_approval": False,
                        "approval_reason": "",

                        "resource_guard": {
                            "decision": resource_decision,
                            "risk_level": "HIGH",
                            "violations": [
                                "SQL_CORRECTION_FAILED",
                            ],
                            "reason": error_msg,
                        },

                        "sql_retry": {
                            "attempted": True,
                            "retry_count": execution_retry_count,
                            "max_retries": MAX_SQL_RETRIES,
                            "corrected": False,
                        },

                        "confidence": confidence,

                        "latency_ms": latency_ms,

                        "error": error_msg,
                    }

                logger.info(
                    "Corrected SQL generated:\n%s",
                    generated_sql,
                )

                # ------------------------------------------------
                # Safety validation of corrected SQL
                # ------------------------------------------------

                corrected_safety = check_sql(
                    generated_sql
                )

                if not corrected_safety.get(
                    "allowed",
                    False,
                ):

                    safety_reason = corrected_safety.get(
                        "reason",
                        "Corrected SQL failed the safety policy.",
                    )

                    latency_ms = int(
                        (
                            time.monotonic()
                            - start_time
                        )
                        * 1000
                    )

                    logger.warning(
                        "Corrected SQL blocked by safety guard: %s",
                        safety_reason,
                    )

                    _log_query(
                        question,
                        generated_sql,
                        latency_ms,
                        tables_used,
                        error=safety_reason,
                    )

                    confidence = calculate_confidence(
                        sql_safe=False,
                        schema_valid=False,
                        resource_decision="BLOCK",
                        execution_success=False,
                        result_quality=0,
                        table_correct=False,
                    )

                    return {
                        "sql": generated_sql,
                        "results": [],
                        "tables_used": _extract_table_names(
                            generated_sql
                        ),

                        "semantic_evaluation": {},

                        "explanation": {},

                        "visualization": {},

                        "cache": {
                            "hit": False,
                        },

                        "requires_approval": False,
                        "approval_reason": "",

                        "resource_guard": {
                            "decision": "BLOCK",
                            "risk_level": corrected_safety.get(
                                "risk_level",
                                "CRITICAL",
                            ),
                            "violations": [
                                "SQL_SAFETY",
                            ],
                            "reason": safety_reason,
                        },

                        "sql_retry": {
                            "attempted": True,
                            "retry_count": execution_retry_count,
                            "max_retries": MAX_SQL_RETRIES,
                            "corrected": False,
                        },

                        "confidence": confidence,

                        "latency_ms": latency_ms,

                        "error": safety_reason,
                    }

                # ------------------------------------------------
                # Schema validation of corrected SQL
                # ------------------------------------------------

                corrected_valid, corrected_schema_error = (
                    validate_sql_schema(
                        generated_sql
                    )
                )

                if not corrected_valid:

                    schema_reason = (
                        corrected_schema_error
                        or "Corrected SQL failed schema validation."
                    )

                    logger.warning(
                        "Corrected SQL failed schema validation: %s",
                        schema_reason,
                    )

                    # The corrected SQL becomes the next candidate.
                    #
                    # The next loop iteration will execute it. If the
                    # database reports another retryable SQL error,
                    # another bounded correction attempt can occur.

                    if not is_retryable_error(
                        schema_reason
                    ):

                        latency_ms = int(
                            (
                                time.monotonic()
                                - start_time
                            )
                            * 1000
                        )

                        _log_query(
                            question,
                            generated_sql,
                            latency_ms,
                            tables_used,
                            error=schema_reason,
                        )

                        confidence = calculate_confidence(
                            sql_safe=True,
                            schema_valid=False,
                            resource_decision="BLOCK",
                            execution_success=False,
                            result_quality=0,
                            table_correct=False,
                        )

                        return {
                            "sql": generated_sql,
                            "results": [],
                            "tables_used": _extract_table_names(
                                generated_sql
                            ),

                            "semantic_evaluation": {},

                            "explanation": {},

                            "visualization": {},

                            "cache": {
                                "hit": False,
                            },

                            "requires_approval": False,
                            "approval_reason": "",

                            "resource_guard": {
                                "decision": "BLOCK",
                                "risk_level": "HIGH",
                                "violations": [
                                    "SCHEMA_VALIDATION",
                                ],
                                "reason": schema_reason,
                            },

                            "sql_retry": {
                                "attempted": True,
                                "retry_count": execution_retry_count,
                                "max_retries": MAX_SQL_RETRIES,
                                "corrected": False,
                            },

                            "confidence": confidence,

                            "latency_ms": latency_ms,

                            "error": schema_reason,
                        }

                    # Retry loop will execute the candidate.
                    continue

                # ------------------------------------------------
                # Resource validation of corrected SQL
                # ------------------------------------------------

                corrected_resource = check_query_resources(
                    generated_sql
                )

                corrected_resource_decision = (
                    corrected_resource.get(
                        "decision",
                        "BLOCK",
                    )
                )

                if corrected_resource_decision == "BLOCK":

                    resource_reason = corrected_resource.get(
                        "reason",
                        "Corrected SQL failed resource validation.",
                    )

                    latency_ms = int(
                        (
                            time.monotonic()
                            - start_time
                        )
                        * 1000
                    )

                    logger.warning(
                        "Corrected SQL blocked by resource guard: %s",
                        resource_reason,
                    )

                    _log_query(
                        question,
                        generated_sql,
                        latency_ms,
                        tables_used,
                        error=resource_reason,
                    )

                    return {
                        "sql": generated_sql,
                        "results": [],
                        "tables_used": _extract_table_names(
                            generated_sql
                        ),

                        "semantic_evaluation": {},

                        "explanation": {},

                        "visualization": {},

                        "cache": {
                            "hit": False,
                        },

                        "requires_approval": False,
                        "approval_reason": "",

                        "resource_guard": {
                            "decision": "BLOCK",
                            "risk_level": corrected_resource.get(
                                "risk_level",
                                "HIGH",
                            ),
                            "violations": [
                                "RESOURCE_GUARD",
                            ],
                            "reason": resource_reason,
                        },

                        "sql_retry": {
                            "attempted": True,
                            "retry_count": execution_retry_count,
                            "max_retries": MAX_SQL_RETRIES,
                            "corrected": False,
                        },

                        "confidence": calculate_confidence(
                            sql_safe=True,
                            schema_valid=corrected_valid,
                            resource_decision="BLOCK",
                            execution_success=False,
                            result_quality=0,
                            table_correct=False,
                        ),

                        "latency_ms": latency_ms,

                        "error": resource_reason,
                    }

                # ------------------------------------------------
                # Accept corrected SQL
                # ------------------------------------------------

                tables_used = _extract_table_names(
                    generated_sql
                )

                sql_correction_applied = True

                logger.info(
                    "Corrected SQL accepted. "
                    "Continuing execution with corrected SQL."
                )

                # ------------------------------------------------
                # Continue loop and execute corrected SQL
                # ------------------------------------------------

                continue

        # ====================================================
        # STEP — Semantic evaluation
        # ====================================================

        try:

            semantic_evaluation = evaluate_semantics(
                question=question,
                generated_sql=generated_sql,
                results=results,
            )

            logger.info(
                "Semantic evaluation: correct=%s score=%.2f issues=%s",
                semantic_evaluation["is_correct"],
                semantic_evaluation["score"],
                semantic_evaluation["issues"],
            )

        except Exception as exc:

            logger.warning(
                "Semantic evaluation unavailable: %s",
                exc,
            )

            semantic_evaluation = {
                "is_correct": None,
                "score": None,
                "reason": (
                    "Semantic evaluation was temporarily unavailable."
                ),
                "issues": [
                    "SEMANTIC_EVALUATION_UNAVAILABLE",
                ],
            }

        # ====================================================
        # STEP 15 — Calculate result quality
        # ====================================================

        if semantic_evaluation["score"] is not None:

            result_quality = (
                semantic_evaluation["score"]
                * 100.0
            )

        else:

            result_quality = 0.0

        # ====================================================
        # STEP 16 — Deterministic table correctness
        # ====================================================

        try:

            table_correctness = check_table_correctness(
                sql=generated_sql,
            )

            table_correct = table_correctness["table_correct"]

            logger.info(
                "Table correctness: correct=%s tables=%s",
                table_correct,
                table_correctness.get("tables_used"),
            )

        except Exception as exc:

            logger.warning(
                "Table correctness verification unavailable: %s",
                exc,
            )

            table_correctness = {
                "table_correct": None,
                "tables_used": [],
                "invalid_tables": [],
                "invalid_columns": [],
                "issues": [
                    "TABLE_CORRECTNESS_UNAVAILABLE",
                ],
            }

            table_correct = None

        # ====================================================
        # STEP 17 — Calculate confidence
        # ====================================================
        confidence = calculate_confidence(
                sql_safe=True,
                schema_valid=True,
                resource_decision=resource_decision,
                execution_success=execution_success,
                result_quality=result_quality,
                table_correct=table_correct,
                semantic_correct=semantic_evaluation.get(
                        "is_correct"
                ),
            )

        logger.info(
            "Confidence score: %.2f (%s)",
            confidence["score"],
            confidence["level"],
        )
        
        # ====================================================
        # STEP 18 — Calculate latency
        # ====================================================

        latency_ms = int(
            (
                time.monotonic()
                - start_time
            )
            * 1000
        )

        # ====================================================
        # STEP 18 — Log successful query
        # ====================================================

        _log_query(
            question,
            generated_sql,
            latency_ms,
            tables_used,
            error=None,
        )

        # ====================================================
        # STEP 19 — Query explanation
        # ====================================================

        explanation = await explain_query(
            question=question,
            sql=generated_sql,
            tables_used=tables_used,
            llm=_get_llm(),
        )

        logger.info(
            "Query explanation generated: %s",
            explanation["summary"],
        )

        # ====================================================
        # STEP 20 — Result visualization recommendation
        # ====================================================

        visualization = recommend_visualization(
            results
        )

        logger.info(
            "Visualization recommendation: "
            "recommended=%s chart_type=%s",
            visualization["recommended"],
            visualization["chart_type"],
        )

        chart = render_chart(
            results,
            visualization,
        )

        logger.info(
            "Chart rendering: rendered=%s type=%s",
            chart["rendered"],
            chart.get("chart_type"),
        )

        # ====================================================
        # STEP 20 — Successful response
        # ====================================================

        response = {
            "sql": generated_sql,
            "results": results,
            "tables_used": tables_used,

            "semantic_evaluation": semantic_evaluation,

            "explanation": explanation,

            "visualization": {
                **visualization,
                "chart": chart,
            },
            
            "cache": {
                "hit": False,
            },

            "requires_approval": False,
            "approval_reason": "",

            "resource_guard": {
                "decision": resource_decision,
                "risk_level": resource_risk_level,
                "violations": resource_violations,
                "reason": resource_reason,
            },

            "confidence": confidence,

            "latency_ms": latency_ms,

            "error": "",
        }

        # ====================================================
        # STEP — Store successful response in cache
        # ====================================================

        set_cached_response(
            question,
            response,
        )

        logger.info(
            "Query response cached for question: %s",
            question,
        )

        # ====================================================
        # STEP — Record successful telemetry event
        # ====================================================

        telemetry_event = QueryEvent(
            question=question,
            status="SUCCESS",
            latency_ms=latency_ms,

            sql_generated=bool(generated_sql),
            sql_safe=True,

            sql_correction_attempted=sql_correction_attempted,
            sql_correction_count=execution_retry_count,
            sql_correction_applied=sql_correction_applied,

            cache_hit=False,

            resource_decision=resource_decision,

            semantic_correct=semantic_evaluation.get(
                "is_correct"
            ),
            semantic_score=semantic_evaluation.get(
                "score",
                0.0,
            ),

            confidence_score=confidence["score"],
            confidence_level=confidence["level"],

            tables_used=tables_used,

            error="",
        )

        record_event(telemetry_event)

        logger.info(
            "Telemetry event recorded: request_id=%s",
            telemetry_event.request_id,
        )

        return response
    
    # ========================================================
    # GLOBAL ERROR HANDLING
    # ========================================================

    except Exception as exc:

        latency_ms = int(
            (
                time.monotonic()
                - start_time
            )
            * 1000
        )

        error_msg = str(
            exc
        )

        logger.error(
            "run_query failed: %s",
            exc,
            exc_info=True,
        )

        _log_query(
            question,
            generated_sql,
            latency_ms,
            tables_used,
            error=error_msg,
        )

        raise