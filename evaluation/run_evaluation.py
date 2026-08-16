import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Run evaluation from project root so relative paths work:
# ./data/olist.db
# ./chroma_store
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.sql_chain import run_query
from model.database import get_engine
from sqlalchemy import text


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(__file__).parent / "dataset.json"

OUTPUT_PATH = (
    Path(__file__).parent
    / "evaluation_results.json"
)

RESUME_EVALUATION = True

# ============================================================
# DATASET
# ============================================================

def load_dataset() -> list[dict[str, Any]]:
    """
    Load the evaluation dataset.
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ============================================================
# EXPECTED SQL EXECUTION
# ============================================================

def execute_expected_sql(
    sql: str
) -> list[dict[str, Any]]:
    """
    Execute ground-truth SQL directly against
    the database.

    This gives us expected results without
    hardcoding result values.
    """

    engine = get_engine()

    with engine.connect() as conn:

        result = conn.execute(
            text(sql)
        )

        columns = list(
            result.keys()
        )

        rows = result.fetchall()

        return [
            dict(
                zip(
                    columns,
                    row
                )
            )
            for row in rows
        ]


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_value(
    value: Any
) -> Any:
    """
    Normalize individual values before comparison.
    """

    if value is None:
        return None

    if isinstance(value, float):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return str(value)

        return round(
            value,
            6
        )

    return value


def normalize_rows(
    rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Normalize row values while preserving
    column names.
    """

    normalized = []

    for row in rows:

        normalized_row = {
            key: normalize_value(value)
            for key, value in row.items()
        }

        normalized.append(
            normalized_row
        )

    return normalized


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_result_columns(
    rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Normalize result column names.

    This allows semantically equivalent queries
    to pass even when aliases differ.

    Example:

        total_revenue

    vs

        total_delivered_revenue

    The evaluator primarily compares values
    after column-name normalization.
    """

    normalized = []

    for row in rows:

        normalized_row = {}

        for key, value in row.items():

            normalized_key = str(
                key
            ).strip().lower()

            normalized_row[
                normalized_key
            ] = normalize_value(value)

        normalized.append(
            normalized_row
        )

    return normalized


def row_values(
    rows: list[dict[str, Any]]
) -> list[tuple]:
    """
    Extract row values without depending
    on column aliases.

    Example:

        {"total_revenue": 100}

    and

        {"average_revenue": 100}

    both become:

        [(100,)]
    """

    normalized = []

    for row in rows:

        values = tuple(
            normalize_value(value)
            for value in row.values()
        )

        normalized.append(
            values
        )

    return normalized


def results_match(
    generated: list[dict[str, Any]],
    expected: list[dict[str, Any]]
) -> bool:
    """
    Compare generated and expected results.

    Comparison intentionally ignores column aliases
    because aliases do not change query semantics.
    """

    generated = normalize_result_columns(
        generated
    )

    expected = normalize_result_columns(
        expected
    )

    # Different number of rows
    if len(generated) != len(expected):
        return False

    # Compare values rather than aliases
    generated_values = row_values(
        generated
    )

    expected_values = row_values(
        expected
    )

    return generated_values == expected_values


# ============================================================
# TABLE COMPARISON
# ============================================================

def tables_match(
    generated_tables: list[str],
    expected_tables: list[str]
) -> bool:
    """
    Compare tables used by generated SQL
    against expected tables.
    """

    generated = {
        table.lower()
        for table in generated_tables
    }

    expected = {
        table.lower()
        for table in expected_tables
    }

    return generated == expected


# ============================================================
# SQL NORMALIZATION
# ============================================================

def normalize_sql(
    sql: str
) -> str:
    """
    Normalize SQL for exact SQL comparison.

    This removes superficial formatting differences
    such as:

    - whitespace
    - trailing semicolon
    - casing
    """

    if not sql:
        return ""

    sql = sql.strip()

    # Remove markdown code fences
    sql = re.sub(
        r"```sql",
        "",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"```",
        "",
        sql
    )

    # Normalize whitespace
    sql = re.sub(
        r"\s+",
        " ",
        sql
    )

    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()

    # Case-insensitive comparison
    sql = sql.lower()

    return sql


def sql_exact_match(
    generated_sql: str,
    expected_sql: str
) -> bool:
    """
    Check whether generated SQL matches expected SQL
    after superficial normalization.
    """

    generated = normalize_sql(
        generated_sql
    )

    expected = normalize_sql(
        expected_sql
    )

    if not generated or not expected:
        return False

    return generated == expected


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def is_rate_limited(
    error: str
) -> bool:
    """
    Detect Gemini/API rate-limit failures.
    """

    if not error:
        return False

    error_lower = error.lower()

    indicators = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "ratelimit",
        "generate_content_free_tier_requests",
    ]

    return any(
        indicator in error_lower
        for indicator in indicators
    )


def classify_error(
    error: str
) -> str:
    """
    Classify errors into production-style categories.
    """

    if not error:
        return ""

    if is_rate_limited(error):
        return "RATE_LIMITED"

    error_lower = error.lower()

    if (
        "evaluation" in error_lower
        or "expected_sql" in error_lower
        or "dataset" in error_lower
    ):
        return "EVALUATION_ERROR"

    return "EXECUTION_ERROR"


# ============================================================
# STATUS CLASSIFICATION
# ============================================================

def classify_status(
    *,
    generated_sql: str,
    execution_success: bool,
    result_correct: bool,
    table_correct: bool,
    sql_correct: bool,
    error_type: str,
) -> str:
    """
    Determine the final production-style evaluation status.
    """

    # API quota/rate limit
    if error_type == "RATE_LIMITED":
        return "RATE_LIMITED"

    # Internal evaluator problem
    if error_type == "EVALUATION_ERROR":
        return "EVALUATION_ERROR"

    # SQL never executed
    if not execution_success:

        if not generated_sql:
            return "WRONG SQL"

        return "EXECUTION ERROR"

    # Perfect exact match
    if (
        sql_correct
        and table_correct
        and result_correct
    ):
        return "PASS"

    # Different SQL but same semantic result
    if (
        not sql_correct
        and table_correct
        and result_correct
    ):
        return "EQUIVALENT SQL"

    # Same tables but wrong answer
    if (
        table_correct
        and not result_correct
    ):
        return "WRONG RESULT"

    # Wrong tables / incorrect query structure
    return "WRONG SQL"


# ============================================================
# SINGLE QUESTION EVALUATION
# ============================================================

def classify_failure(
    status: str,
    generated_sql: str,
    expected_sql: str,
    error: str = "",
) -> tuple[str, str]:
    """
    Classify why an evaluation failed.

    Returns:
        (failure_category, failure_reason)
    """

    if status == "RATE_LIMITED":
        return (
            "RATE_LIMITED",
            "The LLM provider quota or rate limit was exceeded.",
        )

    if status == "EXECUTION ERROR":
        return (
            "EXECUTION_ERROR",
            error or "Generated SQL failed during execution.",
        )

    if status == "EVALUATION_ERROR":
        return (
            "EVALUATION_ERROR",
            error or "The evaluation process encountered an error.",
        )

    if status == "WRONG SQL":
        return (
            "SQL_GENERATION_ERROR",
            "Generated SQL did not match the expected SQL structure or semantics.",
        )

    if status == "WRONG RESULT":

        generated_lower = (
            generated_sql.lower()
        )

        expected_lower = (
            expected_sql.lower()
        )

        # Aggregation-related failure
        aggregation_functions = [
            "count(",
            "sum(",
            "avg(",
            "min(",
            "max(",
        ]

        generated_aggregations = [
            fn
            for fn in aggregation_functions
            if fn in generated_lower
        ]

        expected_aggregations = [
            fn
            for fn in aggregation_functions
            if fn in expected_lower
        ]

        if (
            generated_aggregations
            != expected_aggregations
        ):
            return (
                "AGGREGATION_ERROR",
                "Generated SQL uses a different aggregation strategy from the expected query.",
            )

        # Join-related failure
        generated_has_join = (
            " join "
            in generated_lower
        )

        expected_has_join = (
            " join "
            in expected_lower
        )

        if (
            generated_has_join
            != expected_has_join
        ):
            return (
                "JOIN_ERROR",
                "Generated SQL uses a different join strategy from the expected query.",
            )

        # Filter-related failure
        generated_has_where = (
            " where "
            in generated_lower
        )

        expected_has_where = (
            " where "
            in expected_lower
        )

        if (
            generated_has_where
            != expected_has_where
        ):
            return (
                "FILTER_ERROR",
                "Generated SQL uses a different filtering strategy from the expected query.",
            )

        return (
            "SEMANTIC_RESULT_ERROR",
            "Generated SQL executed successfully but produced an incorrect result.",
        )

    return (
        "",
        "",
    )
def classify_failure(
    status: str,
    generated_sql: str,
    expected_sql: str,
    error: str = "",
) -> tuple[str, str]:
    """
    Classify the reason for an unsuccessful evaluation.

    Returns:
        failure_category, failure_reason
    """

    if status == "RATE_LIMITED":
        return (
            "RATE_LIMITED",
            "LLM provider quota or rate limit was exceeded.",
        )

    if status == "EXECUTION ERROR":
        return (
            "EXECUTION_ERROR",
            error
            or "Generated SQL failed during execution.",
        )

    if status == "EVALUATION_ERROR":
        return (
            "EVALUATION_ERROR",
            error
            or "The evaluation pipeline encountered an error.",
        )

    if status == "WRONG SQL":
        return (
            "SQL_GENERATION_ERROR",
            "Generated SQL did not match the expected SQL structure.",
        )

    if status == "WRONG RESULT":

        generated_lower = (
            generated_sql.lower()
        )

        expected_lower = (
            expected_sql.lower()
        )

        # ----------------------------------------------------
        # Aggregation mismatch
        # ----------------------------------------------------

        aggregation_functions = [
            "count(",
            "sum(",
            "avg(",
            "min(",
            "max(",
        ]

        generated_aggregations = [
            fn
            for fn in aggregation_functions
            if fn in generated_lower
        ]

        expected_aggregations = [
            fn
            for fn in aggregation_functions
            if fn in expected_lower
        ]

        if (
            generated_aggregations
            != expected_aggregations
        ):
            return (
                "AGGREGATION_ERROR",
                "Generated SQL uses a different aggregation strategy from the expected SQL.",
            )

        # ----------------------------------------------------
        # JOIN mismatch
        # ----------------------------------------------------

        generated_has_join = (
            " join "
            in generated_lower
        )

        expected_has_join = (
            " join "
            in expected_lower
        )

        if (
            generated_has_join
            != expected_has_join
        ):
            return (
                "JOIN_ERROR",
                "Generated SQL uses a different join strategy from the expected SQL.",
            )

        # ----------------------------------------------------
        # WHERE/filter mismatch
        # ----------------------------------------------------

        generated_has_where = (
            " where "
            in generated_lower
        )

        expected_has_where = (
            " where "
            in expected_lower
        )

        if (
            generated_has_where
            != expected_has_where
        ):
            return (
                "FILTER_ERROR",
                "Generated SQL uses a different filtering strategy from the expected SQL.",
            )

        # ----------------------------------------------------
        # Generic semantic error
        # ----------------------------------------------------

        return (
            "SEMANTIC_RESULT_ERROR",
            "Generated SQL executed successfully but produced an incorrect result.",
        )

    return (
        "",
        "",
    )

async def evaluate_question(
    item: dict[str, Any]
) -> dict[str, Any]:

    question = item["question"]

    expected_sql = item[
        "expected_sql"
    ]

    expected_tables = set(
        item["expected_tables"]
    )

    start_time = time.perf_counter()

    generated_sql = ""
    generated_results = []
    generated_tables = set()

    try:

        # ----------------------------------------------------
        # 1. Execute expected SQL
        # ----------------------------------------------------

        expected_results = (
            execute_expected_sql(
                expected_sql
            )
        )

        # ----------------------------------------------------
        # 2. Execute model-generated query
        # ----------------------------------------------------

        try:

            result = await run_query(
                question
            )

        except Exception as exc:

            error_message = str(
                exc
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            # ------------------------------------------------
            # Rate-limit detection
            # ------------------------------------------------

            if (
                "RESOURCE_EXHAUSTED"
                in error_message
                or "429"
                in error_message
                or "quota"
                in error_message.lower()
            ):

                status = (
                    "RATE_LIMITED"
                )

                failure_category = (
                    "RATE_LIMITED"
                )

                failure_reason = (
                    "LLM provider quota or rate limit was exceeded."
                )

                return {
                    "id": item["id"],
                    "question": question,

                    "status": status,

                    "generated_sql": "",
                    "expected_sql": expected_sql,

                    "sql_correct": False,
                    "table_correct": False,
                    "result_correct": False,

                    "execution_success": False,

                    "generated_tables": [],
                    "expected_tables": sorted(
                        expected_tables
                    ),

                    "generated_results": [],
                    "expected_results": expected_results,

                    "latency_ms": round(
                        elapsed * 1000,
                        2
                    ),

                    "error_type": (
                        "RATE_LIMITED"
                    ),

                    "failure_category": (
                        failure_category
                    ),

                    "failure_reason": (
                        failure_reason
                    ),

                    "error": error_message,
                }

            # -----------------------------------------------
            # Other model execution errors
            # -----------------------------------------------

            status = (
                "EVALUATION_ERROR"
            )

            failure_category = (
                "EVALUATION_ERROR"
            )

            failure_reason = (
                error_message
                or
                "The model query could not be executed."
            )

            return {
                "id": item["id"],
                "question": question,

                "status": status,

                "generated_sql": "",
                "expected_sql": expected_sql,

                "sql_correct": False,
                "table_correct": False,
                "result_correct": False,

                "execution_success": False,

                "generated_tables": [],
                "expected_tables": sorted(
                    expected_tables
                ),

                "generated_results": [],
                "expected_results": expected_results,

                "latency_ms": round(
                    elapsed * 1000,
                    2
                ),

                "error_type": (
                    "EVALUATION_ERROR"
                ),

                "failure_category": (
                    failure_category
                ),

                "failure_reason": (
                    failure_reason
                ),

                "error": error_message,
            }

        # ----------------------------------------------------
        # 3. Extract generated query information
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - start_time
        )

        generated_sql = result.get(
            "sql",
            ""
        )

        generated_results = result.get(
            "results",
            []
        )

        generated_tables = set(
            result.get(
                "tables_used",
                []
            )
        )

        error = result.get(
            "error",
            ""
        )

        # ----------------------------------------------------
        # 4. Determine execution success
        # ----------------------------------------------------

        execution_success = bool(
            generated_sql
            and not error
        )

        # ----------------------------------------------------
        # 5. Individual evaluation dimensions
        # ----------------------------------------------------

        sql_correct = False

        if generated_sql:

            sql_correct = (
                sql_exact_match(
                    generated_sql,
                    expected_sql
                )
            )

        table_correct = (
            tables_match(
                list(generated_tables),
                list(expected_tables)
            )
        )

        result_correct = False

        if execution_success:

            result_correct = (
                results_match(
                    generated_results,
                    expected_results
                )
            )

        # ----------------------------------------------------
        # 6. Error classification
        # ----------------------------------------------------

        error_type = classify_error(
            error
        )

        # ----------------------------------------------------
        # 7. Final status
        # ----------------------------------------------------

        status = classify_status(
            generated_sql=generated_sql,
            execution_success=execution_success,
            result_correct=result_correct,
            table_correct=table_correct,
            sql_correct=sql_correct,
            error_type=error_type,
        )

        # ----------------------------------------------------
        # 8. Failure analysis
        # ----------------------------------------------------

        failure_category = ""
        failure_reason = ""

        if status not in {
            "PASS",
            "EQUIVALENT SQL",
        }:

            (
                failure_category,
                failure_reason,
            ) = classify_failure(
                status=status,
                generated_sql=generated_sql,
                expected_sql=expected_sql,
                error=error,
            )

        # ----------------------------------------------------
        # 9. Return evaluation result
        # ----------------------------------------------------

        return {
            "id": item["id"],
            "question": question,

            "status": status,

            "generated_sql": generated_sql,
            "expected_sql": expected_sql,

            "sql_correct": sql_correct,
            "table_correct": table_correct,
            "result_correct": result_correct,

            "execution_success": (
                execution_success
            ),

            "generated_tables": sorted(
                generated_tables
            ),

            "expected_tables": sorted(
                expected_tables
            ),

            "generated_results": (
                generated_results
            ),

            "expected_results": (
                expected_results
            ),

            "latency_ms": round(
                elapsed * 1000,
                2
            ),

            "error_type": error_type,

            "failure_category": (
                failure_category
            ),

            "failure_reason": (
                failure_reason
            ),

            "error": error,
        }

    # ========================================================
    # Unexpected evaluation-pipeline error
    # ========================================================

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        error_message = str(
            exc
        )

        error_type = (
            classify_error(
                error_message
            )
        )

        if (
            error_type
            == "RATE_LIMITED"
        ):

            status = (
                "RATE_LIMITED"
            )

            failure_category = (
                "RATE_LIMITED"
            )

            failure_reason = (
                "LLM provider quota or rate limit was exceeded."
            )

        else:

            status = (
                "EVALUATION_ERROR"
            )

            failure_category = (
                "EVALUATION_ERROR"
            )

            failure_reason = (
                error_message
                or
                "The evaluation pipeline encountered an unexpected error."
            )

        return {
            "id": item["id"],
            "question": question,

            "status": status,

            "generated_sql": generated_sql,
            "expected_sql": expected_sql,

            "sql_correct": False,
            "table_correct": False,
            "result_correct": False,

            "execution_success": False,

            "generated_tables": sorted(
                generated_tables
            ),

            "expected_tables": sorted(
                expected_tables
            ),

            "generated_results": (
                generated_results
            ),

            "expected_results": [],

            "latency_ms": round(
                elapsed * 1000,
                2
            ),

            "error_type": (
                error_type
                or "EVALUATION_ERROR"
            ),

            "failure_category": (
                failure_category
            ),

            "failure_reason": (
                failure_reason
            ),

            "error": error_message,
        }


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_accuracy(
    results: list[dict[str, Any]],
    field: str
) -> float:
    """
    Calculate accuracy while excluding
    rate-limited questions.
    """

    evaluated = [
        r
        for r in results
        if r["status"]
        not in {
            "RATE_LIMITED",
            "EVALUATION_ERROR",
        }
    ]

    if not evaluated:
        return 0.0

    correct = sum(
        bool(r.get(field))
        for r in evaluated
    )

    return (
        correct
        / len(evaluated)
        * 100
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

async def run_evaluation():

    dataset = load_dataset()

    # ========================================================
    # RESUME PREVIOUS EVALUATION
    # ========================================================

    previous_results = {}

    if (
        RESUME_EVALUATION
        and OUTPUT_PATH.exists()
    ):

        try:

            with open(
                OUTPUT_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                previous_report = json.load(
                    f
                )

            for result in previous_report.get(
                "questions",
                []
            ):

                # Do not consider rate-limited or
                # evaluation-error questions completed.
                if result.get(
                    "status"
                ) not in {
                    "RATE_LIMITED",
                    "EVALUATION_ERROR",
                }:

                    previous_results[
                        result["id"]
                    ] = result

            print(
                f"Found "
                f"{len(previous_results)} "
                f"previously completed evaluations."
            )

        except (
            json.JSONDecodeError,
            OSError,
            TypeError,
        ):

            print(
                "Could not load previous "
                "evaluation results. "
                "Starting fresh."
            )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TEXT-TO-SQL LLM EVALUATION"
    )
    print("=" * 70)
    print()

    print(
        f"Evaluation dataset size: "
        f"{len(dataset)}"
    )

    print()

    # Preserve results from previous runs.
    results = list(
        previous_results.values()
    )

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    # Sequential execution helps reduce the
    # probability of hitting Gemini request limits.

    evaluation_stopped = False

    for index, item in enumerate(
        dataset,
        start=1
    ):

        # ----------------------------------------------------
        # Skip questions already evaluated
        # ----------------------------------------------------

        if (
            item["id"]
            in previous_results
        ):

            print(
                f"[{index}/{len(dataset)}] "
                f"{item['question']}"
            )

            print(
                "    SKIPPED "
                "(already evaluated)"
            )

            continue

        # ----------------------------------------------------
        # Evaluate new question
        # ----------------------------------------------------

        print(
            f"[{index}/{len(dataset)}] "
            f"{item['question']}"
        )

        evaluation = await evaluate_question(
            item
        )

        results.append(
            evaluation
        )

        print(
            f"    "
            f"{evaluation['status']}"
            f" | "
            f"{evaluation['latency_ms']} ms"
        )

        # ----------------------------------------------------
        # Stop immediately on rate limit
        # ----------------------------------------------------

        if evaluation["status"] == "RATE_LIMITED":

            evaluation_stopped = True

            print()
            print(
                "⚠️ Model rate limit reached."
            )

            print(
                "Stopping evaluation to avoid "
                "unnecessary API calls."
            )

            break

    # ========================================================
    # METRICS
    # ========================================================

    total = len(
        dataset
    )

    rate_limited_count = sum(
        r["status"] == "RATE_LIMITED"
        for r in results
    )

    evaluation_error_count = sum(
        r["status"] == "EVALUATION_ERROR"
        for r in results
    )

    execution_error_count = sum(
        r["status"] == "EXECUTION ERROR"
        for r in results
    )

    wrong_sql_count = sum(
        r["status"] == "WRONG SQL"
        for r in results
    )

    wrong_result_count = sum(
        r["status"] == "WRONG RESULT"
        for r in results
    )

    equivalent_sql_count = sum(
        r["status"] == "EQUIVALENT SQL"
        for r in results
    )

    pass_count = sum(
        r["status"] == "PASS"
        for r in results
    )

    # --------------------------------------------------------
    # Only completed evaluations should contribute to
    # accuracy and latency metrics.
    # --------------------------------------------------------

    evaluated_results = [
        r
        for r in results
        if r["status"]
        not in {
            "RATE_LIMITED",
            "EVALUATION_ERROR",
        }
    ]

    evaluated_count = len(
        evaluated_results
    )

    # Questions that have not been attempted yet.
    skipped_count = (
        total - len(results)
    )

    # ========================================================
    # ACCURACY METRICS
    # ========================================================

    sql_accuracy = calculate_accuracy(
        results,
        "sql_correct"
    )

    result_accuracy = calculate_accuracy(
        results,
        "result_correct"
    )

    table_accuracy = calculate_accuracy(
        results,
        "table_correct"
    )

    execution_accuracy = (
        (
            sum(
                r["execution_success"]
                for r in evaluated_results
            )
            / evaluated_count
            * 100
        )
        if evaluated_count
        else 0.0
    )

    # ========================================================
    # LATENCY
    # ========================================================

    evaluated_latencies = [
        r["latency_ms"]
        for r in evaluated_results
    ]

    if evaluated_latencies:

        average_latency = (
            statistics.mean(
                evaluated_latencies
            )
        )

        sorted_latencies = sorted(
            evaluated_latencies
        )

        p95_index = min(
            len(sorted_latencies) - 1,
            math.ceil(
                0.95
                * len(sorted_latencies)
            ) - 1,
        )

        p95_latency = (
            sorted_latencies[
                p95_index
            ]
        )

    else:

        average_latency = 0.0
        p95_latency = 0.0

    # ========================================================
    # OVERALL SCORE
    # ========================================================

    # Result correctness is weighted highest because
    # the final business answer must be correct.

    overall_score = (
        0.50 * result_accuracy
        + 0.30 * sql_accuracy
        + 0.20 * table_accuracy
    )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("TEXT-TO-SQL EVALUATION REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # Evaluation Summary
    # --------------------------------------------------------

    print()
    print("Evaluation Summary")
    print("-" * 70)

    summary_metrics = [
        ("Dataset size", total),
        ("Evaluated", evaluated_count),
        ("Skipped", skipped_count),
        ("Rate limited", rate_limited_count),
        ("Evaluation errors", evaluation_error_count),
        ("Evaluation stopped", str(evaluation_stopped)),
    ]

    for label, value in summary_metrics:
        print(f"{label:<28}: {value}")

    # --------------------------------------------------------
    # Accuracy Metrics
    # --------------------------------------------------------

    print()
    print("Accuracy Metrics")
    print("-" * 70)

    accuracy_metrics = [
        ("SQL correctness", sql_accuracy),
        ("Result correctness", result_accuracy),
        ("Table correctness", table_accuracy),
        ("SQL execution accuracy", execution_accuracy),
    ]

    for label, value in accuracy_metrics:
        print(f"{label:<28}: {value:.2f}%")

    # --------------------------------------------------------
    # Evaluation Outcomes
    # --------------------------------------------------------

    print()
    print("Evaluation Outcomes")
    print("-" * 70)

    status_counts = [
        ("PASS", pass_count),
        ("EQUIVALENT SQL", equivalent_sql_count),
        ("WRONG RESULT", wrong_result_count),
        ("WRONG SQL", wrong_sql_count),
        ("EXECUTION ERROR", execution_error_count),
        ("EVALUATION ERROR", evaluation_error_count),
        ("RATE LIMITED", rate_limited_count),
    ]

    for label, count in status_counts:
        print(f"{label:<28}: {count}")

    # --------------------------------------------------------
    # Failure Analysis
    # --------------------------------------------------------

    failure_categories = {}

    for evaluation in results:

        category = evaluation.get(
            "failure_category",
            ""
        )

        if not category:
            continue

        failure_categories[category] = (
            failure_categories.get(
                category,
                0
            ) + 1
        )

    print()
    print("Failure Analysis")
    print("-" * 70)

    if failure_categories:

        for category, count in sorted(
            failure_categories.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            print(
                f"{category:<28}: "
                f"{count}"
            )

    else:

        print(
            "No failures detected."
        )

        # --------------------------------------------------------
    # Failure Details
    # --------------------------------------------------------

    failed_evaluations = [
        evaluation
        for evaluation in results
        if evaluation.get("status")
        not in {
            "PASS",
            "EQUIVALENT SQL",
        }
    ]

    print()
    print("Failure Details")
    print("-" * 70)

    if failed_evaluations:

        for evaluation in failed_evaluations:

            print()
            print(
                f"{evaluation.get('id', 'UNKNOWN')} "
                f"| "
                f"{evaluation.get('status', 'UNKNOWN')}"
            )

            print(
                f"{'Question':<20}: "
                f"{evaluation.get('question', '')}"
            )

            print(
                f"{'Failure category':<20}: "
                f"{evaluation.get('failure_category', 'N/A')}"
            )

            print(
                f"{'Failure reason':<20}: "
                f"{evaluation.get('failure_reason', 'N/A')}"
            )

            print(
                f"{'Error type':<20}: "
                f"{evaluation.get('error_type', 'N/A')}"
            )

            print(
                f"{'Latency':<20}: "
                f"{evaluation.get('latency_ms', 0):.2f} ms"
            )

            generated_sql = evaluation.get(
                "generated_sql",
                ""
            )

            expected_sql = evaluation.get(
                "expected_sql",
                ""
            )

            if generated_sql:

                print()
                print("Generated SQL:")
                print(generated_sql)

            if expected_sql:

                print()
                print("Expected SQL:")
                print(expected_sql)

            print()
            print("-" * 70)

    else:

        print(
            "No failed evaluations."
        )

    # --------------------------------------------------------
    # Performance Metrics
    # --------------------------------------------------------

    print()
    print("Performance Metrics")
    print("-" * 70)

    performance_metrics = [
        ("Average latency", f"{average_latency:.2f} ms"),
        ("P95 latency", f"{p95_latency:.2f} ms"),
    ]

    for label, value in performance_metrics:
        print(f"{label:<28}: {value}")

    # --------------------------------------------------------
    # Overall Score
    # --------------------------------------------------------

    print()
    print("Overall Evaluation Score")
    print("-" * 70)

    print(
        f"{'Overall score':<28}: "
        f"{overall_score:.2f}%"
    )

    print("=" * 70)
    print()
    

    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    report = {
        "metrics": {

            "dataset_size": total,

            "evaluated_count": (
                evaluated_count
            ),

            "skipped_count": (
                skipped_count
            ),

            "rate_limited_count": (
                rate_limited_count
            ),

            "evaluation_stopped": (
                evaluation_stopped
            ),

            "evaluation_error_count": (
                evaluation_error_count
            ),

            "execution_error_count": (
                execution_error_count
            ),

            "pass_count": (
                pass_count
            ),

            "equivalent_sql_count": (
                equivalent_sql_count
            ),

            "wrong_result_count": (
                wrong_result_count
            ),

            "wrong_sql_count": (
                wrong_sql_count
            ),

            "sql_accuracy": round(
                sql_accuracy,
                2
            ),

            "result_accuracy": round(
                result_accuracy,
                2
            ),

            "table_accuracy": round(
                table_accuracy,
                2
            ),

            "execution_accuracy": round(
                execution_accuracy,
                2
            ),

            "average_latency_ms": round(
                average_latency,
                2
            ),

            "p95_latency_ms": round(
                p95_latency,
                2
            ),

            "overall_score": round(
                overall_score,
                2
            ),
        },

        "questions": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        "Detailed results saved to: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        run_evaluation()
    )