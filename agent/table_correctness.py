"""
table_correctness.py

Deterministic verification of tables used by generated SQL.

The purpose of this module is to provide an observable table-correctness
signal to the confidence engine without making another LLM call.

The verifier compares:
    1. Tables referenced by the generated SQL
    2. Columns referenced by the generated SQL
    3. The semantic schema

It does NOT:
    - execute SQL
    - call an LLM
    - determine whether the user's natural-language question is correct
    - replace SQL safety validation
    - replace schema validation

It answers a narrower question:

    "Are the tables referenced by this SQL consistent with the
     columns and relationships defined in our semantic schema?"
"""

import re
from typing import Any

import sqlglot
from sqlglot import exp

from agent.semantic_layer import SEMANTIC_SCHEMA


# ============================================================
# SCHEMA INDEXES
# ============================================================

def _build_schema_indexes() -> tuple[
    set[str],
    dict[str, set[str]],
]:
    """
    Build fast lookup structures from the semantic schema.

    Returns
    -------
    tuple
        (
            valid_tables,
            table_columns
        )
    """

    valid_tables: set[str] = set()

    table_columns: dict[str, set[str]] = {}

    for table in SEMANTIC_SCHEMA:

        table_name = table["table_name"].lower()

        valid_tables.add(table_name)

        table_columns[table_name] = {
            column["name"].lower()
            for column in table.get("columns", [])
        }

    return (
        valid_tables,
        table_columns,
    )


VALID_TABLES, TABLE_COLUMNS = _build_schema_indexes()


# ============================================================
# SQL TABLE EXTRACTION
# ============================================================

def extract_sql_tables(
    sql: str,
) -> list[str]:
    """
    Extract table names referenced by FROM and JOIN clauses.

    This intentionally performs lightweight deterministic parsing.
    Full SQL parsing remains the responsibility of the existing
    schema validator.

    Examples
    --------
    FROM fact_orders fo
        -> fact_orders

    JOIN dim_products dp
        -> dim_products
    """

    if not sql or not sql.strip():

        return []

    pattern = re.compile(
        r"""
        \b
        (?:FROM|JOIN)
        \s+
        (?:
            [A-Za-z_][A-Za-z0-9_]*\.
        )?
        ([A-Za-z_][A-Za-z0-9_]*)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    matches = pattern.findall(sql)

    tables: list[str] = []

    for table in matches:

        normalized = table.lower()

        if normalized not in tables:

            tables.append(
                normalized
            )

    return tables

# ============================================================
# CTE EXTRACTION
# ============================================================

def extract_cte_info(
    sql: str,
) -> tuple[set[str], dict[str, set[str]]]:
    """
    Extract CTE names and their output columns.

    Example:

        WITH YearlyRevenue AS (
            SELECT
                ... AS revenue_2017,
                ... AS revenue_2018
            FROM fact_orders
        )

    Returns:

        (
            {"yearlyrevenue"},
            {
                "yearlyrevenue": {
                    "revenue_2017",
                    "revenue_2018",
                }
            }
        )

    CTEs are query-level objects, not physical database tables.
    """

    if not sql or not sql.strip():
        return set(), {}

    try:
        tree = sqlglot.parse_one(sql)

    except Exception:
        return set(), {}

    cte_names: set[str] = set()
    cte_columns: dict[str, set[str]] = {}

    for cte in tree.find_all(exp.CTE):

        cte_name = cte.alias_or_name

        if not cte_name:
            continue

        cte_name = cte_name.lower()

        cte_names.add(cte_name)

        output_columns: set[str] = set()

        cte_body = cte.this

        for select in cte_body.find_all(exp.Select):

            for projection in select.expressions:

                alias = projection.alias

                if alias:
                    output_columns.add(
                        alias.lower()
                    )
                    continue

                if isinstance(
                    projection,
                    exp.Column,
                ):
                    output_columns.add(
                        projection.name.lower()
                    )

        cte_columns[cte_name] = output_columns

    return (
        cte_names,
        cte_columns,
    )


# ============================================================
# SQL ALIAS EXTRACTION
# ============================================================

def extract_table_aliases(
    sql: str,
) -> dict[str, str]:
    """
    Extract aliases assigned to FROM/JOIN tables.

    Returns:

        {
            "fo": "fact_orders",
            "dp": "dim_products"
        }

    The table name itself is also treated as a valid reference.
    """

    if not sql or not sql.strip():

        return {}

    pattern = re.compile(
        r"""
        \b
        (?:FROM|JOIN)
        \s+
        (?:
            [A-Za-z_][A-Za-z0-9_]*\.
        )?
        (?P<table>[A-Za-z_][A-Za-z0-9_]*)
        (?:
            \s+
            (?:AS\s+)?
            (?P<alias>[A-Za-z_][A-Za-z0-9_]*)
        )?
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    aliases: dict[str, str] = {}

    for match in pattern.finditer(sql):

        table = match.group(
            "table"
        ).lower()

        alias = match.group(
            "alias"
        )

        aliases[table] = table

        if alias:

            aliases[alias.lower()] = table

    return aliases


# ============================================================
# SQL COLUMN REFERENCES
# ============================================================

def extract_qualified_columns(
    sql: str,
) -> list[tuple[str, str]]:
    """
    Extract qualified column references such as:

        fo.order_total_usd
        dp.category_name

    Returns:

        [
            ("fo", "order_total_usd"),
            ("dp", "category_name")
        ]
    """

    if not sql or not sql.strip():

        return []

    pattern = re.compile(
        r"\b"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"\."
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"\b",
        re.IGNORECASE,
    )

    references: list[tuple[str, str]] = []

    for match in pattern.findall(sql):

        alias = match[0].lower()

        column = match[1].lower()

        item = (
            alias,
            column,
        )

        if item not in references:

            references.append(item)

    return references


# ============================================================
# TABLE CORRECTNESS
# ============================================================

def check_table_correctness(
    *,
    sql: str,
) -> dict[str, Any]:
    """
    Deterministically verify tables referenced by generated SQL.

    Parameters
    ----------
    sql:
        Generated SQL query.

    Returns
    -------
    dict

        {
            "table_correct": bool,
            "tables_used": [...],
            "invalid_tables": [...],
            "invalid_columns": [...],
            "issues": [...]
        }

    Notes
    -----
    This verifies structural consistency against the semantic schema.

    It does NOT prove that the SQL semantically answers the user's
    question. That responsibility remains with semantic_evaluator.py.
    """

    if not sql or not sql.strip():

        return {
            "table_correct": False,
            "tables_used": [],
            "invalid_tables": [],
            "invalid_columns": [],
            "issues": [
                "EMPTY_SQL",
            ],
        }

    # --------------------------------------------------------
    # Extract CTE information
    # --------------------------------------------------------

    cte_names, cte_columns = extract_cte_info(
        sql
    )

    # --------------------------------------------------------
    # Extract physical tables
    # --------------------------------------------------------

    tables_used = [
        table
        for table in extract_sql_tables(sql)
        if table not in cte_names
    ]

    invalid_tables = [
        table
        for table in tables_used
        if table not in VALID_TABLES
    ]

    # --------------------------------------------------------
    # Extract aliases
    # --------------------------------------------------------

    aliases = extract_table_aliases(
        sql
    )

    # --------------------------------------------------------
    # Validate qualified columns
    # --------------------------------------------------------

    qualified_columns = extract_qualified_columns(
        sql
    )

    invalid_columns: list[str] = []

    for alias, column in qualified_columns:

        table = aliases.get(
            alias
        )

        if table is None:

            invalid_columns.append(
                f"{alias}.{column}"
            )

            continue

        # ----------------------------------------------------
        # CTE reference
        # ----------------------------------------------------

        if table in cte_names:

            valid_columns = cte_columns.get(
                table,
                set(),
            )

            if column not in valid_columns:

                invalid_columns.append(
                    f"{table}.{column}"
                )

            continue

        # ----------------------------------------------------
        # Physical table reference
        # ----------------------------------------------------

        valid_columns = TABLE_COLUMNS.get(
            table,
            set(),
        )

        if column not in valid_columns:

            invalid_columns.append(
                f"{table}.{column}"
            )
    # --------------------------------------------------------
    # Build issues
    # --------------------------------------------------------

    issues: list[str] = []

    if invalid_tables:

        issues.append(
            "INVALID_TABLE_REFERENCE"
        )

    if invalid_columns:

        issues.append(
            "INVALID_COLUMN_REFERENCE"
        )

    # --------------------------------------------------------
    # Final correctness
    # --------------------------------------------------------

    table_correct = (
        len(tables_used) > 0
        and not invalid_tables
        and not invalid_columns
    )

    return {
        "table_correct": table_correct,
        "tables_used": tables_used,
        "invalid_tables": invalid_tables,
        "invalid_columns": invalid_columns,
        "issues": issues,
    }