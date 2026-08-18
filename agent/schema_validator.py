"""
schema_validator.py

Production-grade validation of LLM-generated SQL against
the known semantic schema.

Responsibilities:
    - Validate referenced tables.
    - Validate referenced columns.
    - Resolve table aliases.
    - Understand CTEs.
    - Ignore CTE names when validating physical tables.
    - Return both simple and detailed validation results.

The validator is deterministic and does not call an LLM.
"""

import logging
from typing import Any

import sqlglot
from sqlglot import exp

from agent.semantic_layer import SEMANTIC_SCHEMA


logger = logging.getLogger(__name__)


# ============================================================
# SEMANTIC SCHEMA INDEXES
# ============================================================

TABLE_COLUMNS: dict[str, set[str]] = {
    table["table_name"]: {
        column["name"]
        for column in table.get("columns", [])
    }
    for table in SEMANTIC_SCHEMA
}


VALID_TABLES: set[str] = set(
    TABLE_COLUMNS.keys()
)


# ============================================================
# SQL PARSING
# ============================================================

def parse_sql(
    sql: str,
) -> exp.Expression:
    """
    Parse SQL using SQLGlot.

    SQLite is used as the SQL dialect because the application
    currently executes against SQLite.
    """

    if not sql or not sql.strip():

        raise ValueError(
            "SQL query is empty."
        )

    try:

        return sqlglot.parse_one(
            sql,
            read="sqlite",
        )

    except Exception as exc:

        logger.warning(
            "SQL parsing failed: %s",
            exc,
        )

        raise ValueError(
            f"Invalid SQL syntax: {exc}"
        ) from exc


# ============================================================
# CTE EXTRACTION
# ============================================================

def extract_cte_names(
    tree: exp.Expression,
) -> set[str]:
    """
    Extract CTE names from the SQL AST.

    Example:

        WITH revenue AS (...)
        SELECT * FROM revenue

    'revenue' is a CTE, not a physical database table.
    """

    cte_names: set[str] = set()

    for cte in tree.find_all(
        exp.CTE
    ):

        alias = cte.alias_or_name

        if alias:

            cte_names.add(
                alias.lower()
            )

    return cte_names


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_tables(
    sql: str,
) -> set[str]:
    """
    Extract physical table names referenced by SQL.

    CTE references are excluded.
    """

    tree = parse_sql(sql)

    cte_names = extract_cte_names(
        tree
    )

    tables: set[str] = set()

    for table in tree.find_all(
        exp.Table
    ):

        table_name = table.name

        if not table_name:
            continue

        if table_name.lower() in cte_names:
            continue

        tables.add(
            table_name
        )

    return tables


# ============================================================
# TABLE ALIAS EXTRACTION
# ============================================================

def extract_table_aliases(
    tree: exp.Expression,
) -> dict[str, str]:
    """
    Build a mapping:

        alias -> physical table

    Example:

        FROM fact_orders fo

    becomes:

        {
            "fo": "fact_orders"
        }
    """

    aliases: dict[str, str] = {}

    for table in tree.find_all(
        exp.Table
    ):

        table_name = table.name

        if not table_name:
            continue

        alias = table.alias

        if alias:

            aliases[
                alias.lower()
            ] = table_name

        aliases[
            table_name.lower()
        ] = table_name

    return aliases


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(
    tree: exp.Expression,
    referenced_tables: set[str],
) -> tuple[bool, list[str]]:
    """
    Validate columns referenced by the SQL query.

    Handles:
        - qualified physical-table columns
        - unqualified physical-table columns
        - table aliases
        - CTE output columns
        - SELECT aliases
        - wildcard references

    CTE-derived columns are treated as valid when they are
    produced by the CTE's SELECT statement.
    """

    aliases = extract_table_aliases(
        tree
    )

    invalid_columns: list[str] = []

    # --------------------------------------------------------
    # 1. Build physical-table column index
    # --------------------------------------------------------

    available_columns: set[str] = set()

    for table_name in referenced_tables:

        columns = TABLE_COLUMNS.get(
        table_name
    )

        if columns:
            available_columns.update(
            col.lower()
            for col in columns
        )

    # --------------------------------------------------------
    # 2. Extract CTE output columns
    # --------------------------------------------------------

    cte_columns: dict[str, set[str]] = {}

    for cte in tree.find_all(
        exp.CTE
    ):

        cte_name = cte.alias_or_name

        if not cte_name:
            continue

        cte_name = cte_name.lower()

        output_columns: set[str] = set()

        # The CTE body is the SELECT/query inside AS (...)
        cte_body = cte.this

        for select in cte_body.find_all(
            exp.Select
        ):

            for projection in select.expressions:

                # --------------------------------------------
                # Explicit alias
                #
                # SUM(...) AS total_revenue
                # --------------------------------------------

                alias = projection.alias

                if alias:
                    output_columns.add(
                        alias.lower()
                    )
                    continue

                # --------------------------------------------
                # Direct column
                #
                # SELECT order_total_usd
                # --------------------------------------------

                if isinstance(
                    projection,
                    exp.Column
                ):

                    output_columns.add(
                        projection.name.lower()
                    )

        cte_columns[
            cte_name
        ] = output_columns

    # --------------------------------------------------------
    # 3. Extract SELECT aliases
    # --------------------------------------------------------
    #
    # Example:
    #
    # SELECT
    #     strftime('%Y-%m', created_at) AS month
    #
    # Later:
    #
    # GROUP BY month
    # ORDER BY month
    #
    # "month" is a derived SQL alias, not a physical
    # database column.
    #

    select_aliases: set[str] = set()

    for select in tree.find_all(exp.Select):

        for projection in select.expressions:

            alias = projection.alias

            if alias:
                select_aliases.add(
                    alias.lower()
                )

    # --------------------------------------------------------
    # 3. Validate column references
    # --------------------------------------------------------

    for column in tree.find_all(
        exp.Column
    ):

        column_name = column.name

        if not column_name:
            continue

        column_name_lower = (
            column_name.lower()
        )

        # Wildcards are valid.
        if column_name == "*":
            continue

        table_reference = column.table

        # ----------------------------------------------------
        # Qualified column
        #
        # Example:
        #
        # fo.order_total_usd
        # dp.category_name
        # ----------------------------------------------------

        if table_reference:

            table_reference_lower = (
                table_reference.lower()
            )

            # -----------------------------------------------
            # Is this a CTE reference?
            # -----------------------------------------------

            if (
                table_reference_lower
                in cte_columns
            ):

                if (
                    column_name_lower
                    not in cte_columns[
                        table_reference_lower
                    ]
                ):

                    invalid_columns.append(
                        f"{table_reference}.{column_name}"
                    )

                continue

            # -----------------------------------------------
            # Resolve physical-table alias
            # -----------------------------------------------

            resolved_table = aliases.get(
                table_reference_lower
            )

            if resolved_table is None:
                continue

            valid_columns = (
                TABLE_COLUMNS.get(
                    resolved_table
                )
            )

            if valid_columns is None:
                continue

            valid_columns_lower = {
                col.lower()
                for col in valid_columns
            }

            if (
                column_name_lower
                not in valid_columns_lower
            ):

                invalid_columns.append(
                    f"{table_reference}.{column_name}"
                )

            continue

        # ----------------------------------------------------
        # Unqualified column
        #
        # Example:
        #
        # SELECT order_total_usd
        # FROM fact_orders
        #
        # OR:
        #
        # SELECT total_revenue
        # FROM revenue
        # ----------------------------------------------------

        if (
            column_name_lower
            in available_columns
        ):
            continue

        # SELECT aliases are valid references in
        # GROUP BY / ORDER BY / HAVING clauses.
        if column_name_lower in select_aliases:
            continue

        # Check whether the column belongs to
        # a CTE used by the query.

        if any(
            column_name_lower in columns
            for columns in cte_columns.values()
        ):

            continue

        invalid_columns.append(
            column_name
        )

    return (
        len(invalid_columns) == 0,
        sorted(
            set(invalid_columns)
        ),
    )


# ============================================================
# DETAILED VALIDATION
# ============================================================

def validate_sql_schema_detailed(
    sql: str,
) -> dict[str, Any]:
    """
    Perform complete semantic-schema validation.

    Returns:

        {
            "valid": bool,
            "tables": [...],
            "invalid_tables": [...],
            "invalid_columns": [...],
            "ctes": [...],
            "error": str | None
        }
    """

    result: dict[str, Any] = {
        "valid": False,
        "tables": [],
        "invalid_tables": [],
        "invalid_columns": [],
        "ctes": [],
        "error": None,
    }

    try:

        tree = parse_sql(sql)

    except Exception as exc:

        result["error"] = str(
            exc
        )

        return result

    # --------------------------------------------------------
    # CTEs
    # --------------------------------------------------------

    cte_names = extract_cte_names(
        tree
    )

    result["ctes"] = sorted(
        cte_names
    )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    tables = extract_tables(
        sql
    )

    result["tables"] = sorted(
        tables
    )

    invalid_tables = (
        tables - VALID_TABLES
    )

    result["invalid_tables"] = sorted(
        invalid_tables
    )

    if invalid_tables:

        result["error"] = (
            "Unknown tables: "
            + ", ".join(
                sorted(invalid_tables)
            )
        )

        return result

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    columns_valid, invalid_columns = (
        validate_columns(
            tree,
            tables,
        )
    )

    result[
        "invalid_columns"
    ] = invalid_columns

    if not columns_valid:

        result["error"] = (
            "Unknown columns: "
            + ", ".join(
                invalid_columns
            )
        )

        return result

    # --------------------------------------------------------
    # Everything passed
    # --------------------------------------------------------

    result["valid"] = True

    return result


# ============================================================
# BACKWARD-COMPATIBLE PUBLIC API
# ============================================================

def validate_sql_schema(
    sql: str,
) -> tuple[bool, str | None]:
    """
    Backward-compatible schema validation API.

    Returns:

        (True, None)

    or:

        (False, "reason")
    """

    result = (
        validate_sql_schema_detailed(
            sql
        )
    )

    if result["valid"]:

        return (
            True,
            None,
        )

    return (
        False,
        result["error"],
    )