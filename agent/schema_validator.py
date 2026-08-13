"""
schema_validator.py

Validates generated SQL against
the known semantic schema.
"""

import re

from agent.semantic_layer import SEMANTIC_SCHEMA


VALID_TABLES = {
    table["table_name"]
    for table in SEMANTIC_SCHEMA
}


def extract_tables(sql: str):
    """
    Extract table names from
    FROM and JOIN clauses.
    """

    pattern = r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"

    matches = re.findall(
        pattern,
        sql,
        flags=re.IGNORECASE
    )

    return set(matches)


def validate_sql_schema(sql: str):
    """
    Validate referenced tables.
    """

    tables = extract_tables(sql)

    invalid_tables = (
        tables - VALID_TABLES
    )

    if invalid_tables:
        return (
            False,
            f"Unknown tables: {', '.join(invalid_tables)}"
        )

    return True, None