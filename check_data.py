import sqlite3

conn = sqlite3.connect("data/olist.db")
cur = conn.cursor()

tables = [
    "dim_users",
    "fact_orders",
    "dim_products",
    "dim_sellers",
    "dim_reviews"
]

for table in tables:
    cur.execute(
        f"SELECT COUNT(*) FROM {table}"
    )
    print(
        table,
        cur.fetchone()[0]
    )