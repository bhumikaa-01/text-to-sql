from model.database import get_engine
from sqlalchemy import text

engine = get_engine()

with engine.connect() as conn:
    result = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'")
    )

    print("TABLES:")
    print(result.fetchall())