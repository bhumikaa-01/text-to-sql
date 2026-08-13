# INTERN NOTE: Factory pattern for database engine creation
# This lets us swap between SQLite (dev) and PostgreSQL (prod)
# by only changing the DATABASE_URL env var. The application
# code never needs to know which database it's talking to.
# The engine and sessionmaker are cached at module level so that
# connection-pool resources (file handles, sockets) are shared across
# all requests instead of being recreated on every call.

from __future__ import annotations

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import os

# Module-level singletons — initialized once on first use.
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None  # type: ignore[type-arg]


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL", "sqlite:///./data/olist.db")

        print("=" * 50)
        print("DATABASE_URL =", url)
        print("CURRENT WORKING DIR =", os.getcwd())
        print("=" * 50)

        if url.startswith("sqlite"):
            _engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=NullPool,
            )
        else:
            _engine = create_engine(url)

    return _engine


def get_session() -> Session:
    """Return a new Session bound to the shared engine.

    The returned Session supports the context-manager protocol so callers
    can write ``with get_session() as session:`` to guarantee cleanup::

        with get_session() as session:
            session.add(obj)
            session.commit()
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal()
