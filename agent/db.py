"""Database setup — loads the Chinook SQLite database into memory."""

import sqlite3

import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


def get_engine_for_chinook_db():
    """Pull the Chinook SQL file, populate an in-memory SQLite DB, and return an engine."""
    url = (
        "https://raw.githubusercontent.com/lerocha/chinook-database/"
        "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    )
    response = requests.get(url)
    response.raise_for_status()
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


# Module-level singletons so the DB is loaded once and shared everywhere.
engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)
