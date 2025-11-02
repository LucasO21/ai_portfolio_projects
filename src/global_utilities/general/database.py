
import os
import sqlalchemy as sql
import sqlite3

from src.global_utilities.general.paths import SALES_PIPELINE_ASSISTANT_DIR, SALES_PIPELINE_ASSISTANT_DB_DIR

def get_sqlite_connection(
    path: str = SALES_PIPELINE_ASSISTANT_DB_DIR,
):

    sql_engine = sql.create_engine(path)

    conn = sql_engine.connect()

    return {
        "engine": sql_engine,
        "connection": conn,
        "path": path,
    }


