import os
from typing import Any

from psycopg2.sql import SQL

from watchlist_step.db.connection import DatabaseConfig, PsqlDatabase

config: DatabaseConfig = {
    "HOST": os.environ["ALERTS_DB_HOST"],
    "USER": os.environ["ALERTS_DB_USER"],
    "PASSWORD": os.environ["ALERTS_DB_PASSWORD"],
    "PORT": os.getenv("ALERTS_DB_PORT", "5432"),
    "DB_NAME": os.environ["ALERTS_DB_NAME"],
}


def run(p: float = 0.0025, seed: int = 42, radius: float = 1):
    alerts_db = PsqlDatabase(config)
    rows: list[tuple[Any, ...]]
    with alerts_db.conn() as conn:
        with conn.cursor() as cursor:
            query = SQL("""
            SELECT
                p.oid,
                meanra,
                meandec
            FROM
                probability p
                TABLESAMPLE SYSTEM (%s) REPEATABLE (%s)
            JOIN
                object o
            ON
                o.oid = p.oid
            WHERE
                classifier_name = 'lc_classifier'
                AND class_name = 'AGN'
                AND ranking = 1
            """)
            cursor.execute(query, (p * 100.0, seed))
            rows = cursor.fetchall()
    with open("./sample.csv", "w+") as file:
        file.write("name,ra,dec,radius\n")
        file.writelines(
            map(
                lambda row: f"{row[0]},{row[1]:.5f},{row[2]:.4f},{radius}\n",
                rows,
            )
        )
