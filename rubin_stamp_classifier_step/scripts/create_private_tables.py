"""Create the private classifier / taxonomy / probability tables from the
db_plugins models: those three and nothing else (no create_all over the schema).

    python scripts/create_private_tables.py --db-credentials creds.json --schema multisurvey_ztf \
        [--recreate probability_private ...] [--reader readonly_user] [--writer rubin_stamp_ms]

The json holds user, password, host, port and dbname; the user needs DDL
rights on the schema. Tables that already exist are left alone unless named in
--recreate, which drops them first (rows included) so a table made by hand is
rebuilt from the model. --reader gets SELECT on the three tables; --writer
gets SELECT on classifier/taxonomy and SELECT, INSERT, UPDATE on probability.
Every statement is printed as it runs.
"""
import argparse
import json

from db_plugins.db.sql.models_pipeline import (
    Base,
    ClassifierPrivate,
    ProbabilityPrivate,
    TaxonomyPrivate,
)
from sqlalchemy import create_engine, text

PRIVATE_MODELS = (ClassifierPrivate, TaxonomyPrivate, ProbabilityPrivate)
PRIVATE_TABLES = tuple(model.__tablename__ for model in PRIVATE_MODELS)


def statements(schema: str, recreate: list[str], reader: str | None, writer: str | None) -> list[str]:
    """The raw SQL around the create: drops for --recreate, grants for --reader/--writer."""
    sql = []
    for name in recreate:
        if name not in PRIVATE_TABLES:
            raise ValueError(f"{name} is not a private table; only {PRIVATE_TABLES} can be recreated")
        sql.append(f"DROP TABLE IF EXISTS {schema}.{name} CASCADE")
    if reader:
        sql += [f"GRANT SELECT ON {schema}.{table} TO {reader}" for table in PRIVATE_TABLES]
    if writer:
        for table in PRIVATE_TABLES:
            privileges = "SELECT, INSERT, UPDATE" if table == ProbabilityPrivate.__tablename__ else "SELECT"
            sql.append(f"GRANT {privileges} ON {schema}.{table} TO {writer}")
    return sql


def create_tables(engine, schema: str) -> None:
    """Create the three private tables (existing ones are skipped) and the
    partitions of the partitioned ones, in `schema` (the engine's search_path)."""
    Base.metadata.create_all(engine, tables=[model.__table__ for model in PRIVATE_MODELS])
    with engine.connect() as conn:
        for model in PRIVATE_MODELS:
            if model.__n_partitions__:
                model.__create_partitions__(conn, schema)


def run(sql: list[str], engine) -> None:
    with engine.begin() as conn:
        for statement in sql:
            print(statement)
            conn.execute(text(statement))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-credentials", required=True, help="json with user, password, host, port, dbname")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--recreate", nargs="*", default=[], metavar="TABLE", help="private tables to drop first")
    parser.add_argument("--reader", help="role granted SELECT on the three tables")
    parser.add_argument("--writer", help="role granted SELECT, INSERT, UPDATE on probability_private")
    args = parser.parse_args()

    with open(args.db_credentials) as f:
        c = json.load(f)
    url = f"postgresql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}"
    engine = create_engine(url, connect_args={"options": f"-csearch_path={args.schema}"})

    run(statements(args.schema, args.recreate, reader=None, writer=None), engine)
    print(f"create_all: {', '.join(PRIVATE_TABLES)} (+ partitions) in {args.schema}")
    create_tables(engine, args.schema)
    run(statements(args.schema, [], reader=args.reader, writer=args.writer), engine)


if __name__ == "__main__":
    main()
