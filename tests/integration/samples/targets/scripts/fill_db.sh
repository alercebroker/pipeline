#!/bin/bash
set -e

#https://gis.stackexchange.com/a/13974

# Creating tables
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION q3c;
    CREATE TABLE watchlist_target (
        id INTEGER PRIMARY KEY,
        ra DOUBLE PRECISION,
        dec DOUBLE PRECISION,
        radius DOUBLE PRECISION
    );
    CREATE TABLE watchlist_match (
        id SERIAL,
        target_id INTEGER REFERENCES watchlist_target(id),
        object_id VARCHAR(20),
        candid VARCHAR(200),
        date TIMESTAMP
    );
    COPY watchlist_target (id, ra, dec, radius) FROM '/data/targets.csv' DELIMITER ',' CSV;
EOSQL
