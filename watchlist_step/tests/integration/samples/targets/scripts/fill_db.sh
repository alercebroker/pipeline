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
        radius DOUBLE PRECISION,
        filter JSONB
    );
    CREATE TABLE watchlist_match (
        id SERIAL,
        target_id INTEGER REFERENCES watchlist_target(id),
        object_id VARCHAR,
        candid VARCHAR,
        date TIMESTAMP,
        ready_to_notify BOOLEAN,
        values JSONB
    );
    CREATE INDEX ON watchlist_target (q3c_ang2ipix(ra, dec));
    CLUSTER watchlist_target_q3c_ang2ipix_idx ON watchlist_target;
    COPY watchlist_target (id, ra, dec, radius, filter) FROM '/data/targets.csv' DELIMITER ';' CSV;
EOSQL
