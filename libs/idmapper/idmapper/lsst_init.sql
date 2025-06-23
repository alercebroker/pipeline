CREATE TABLE IF NOT EXISTS lsst_idmapper (
    lsst_id_serial SERIAL PRIMARY KEY,
    lsst_diaObjectId int8 UNIQUE NOT NULL
);
CREATE INDEX IF NOT EXISTS lsst_diaObjectId_idx
ON lsst_idmapper
USING HASH (lsst_diaObjectId);