import polars as pl

from xmatch_client import XmatchClient

if __name__ == "__main__":
    df = pl.read_csv("examples/data/coordinates.csv", infer_schema_length=None)

    oids = df["AllWISE"].cast(str).to_list()
    ras = df["RAJ2000"].cast(float).to_list()
    decs = df["DEJ2000"].cast(float).to_list()

    print("Input data:\n", df.head())

    xmatch_config = {"base_url": "http://localhost:8081", "batch_size": 500}

    xmatch_client = XmatchClient(**xmatch_config)

    results = xmatch_client.conesearch_with_metadata(ras=ras, decs=decs, oids=oids)

    print("\n", "Output data:\n", pl.DataFrame(results).head())
