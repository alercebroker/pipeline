import os

import polars as pl
import polars.testing as pl_test

url = f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"

EPS = 1e-4


def main():
    expected = pl.read_parquet("data/object_stats.parquet")
    db = (
        pl.read_database_uri("SELECT * FROM pruebas_multisurvey_mapper.object;", url)
        .select(expected.columns)
        .sort("oid")
    )

    pl.Config.set_float_precision(10)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)

    for x in ["ra", "dec"]:
        print(f"Mean {x}:")
        print(
            expected.join(db, on="oid", suffix="_db")
            .filter((pl.col(f"mean{x}") - pl.col(f"mean{x}_db")).abs() > EPS)
            .select("oid", f"mean{x}", f"mean{x}_db", f"sigma{x}", f"sigma{x}_db")
        )
        print("-----")

        print(f"Sigma {x}:")
        print(
            expected.join(db, on="oid", suffix="_db")
            .filter((pl.col(f"sigma{x}") - pl.col(f"sigma{x}_db")).abs() > EPS)
            .select("oid", f"mean{x}", f"mean{x}_db", f"sigma{x}", f"sigma{x}_db")
        )
        print("-----")

    pl.Config.set_float_precision(None)

    # pl_test.assert_frame_equal(expected, db, check_dtypes=False)


if __name__ == "__main__":
    main()
