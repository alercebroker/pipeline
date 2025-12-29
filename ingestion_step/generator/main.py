from random import Random

from fastavro.schema import load_schema
from fastavro.validation import validate_many

from generator.lsst_alert import LsstAlertGenerator

data_path = "data"


def main():
    rng = Random(42)
    generator = LsstAlertGenerator(rng=rng, new_obj_rate=0.1)

    output_schema = load_schema("../schemas/surveys/lsst_v9.0/lsst.v9_0.alert.avsc")

    alerts = [generator.generate_alert() for _ in range(10_000)]
    validate_many(alerts, output_schema, raise_errors=True, strict=True)

    # pd.DataFrame(generator.get_all_objstats_dicts()).to_parquet(
    #     "data/object_stats.parquet", index=False
    # )

    print(f"Written ${len(alerts)=}")


if __name__ == "__main__":
    main()
