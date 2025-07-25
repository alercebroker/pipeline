from random import Random

import pandas as pd

from generator.lsst_alert import LsstAlertGenerator

schema_path = "../schemas/surveys/lsst"
data_path = "data"


def main():
    rng = Random(42)
    generator = LsstAlertGenerator(rng=rng, new_obj_rate=0.1)

    alerts = [generator.generate_alert() for _ in range(10_000)]
    pd.DataFrame(generator.get_objstats()).to_parquet(
        "data/object_stats.parquet", index=False
    )

    print(f"Written ${len(alerts)=}")


if __name__ == "__main__":
    main()
