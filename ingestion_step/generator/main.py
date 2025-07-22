from random import Random

from fastavro.schema import load_schema
from fastavro.validation import validate
from fastavro.write import writer

from generator.lsst_alert import LsstAlertGenerator

schema_path = "../schemas/surveys/lsst"
data_path = "data"


def main():
    rng = Random(42)
    generator = LsstAlertGenerator(rng=rng, new_obj_rate=0.1)

    schema = load_schema(schema_path + "/v7_4_alert.avsc")

    alerts = [generator.generate_alert() for _ in range(100)]
    for alert in alerts:
        validate(alert, schema)

    from pprint import pprint

    pprint(alerts)

    print("---- objstats ----")
    pprint(generator.get_objstats())

    # with open(data_path + "/alerts.avro", "wb") as f:
    #     writer(f, schema, alerts)


if __name__ == "__main__":
    main()
