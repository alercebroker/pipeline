from stamp_classifier_step.strategies.atlas import ATLASStrategy
from conftest import get_binary
import cProfile
import time


def alerts():
    alert = {
        "oid": "oid",
        "tid": "tid",
        "pid": 1,
        "candid": "123",
        "mjd": 1,
        "fid": 1,
        "ra": 1.0,
        "dec": 1.0,
        "rb": 1,
        "rbversion": "a",
        "mag": 1,
        "e_mag": 1,
        "rfid": 1,
        "isdiffpos": 1,
        "e_ra": 1,
        "e_dec": 1,
        "extra_fields": {},
        "aid": "aid",
        "stamps": {
            "science": get_binary("science"),
            "template": None,
            "difference": get_binary("difference"),
        },
    }
    return [alert]


if __name__ == "__main__":
    profiler = cProfile.Profile()

    strategy = ATLASStrategy()
    atlas_alerts = alerts()
    n_alerts = 50
    many_alerts = []
    for i in range(n_alerts):
        alert_copy = atlas_alerts[0].copy()
        alert_copy["aid"] = str(i)
        many_alerts.append(alert_copy)
    t0 = time.time()
    output_dict = strategy.get_probabilities(many_alerts)
    alerts_per_second = n_alerts / (time.time() - t0)
    print(alerts_per_second)
    profiler.enable()
    _ = strategy.get_probabilities(many_alerts)
    profiler.disable()
    profiler.dump_stats("profiling_atlas.prof")
