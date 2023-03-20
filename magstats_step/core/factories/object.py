def alerce_object_factory(raw_object_info: dict):
    if "aid" not in raw_object_info:
        raise ValueError("AID not provided")

    return {
        "aid": raw_object_info["aid"],
        "meanra": -999,
        "meandec": -999,
        "magstats": [],
        "oid": [],
        "tid": [],
        "firstmjd": -999,
        "lastmjd": -999,
        "ndet": -999,
        "sigmara": -999,
        "sigmadec": -999,
    }
