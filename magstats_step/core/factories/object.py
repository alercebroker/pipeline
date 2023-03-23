def alerce_object_factory(raw_object_info: dict):
    if "aid" not in raw_object_info:
        raise ValueError("AID not provided")

    return {
        "aid": raw_object_info["aid"],
        "meanra": None,
        "meandec": None,
        "magstats": [],
        "oid": [],
        "tid": [],
        "firstmjd": None,
        "lastmjd": None,
        "ndet": None,
        "sigmara": None,
        "sigmadec": None,
    }
