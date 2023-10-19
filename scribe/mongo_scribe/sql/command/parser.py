def multistream_detection_to_ztf(command: dict):
    if "oid" not in command:
        raise ValueError("OID was not found")

    if command["sid"] != "ZTF":
        raise ValueError("Detection not from ZTF survey")

    mapping = {
        "mag": "magpsf",
        "e_mag": "sigmapsf",
        "mag_corr": "magpsf_corr",
        "e_mag_corr": "sigmapsf_corr",
        "e_mag_corr_ext": "sigmapsf_corr_ext",
    }

    fid_map = {"g": 1, "r": 2, "i": 3}

    exclude = [
        "aid",
        "sid",
        "tid",
        "new",
        "pid",
        "e_ra",
        "e_dec",
        "extra_fields",
    ]

    new_command = {k: v for k, v in command.items() if k not in exclude}
    for k, v in mapping.items():
        if k in new_command:
            new_command[v] = new_command.pop(k)

    new_command["fid"] = fid_map[new_command["fid"]]
    new_command["candid"] = int(new_command["candid"])
    new_command["parent_candid"] = (
        int(new_command["parent_candid"])
        if new_command["parent_candid"]
        else None
    )

    return new_command
