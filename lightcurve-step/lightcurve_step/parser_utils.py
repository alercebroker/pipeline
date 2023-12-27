def get_fid(fid_as_int: int):
    fid = {1: "g", 2: "r", 0: None, 12: "gr", 3: "i"}
    try:
        return fid[fid_as_int]
    except KeyError:
        return fid_as_int
