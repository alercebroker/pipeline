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
        "tid",
        "new",
        "pid",
        "e_ra",
        "e_dec",
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

"""
def parse_ztf_pf(raw_detection: dict, oid: str):
    forced_photometry = {
        "b_oid": oid,
        "b_measurement_id": raw_detection["measurement_id"],
        "mag_corr": raw_detection["mag_corr"],
        "e_mag_corr": raw_detection["e_mag_corr"],
        "e_mag_corr_ext": raw_detection["e_mag_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        
    }
    return forced_photometry

def parse_ztf_det(raw_detection: dict, oid: str) -> dict:
    detection = {
        "b_oid": oid,
        "b_measurement_id": raw_detection["measurement_id"],
        "magpsf_corr": raw_detection["mag_corr"],
        "sigmapsf_corr": raw_detection["e_mag_corr"],
        "sigmapsf_corr_ext": raw_detection["e_mag_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
    }
    return detection

"""

def parse_fp(raw_detection: dict, oid: str) -> dict:
    forced_photometry = {
        "oid": oid,
        "sid": raw_detection["sid"],
        "measurement_id": raw_detection["measurement_id"],
        "mjd": raw_detection["mjd"],
        "ra": raw_detection["ra"],
        "dec": raw_detection["dec"],
        "band": raw_detection["band"],
    }
    return forced_photometry

def parse_ztf_fp(raw_detection: dict, oid: str) -> dict:
    forced_photometry = {
        "oid": oid,
        "sid": raw_detection["sid"],
        "measurement_id": raw_detection["measurement_id"],
        "pid": raw_detection["pid"],
        "mag": raw_detection["mag"],
        "e_mag": raw_detection["e_mag"],
        "mag_corr": raw_detection["mag_corr"],
        "e_mag_corr": raw_detection["e_mag_corr"],
        "e_mag_corr_ext": raw_detection["e_mag_corr_ext"],
        "isdiffpos": raw_detection["isdiffpos"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        "parent_candid": raw_detection["parent_candid"],
        "has_stamp": raw_detection["has_stamp"],
        "field": raw_detection["field"],
        "rcid": raw_detection["rcid"],
        "rfid": raw_detection["rfid"],
        "sciinpseeing": raw_detection["sciinpseeing"],
        "scibckgnd": raw_detection["scibckgnd"],
        "scisigpix": raw_detection["scisigpix"],
        "magzpsci": raw_detection["magzpsci"],
        "magzpsciunc": raw_detection["magzpsciunc"],
        "magzpscirms": raw_detection["magzpscirms"],
        "clrcoeff": raw_detection["clrcoeff"],
        "clrcounc": raw_detection["clrcounc"],
        "exptime": raw_detection["exptime"],
        "adpctdif1": raw_detection["adpctdif1"],
        "adpctdif2": raw_detection["adpctdif2"],
        "diffmaglim": raw_detection["diffmaglim"],
        "programid": raw_detection["programid"],
        "procstatus": raw_detection["procstatus"],
        "distnr": raw_detection["distnr"],
        "ranr": raw_detection["ranr"],
        "decnr": raw_detection["decnr"],
        "magnr": raw_detection["magnr"],
        "sigmagnr": raw_detection["sigmagnr"],
        "chinr": raw_detection["chinr"],
        "sharpnr": raw_detection["sharpnr"],
    }

    return forced_photometry

def parse_det(raw_detection: dict, oid: str) -> dict:
    detection = {
        "oid": oid,
        "sid": raw_detection["sid"],
        "measurement_id": raw_detection["measurement_id"],
        "mjd": raw_detection["mjd"],
        "ra": raw_detection["ra"],
        "dec": raw_detection["dec"],
        "band": raw_detection["band"],
    }

    return detection

def parse_ztf_det(raw_detection: dict, oid: str) -> dict:
    detection = {
        "oid": oid,
        "sid": raw_detection["sid"],
        "measurement_id": raw_detection["measurement_id"],
        "pid": raw_detection["pid"],
        "diffmaglim": raw_detection["diffmaglim"],
        "isdiffpos": raw_detection["isdiffpos"],
        "nid": raw_detection["nid"],
        "magpsf": raw_detection["mag"],
        "sigmapsf": raw_detection["e_mag"],
        "magap": raw_detection["magap"],
        "sigmagap": raw_detection["sigmagap"],
        "distnr": raw_detection["distnr"],
        "rb": raw_detection["rb"],
        "rbversion": raw_detection["rbversion"],
        "drb": raw_detection.get("drb", None),
        "drbversion": raw_detection.get("drbversion", None),
        "magapbig": raw_detection["magapbig"],
        "sigmagapbig": raw_detection["sigmagapbig"],
        "rfid": raw_detection.get("rfid", None),
        "magpsf_corr": raw_detection["magpsf_corr"],
        "sigmapsf_corr": raw_detection["sigmapsf_corr"],
        "sigmapsf_corr_ext": raw_detection["sigmapsf_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        "parent_candid": raw_detection["parent_candid"],
        "has_stamp": raw_detection["has_stamp"],
    }
    return detection

    
def parse_ztf_ss(candidate: dict, oid: str) -> dict:
    ss = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "ssdistnr": candidate["ssdistnr"],
        "ssmagnr": candidate["ssmagnr"],
        "ssnamenr":None if candidate["ssnamenr"] == 'null' else  candidate["ssnamenr"],
    }
    return ss

def parse_ztf_ps1(candidate: dict, oid: str) -> dict:
    ps1 = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "objectidps1": (
            int(candidate["objectidps1"])
            if candidate["objectidps1"] != "-999" or candidate["objectidps1"] is not None
            else None
        ),
        "sgmag1": candidate["sgmag1"],
        "srmag1": candidate["srmag1"],
        "simag1": candidate["simag1"],
        "szmag1": candidate["szmag1"],
        "sgscore1": candidate["sgscore1"],
        "distpsnr1": candidate["distpsnr1"],
        "objectidps2":(
            int(candidate["objectidps2"])
            if candidate["objectidps2"] != "-999" or candidate["objectidps2"] is not None
            else None
        ),
        "sgmag2": candidate["sgmag2"],
        "srmag2": candidate["srmag2"],
        "simag2": candidate["simag2"],
        "szmag2": candidate["szmag2"],
        "sgscore2": candidate["sgscore2"],
        "distpsnr2": candidate["distpsnr2"],
        "objectidps3":(
            int(candidate["objectidps3"])
            if candidate["objectidps3"] != "-999" or candidate["objectidps3"] is not None
            else None
        ),
        "sgmag3": candidate["sgmag3"],
        "srmag3": candidate["srmag3"],
        "simag3": candidate["simag3"],
        "szmag3": candidate["szmag3"],
        "sgscore3": candidate["sgscore3"],
        "distpsnr3": candidate["distpsnr3"],
        "nmtchps": candidate["nmtchps"],
    }
    return ps1

def parse_ztf_gaia(candidate: dict, oid: str) -> dict:
    gaia = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "neargaia": candidate["neargaia"],
        "neargaiabright": candidate["neargaiabright"],
        "maggaia": candidate["maggaia"],
        "maggaiabright": candidate["maggaiabright"],
    }
    return gaia

def parse_ztf_dq(candidate: dict, oid: str) -> dict:
    dq = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "xpos": candidate["xpos"],
        "ypos": candidate["ypos"],
        "chipsf": candidate["chipsf"],
        "sky": candidate["sky"],
        "fwhm": candidate["fwhm"],
        "classtar": candidate["classtar"],
        "mindtoedge": candidate["mindtoedge"],
        "seeratio": candidate["seeratio"],
        "aimage": candidate["aimage"],
        "bimage": candidate["bimage"],
        "aimagerat": candidate["aimagerat"],
        "bimagerat": candidate["bimagerat"],
        "nneg": candidate["nneg"],
        "nbad": candidate["nbad"],
        "sumrat": candidate["sumrat"],
        "scorr": candidate["scorr"],
        "dsnrms": candidate["dsnrms"],
        "ssnrms": candidate["ssnrms"],
        "magzpsci": candidate["magzpsci"],
        "magzpsciunc": candidate["magzpsciunc"],
        "magzpscirms": candidate["magzpscirms"],
        "nmatches": candidate["nmatches"],
        "clrcoeff": candidate["clrcoeff"],
        "clrcounc": candidate["clrcounc"],
        "zpclrcov": candidate["zpclrcov"],
        "zpmed": candidate["zpmed"],
        "clrmed": candidate["clrmed"],
        "clrrms": candidate["clrrms"],
        "exptime": candidate["exptime"],
    }
    return dq

def parse_ztf_refernece(candidate: dict, oid: str) -> dict:
    reference = {
        "oid": oid,
        "rfid": candidate["rfid"],
        "measurement_id": candidate["measurement_id"],
        "band": candidate["band"],
        "rcid": candidate["rcid"],
        "field": candidate["field"],
        "magnr": candidate["magnr"],
        "sigmagnr": candidate["sigmagnr"],
        "chinr": candidate["chinr"],
        "sharpnr": candidate["sharpnr"],
        "ranr": candidate["ranr"],
        "decnr": candidate["decnr"],
        "mjdstartref": candidate["jdstartref"] - 2400000.5,
        "mjdendref": candidate["jdendref"] - 2400000.5,
        "nframesref": candidate["nframesref"],
    }
    return reference

def parse_obj_stats(raw_magstats, oid: str) -> dict:
    obj = {
        "_oid": oid,
        "oid": oid,
        "meanra": raw_magstats["meanra"],
        "meandec": raw_magstats["meandec"],
        "sigmara": raw_magstats["sigmara"],
        "sigmadec": raw_magstats["sigmadec"],
        "firstmjd": raw_magstats["firstmjd"],
        "lastmjd": raw_magstats["lastmjd"],
        "deltamjd": raw_magstats["deltajd"],
        "n_det": raw_magstats["n_det"],
        "n_forced": raw_magstats["n_fphot"] if raw_magstats["n_fphot"] else 0,
        "n_non_det": raw_magstats["n_ndet"] if raw_magstats["n_ndet"] else 0,
        "corrected": raw_magstats["corrected"],
        "stellar": raw_magstats["stellar"],
    }
    return obj

def parse_magstats(sub_magstats: dict, oid: str, sid: int) -> dict:
    magstats = {
        "oid": oid,
        "sid": sid,
        "band": sub_magstats["band"], 
        "stellar": sub_magstats["stellar"], 
        "corrected": sub_magstats["corrected"], 
        "ndubious": sub_magstats["ndubious"], 
        "dmdt_first": sub_magstats["dmdt_first"], 
        "dm_first": sub_magstats["dm_first"], 
        "sigmadm_first": sub_magstats["sigmadm_first"], 
        "dt_first": sub_magstats["dt_first"], 
        "magmean": sub_magstats["magmean"], 
        "magmedian": sub_magstats["magmedian"], 
        "magmax": sub_magstats["magmax"], 
        "magmin": sub_magstats["magmin"], 
        "magsigma": sub_magstats["magsigma"], 
        "maglast": sub_magstats["maglast"], 
        "magfirst": sub_magstats["magfirst"], 
        "magmean_corr": sub_magstats["magmean_corr"], 
        "magmedian_corr": sub_magstats["magmedian_corr"], 
        "magmax_corr": sub_magstats["magmax_corr"], 
        "magmin_corr": sub_magstats["magmin_corr"], 
        "magsigma_corr": sub_magstats["magsigma_corr"], 
        "maglast_corr": sub_magstats["maglast_corr"], 
        "magfirst_corr": sub_magstats["magfirst_corr"], 
        "saturation_rate": sub_magstats["saturation_rate"], 
    }
    return magstats