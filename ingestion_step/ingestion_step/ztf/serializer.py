import pandas as pd


def serialize_detections(detections: pd.DataFrame):
    # Bad fix for precision loss in kafka
    # for bad_column in ["objectidps1", "objectidps2", "objectidps3", "tblid"]:
    #     detections[bad_column] = detections[bad_column].astype(pd.StringDtype())
    needed_columns = [
             "message_id", "oid", "sid", "tid", "pid", "band", "measurement_id", "mjd", "ra", "e_ra", "dec",
            "e_dec", "mag", "e_mag", "isdiffpos", "has_stamp", "forced", "parent_candid",
            "diffmaglim", "pdiffimfilename", "programpi", "programid", "tblid", "nid", "rcid",
            "field", "xpos", "ypos", "chipsf", "magap", "sigmagap", "distnr", "magnr", "sigmagnr",
            "chinr", "sharpnr", "sky", "magdiff", "fwhm", "classtar", "mindtoedge", "magfromlim",
            "seeratio", "aimage", "bimage", "aimagerat", "bimagerat", "elong", "nneg", "nbad",
            "rb", "ssdistnr", "ssmagnr", "ssnamenr", "sumrat", "magapbig", "sigmagapbig", "ranr",
            "decnr", "sgmag1", "srmag1", "simag1", "szmag1", "sgscore1", "distpsnr1", "ndethist",
            "ncovhist", "jdstarthist", "jdendhist", "scorr", "tooflag", "objectidps1", "objectidps2",
            "sgmag2", "srmag2", "simag2", "szmag2", "sgscore2", "distpsnr2", "objectidps3", "sgmag3",
            "srmag3", "simag3", "szmag3", "sgscore3", "distpsnr3", "nmtchps", "rfid", "jdstartref",
            "jdendref", "nframesref", "rbversion", "dsnrms", "ssnrms", "dsdiff", "magzpsci",
            "magzpsciunc", "magzpscirms", "nmatches", "clrcoeff", "clrcounc", "zpclrcov", "zpmed",
            "clrmed", "clrrms", "neargaia", "neargaiabright", "maggaia", "maggaiabright", "exptime",
            "drb", "drbversion"
        ]
 
    dets = detections[needed_columns]

    return dets

def serialize_prv_candidates(prv_candidates: pd.DataFrame):

    needed_columns = [
             "message_id", "oid", "sid", "tid", "pid", "band", "measurement_id", "mjd", "ra", "e_ra", "dec",
            "e_dec", "mag", "e_mag", "isdiffpos", "has_stamp", "forced", "parent_candid",
            "diffmaglim", "pdiffimfilename", "programpi", "programid", "tblid", "nid", "rcid",
            "field", "xpos", "ypos", "chipsf", "magap", "sigmagap", "distnr", "magnr",
            "sigmagnr", "chinr", "sharpnr", "sky", "magdiff", "fwhm", "classtar", "mindtoedge",
            "magfromlim", "seeratio", "aimage", "bimage", "aimagerat", "bimagerat", "elong",
            "nneg", "nbad", "rb", "ssdistnr", "ssmagnr", "ssnamenr", "sumrat", "magapbig",
            "sigmagapbig", "ranr", "decnr", "scorr", "magzpsci", "magzpsciunc", "magzpscirms",
            "clrcoeff", "clrcounc", "rbversion"
        ]
    
    prv_cands = prv_candidates[needed_columns]
    
    return prv_cands

def serialize_forced_photometries(forced_phots: pd.DataFrame):

    needed_columns = [
             "message_id", "oid", "sid", "tid", "pid", "band", "measurement_id", "mjd", "ra", "e_ra", "dec",
            "e_dec", "mag", "e_mag", "isdiffpos", "has_stamp", "forced", "parent_candid",
            "field", "rcid", "rfid", "sciinpseeing", "scibckgnd", "scisigpix", "magzpsci",
            "magzpsciunc", "magzpscirms", "clrcoeff", "clrcounc", "exptime", "adpctdif1",
            "adpctdif2", "diffmaglim", "programid", "forcediffimflux", "forcediffimfluxunc",
            "procstatus", "distnr", "ranr", "decnr", "magnr", "sigmagnr", "chinr", "sharpnr"
        ]
    
    fphots = forced_phots[needed_columns]

    return fphots


def serialize_non_detections(non_detections: pd.DataFrame):
    non_dets = non_detections
    needed_columns = [
        "message_id",
        "oid",
        "sid",
        "tid",
        "band",
        "mjd",
        "diffmaglim",
    ]

    non_dets = non_dets[needed_columns]

    return non_dets
