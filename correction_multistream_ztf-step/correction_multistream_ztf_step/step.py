import logging
import numpy as np
import pandas as pd
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from typing import List

### Imports for lightcurve database queries
from core.DB.database_sql import (
    PSQLConnection,
    _get_sql_detections,
    _get_sql_forced_photometries,
    _get_sql_non_detections,
)
from core.parsers.parser_sql import (
    parse_sql_detection,
    parse_sql_forced_photometry,
    parse_sql_non_detection,
)

from core.corrector import Corrector

class CorrectionMultistreamZTFStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.CorrectionMultistreamZTFStep")
        self.set_producer_key_field("oid")


    def execute(self, messages: List[dict]) -> dict:
        all_detections = []
        all_non_detections = []
        msg_data = []
    
        for msg in messages: 
            #msg = {'oid': '1234567890', 'measurement_id': '3002139200815010001', 'detections': [{'aid': 'AL25ledvhnivrpyro', 'oid': '1234567890', 'sid': 'ZTF', 'pid': 3002139200815, 'tid': 'ZTF', 'band': 'g', 'measurement_id': '3002139200815010001', 'mjd': 60756.13920139987, 'ra': 102.4124856, 'e_ra': 0.06505842506885529, 'dec': 2.428348, 'e_dec': 0.06499999761581421, 'mag': 18.211593627929688, 'e_mag': 0.13416489958763123, 'isdiffpos': -1, 'has_stamp': True, 'forced': False, 'parent_candid': None, 'extra_fields': {'diffmaglim': 20.361549377441406, 'pdiffimfilename': 'ztf_20250322139201_000461_zg_c03_o_q1_scimrefdiffimg.fits', 'programpi': 'Kulkarni', 'programid': 1, 'tblid': 1, 'nid': 3002, 'rcid': 8, 'field': 461, 'xpos': 3055.9765625, 'ypos': 486.32110595703125, 'chipsf': 21.012855529785156, 'magap': 18.452800750732422, 'sigmagap': 0.07240000367164612, 'distnr': 0.6040137410163879, 'magnr': 15.531000137329102, 'sigmagnr': 0.014000000432133675, 'chinr': 0.7239999771118164, 'sharpnr': -0.013000000268220901, 'sky': -0.8485742807388306, 'magdiff': 0.24120700359344482, 'fwhm': 1.4315736293792725, 'classtar': 0.8849999904632568, 'mindtoedge': 16.523399353027344, 'magfromlim': 1.9087491035461426, 'seeratio': 2.0, 'aimage': 0.7509999871253967, 'bimage': 0.6259999871253967, 'aimagerat': 0.5245975255966187, 'bimagerat': 0.4372810125350952, 'elong': 1.1996805667877197, 'nneg': 4, 'nbad': 0, 'rb': 0.6557142734527588, 'ssdistnr': -999.0, 'ssmagnr': -999.0, 'ssnamenr': 'null', 'sumrat': 0.9966363310813904, 'magapbig': 18.455699920654297, 'sigmagapbig': 0.0868000015616417, 'ranr': 102.41231536865234, 'decnr': 2.428351640701294, 'sgmag1': 15.458499908447266, 'srmag1': 13.499699592590332, 'simag1': 12.519000053405762, 'szmag1': 11.694000244140625, 'sgscore1': 0.9637920260429382, 'distpsnr1': 0.6792148351669312, 'ndethist': 141, 'ncovhist': 852, 'jdstarthist': 2458426.0, 'jdendhist': 2460756.75, 'scorr': 8.626946449279785, 'tooflag': 0, 'objectidps1': 1.1091102783871386e+17, 'objectidps2': 1.1091102783871386e+17, 'sgmag2': -999.0, 'srmag2': 21.4768009185791, 'simag2': -999.0, 'szmag2': -999.0, 'sgscore2': 0.09989289939403534, 'distpsnr2': 9.005562782287598, 'objectidps3': 1.1091102783871386e+17, 'sgmag3': 19.660999298095703, 'srmag3': 18.793399810791016, 'simag3': 18.318500518798828, 'szmag3': 18.097000122070312, 'sgscore3': 0.4681670069694519, 'distpsnr3': 9.469527244567871, 'nmtchps': 17, 'rfid': 461120108, 'jdstartref': 2458369.0, 'jdendref': 2458423.0, 'nframesref': 15, 'rbversion': 't17_f5_c3', 'dsnrms': 8.453817367553711, 'ssnrms': 46.838356018066406, 'dsdiff': -38.38454055786133, 'magzpsci': 26.404592514038086, 'magzpsciunc': 3.219299969714484e-06, 'magzpscirms': 0.031863998621702194, 'nmatches': 2005, 'clrcoeff': -0.1269499957561493, 'clrcounc': 6.498699804069474e-06, 'zpclrcov': -4.410000201460207e-06, 'zpmed': 26.322999954223633, 'clrmed': 0.656000018119812, 'clrrms': 0.18664300441741943, 'neargaia': 0.689031720161438, 'neargaiabright': 0.689031720161438, 'maggaia': 12.743537902832031, 'maggaiabright': 12.743537902832031, 'exptime': 30.0, 'drb': 0.011009779758751392, 'drbversion': 'd6_m7', 'brokerIngestTimestamp': 1742614285, 'surveyPublishTimestamp': 1742614298624.0, 'parent_candid': None}}, {'aid': 'AL25ledvhnivrpyro', 'oid': '1234567890', 'sid': 'ZTF', 'pid': 2974191730815, 'tid': 'ZTF', 'band': 'r', 'measurement_id': 'ZTF18acclklc2974191730815', 'mjd': 60728.191736099776, 'ra': 102.4124856, 'e_ra': 0.0, 'dec': 2.428348, 'e_dec': 0.0, 'mag': 17.095138549804688, 'e_mag': 0.011105540208518505, 'isdiffpos': -1, 'has_stamp': False, 'forced': True, 'parent_candid': '3002139200815010001', 'extra_fields': {'field': 461, 'rcid': 8, 'rfid': 461120208, 'sciinpseeing': 2.894200086593628, 'scibckgnd': 251.1649932861328, 'scisigpix': 9.40414047241211, 'magzpsci': 26.312000274658203, 'magzpsciunc': 3.8145999496919103e-06, 'magzpscirms': 0.023059900850057602, 'clrcoeff': 0.08249370008707047, 'clrcounc': 7.810180250089616e-06, 'exptime': 30.0, 'adpctdif1': 0.06852299720048904, 'adpctdif2': 0.06517399847507477, 'diffmaglim': 19.89859962463379, 'programid': 1, 'forcediffimfluxunc': 49.72494888305664, 'procstatus': '0', 'distnr': 0.6907779574394226, 'ranr': 102.41229248046875, 'decnr': 2.4283649921417236, 'magnr': 13.352999687194824, 'sigmagnr': 0.014999999664723873, 'chinr': 0.574999988079071, 'sharpnr': -0.01600000075995922}}], 'non_detections': [{'aid': 'AL25ledvhnivrpyro', 'oid': '1234567890', 'sid': 'ZTF', 'tid': 'ZTF', 'band': 'r', 'mjd': 60726.28568289988, 'diffmaglim': 20.403099060058594}, {'aid': 'AL25ledvhnivrpyro', 'oid': '1234567890', 'sid': 'ZTF', 'tid': 'ZTF', 'band': 'r', 'mjd': 60728.191736099776, 'diffmaglim': 19.89859962463379}, {'aid': 'AL25ledvhnivrpyro', 'oid': '1234567890', 'sid': 'ZTF', 'tid': 'ZTF', 'band': 'g', 'mjd': 60728.33390050009, 'diffmaglim': 19.596900939941406}], 'timestamp': 1742614288323}
            oid = msg["oid"]

            measurement_id = msg["measurement_id"]
            msg_data.append({"oid": oid, "measurement_id":measurement_id})

            for detection in msg["detections"]:
                parsed_detection = detection.copy()
                parsed_detection["oid"] = oid
                parsed_detection["new"] = True
                all_detections.append(parsed_detection)

            for non_detection in msg["non_detections"]:  
                parsed_non_detection = non_detection.copy()
                parsed_non_detection["oid"] = oid
                all_non_detections.append(parsed_non_detection)
        
        msg_df = pd.DataFrame(msg_data)
        detections_df = pd.DataFrame(all_detections) # We will always have detections BUT not always non_detections
        if all_non_detections:
            non_detections_df = pd.DataFrame(all_non_detections)
        else:
            non_detections_df = pd.DataFrame(columns=["oid", "measurement_id", "band", "mjd", "diffmaglim"]) 

        oids = set(msg_df["oid"].unique()) 

        measurement_ids =  msg_df.groupby("oid")["measurement_id"].apply(lambda x: [str(id) for id in x]).to_dict()
        last_mjds = detections_df.groupby("oid")["mjd"].max().to_dict()

        logger = logging.getLogger("alerce.CorrectionMultistreamZTFStep")
        logger.debug(f"Received {len(detections_df)} detections from messages")
        oids = list(oids)
        detections = detections_df.to_dict('records')
        non_detections = non_detections_df.to_dict('records')


        """Queries the database for all detections and non-detections for each OID and removes duplicates"""
        db_sql_detections = _get_sql_detections(
            oids, self.db_sql, parse_sql_detection
        )
        db_sql_non_detections = _get_sql_non_detections(
            oids, self.db_sql, parse_sql_non_detection
        )
        db_sql_forced_photometries = _get_sql_forced_photometries(
            oids, self.db_sql, parse_sql_forced_photometry
        )
        
        detections = pd.DataFrame(
            detections 
            + db_sql_detections
            + db_sql_forced_photometries
        )

        non_detections = pd.DataFrame(
            non_detections
            + db_sql_non_detections
        )

        self.logger.debug(f"Retrieved {detections.shape[0]} detections")
        detections["measurement_id"] = detections["measurement_id"].astype(str)
        detections["parent_candid"] = detections["parent_candid"].astype(str)

        # TODO: check if this logic is ok
        # TODO: has_stamp in db is not reliable
        # has_stamp true will be on top
        # new true will be on top
        detections = detections.sort_values(
            ["has_stamp", "new"], ascending=[False, False]
        )

        # so this will drop alerts coming from the database if they are also in the stream
        # but will also drop if they are previous detections
        detections = detections.drop_duplicates(
            ["measurement_id", "oid"], keep="first"
        )

        non_detections = non_detections.drop_duplicates(["oid", "band", "mjd"])
        self.logger.debug(
            f"Obtained {len(detections[detections['new']])} new detections"
        )
        
        detections = detections.replace(np.nan, None)
        non_detections = non_detections.replace(np.nan, None) if not non_detections.empty else pd.DataFrame(columns=["oid"])

        if not self.config["FEATURE_FLAGS"].get("SKIP_MJD_FILTER", False):
            detections = detections[detections["mjd"] <= detections["oid"].map(last_mjds)]
        
        corrector = Corrector(detections)
     
        detections = corrector.corrected_as_records()
        coords = corrector.coordinates_as_records()

        non_detections = non_detections.drop_duplicates(
            ["oid", "band", "mjd"]
        )

        return {
            "detections": detections,
            "non_detections": non_detections.to_dict("records"),
            "coords": coords,
            "measurement_ids": measurement_ids,
        }
    
        


    
    @classmethod
    def pre_produce(cls, result: dict):
        result["detections"] = pd.DataFrame(result["detections"]).groupby("oid")
        try:  # At least one non-detection
            result["non_detections"] = pd.DataFrame(result["non_detections"]).groupby("oid")
        except KeyError:  # to reproduce expected error for missing non-detections in loop
            result["non_detections"] = pd.DataFrame(columns=["oid"]).groupby("oid")
        output = []
        for oid, dets in result["detections"]:
            unique_measurement_ids = dets['measurement_id'].unique().tolist()
            output_message = {
                "oid": oid,
                "measurement_id": unique_measurement_ids,    
                "meanra": result["coords"][oid]["meanra"],
                "meandec": result["coords"][oid]["meandec"],
                "detections": dets.to_dict("records"),
            }
            try:
                output_message["non_detections"] = (
                    result["non_detections"].get_group(oid).to_dict("records")
                )
            except KeyError:
                output_message["non_detections"] = []
            output.append(output_message)
        return output

    """

    def post_execute(self, result: dict):
        self.produce_scribe(result["detections"])
        return result

    def produce_scribe(self, detections: list[dict]):
        count = 0
        for detection in detections:
            count += 1
            flush = False
            # Prevent further modification for next step
            detection = deepcopy(detection)
            if not detection.pop("new"):
                continue
            measurement_id = detection.pop("measurement_id")
            oid = detection.get("oid")
            is_forced = detection.pop("forced")
            set_on_insert = not detection.get("has_stamp", False)
            extra_fields = detection["extra_fields"]
            # remove possible elasticc extrafields
            for to_remove in ["prvDiaSources", "prvDiaForcedSources", "fp_hists"]:
                extra_fields.pop(to_remove, None)
            if "diaObject" in extra_fields:
                extra_fields["diaObject"] = pickle.loads(extra_fields["diaObject"])
            detection["extra_fields"] = extra_fields
            scribe_data = {
                "collection": "forced_photometry" if is_forced else "detection",
                "type": "update",
                "criteria": {"measurement_id": measurement_id, "oid": oid},
                "data": detection,
                "options": {"upsert": True, "set_on_insert": set_on_insert},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            if count == len(detections):
                flush = True
            self.scribe_producer.produce(scribe_payload, flush=flush, key=oid)

    """


    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()





