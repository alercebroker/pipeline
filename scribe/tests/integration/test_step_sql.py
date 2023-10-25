import json
import os
import pytest
import unittest

from sqlalchemy import text

from mongo_scribe.step import MongoScribe

from apf.producers.kafka import KafkaProducer
from db_plugins.db.sql._connection import PsqlDatabase

DB_CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    }
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "TOPICS": ["test_topic"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "command_consumer_2",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 2,
    "TIMEOUT": 10,
}

PRODUCER_CONFIG = {
    "TOPIC": "test_topic",
    "PARAMS": {"bootstrap.servers": "localhost:9092"},
    "SCHEMA": {
        "namespace": "db_operation",
        "type": "record",
        "name": "Command",
        "fields": [
            {"name": "payload", "type": "string"},
        ],
    },
}


@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("kafka_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = PsqlDatabase(DB_CONFIG["PSQL"])
        step_config = {
            "DB_CONFIG": DB_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
        }
        cls.db.create_db()
        with cls.db.session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO step(step_id, name, version, comments, date) 
                    VALUES ('v1', 'version 1', '1', '', current_timestamp)
                    """
                )
            )
            session.commit()
        cls.step = MongoScribe(config=step_config, db="sql")
        cls.producer = KafkaProducer(config=PRODUCER_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_db()

    def test_insert_objects_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {
                    "oid": "ZTF02ululeea",
                    "ndet": 1,
                    "firstmjd": 50001,
                    "g_r_max": 1.0,
                    "g_r_mean_corr": 0.9,
                    "meanra": 45,
                    "meandec": 45,
                },
            }
        )
        command2 = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {
                    "oid": "ZTF03ululeea",
                    "ndet": 1,
                    "firstmjd": 50001,
                    "g_r_max": 1.0,
                    "g_r_mean_corr": 0.9,
                    "meanra": 45,
                    "meandec": 45,
                },
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command2})
        self.producer.producer.flush(1)
        self.step.start()
        with self.db.session() as session:
            result = session.execute(text("SELECT * FROM object"))
            oids = [r[0] for r in result]
            assert "ZTF02ululeea" in oids
            assert "ZTF03ululeea" in oids

    def test_insert_detections_into_database(self):
        command = json.dumps(
            {
                "collection": "detection",
                "type": "update",
                "criteria": {"_id": "932472823", "candid": "932472823"},
                "data": {
                    "aid": "XChGbTDJWt",
                    "corrected": False,
                    "dec": 0.5753534762299036,
                    "dubious": False,
                    "e_dec": 0.10948950797319412,
                    "e_mag": 0.5646288990974426,
                    "e_mag_corr": None,
                    "e_mag_corr_ext": None,
                    "e_ra": 0.010245581157505512,
                    "extra_fields": {
                        "chinr": 1,
                        "distnr": 1.0,
                        "distpsnr1": 1.0,
                        "magnr": 10.0,
                        "sgscore1": 0.5,
                        "sharpnr": 0.0,
                        "sigmagnr": 1.0,
                        "unused": None,
                    },
                    "fid": "i",
                    "has_stamp": True,
                    "isdiffpos": -224123822,
                    "mag": 0.43753013014793396,
                    "mag_corr": None,
                    "mjd": 0.8421688401414276,
                    "oid": "ZTF04ululeea",
                    "parent_candid": "87654321",
                    "pid": -285679341253738006,
                    "ra": 0.10448978320949609,
                    "sid": "lIklphisOV",
                    "stellar": False,
                    "tid": "ZTF",
                },
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})

        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """ SELECT candid, oid FROM detection WHERE oid = 'ZTF04ululeea' """
                )
            )
            result = list(result)[0]
            assert result[0] == 932472823
            assert result[1] == "ZTF04ululeea"

    def test_upsert_non_detections(self):
        command = {
            "collection": "non_detection",
            "type": "update",
            "criteria": {
                "aid": "AL21XXX",
                "oid": "ZTF04ululeea",
                "mjd": 55000,
                "fid": "g",
            },
            "data": {
                "diffmaglim": 0.1,
                "pid": 4.3,
                "isdiffpos": 1,
                "ra": 99.0,
                "dec": 55.0,
                "magpsf": 220.0,
                "sigmapsf": 33.0,
                "step_id_corr": "steppu",
            },
        }

        self.producer.produce({"payload": json.dumps(command)})
        command["criteria"]["fid"] = "r"
        self.producer.produce({"payload": json.dumps(command)})

        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """ SELECT fid, mjd, diffmaglim FROM non_detection WHERE oid = 'ZTF04ululeea' """
                )
            )
            result = list(result)
            assert len(result) == 2
            assert result[0][0] == 1
            assert result[0][1] == 55000
            assert result[0][2] == 0.1

    def test_upsert_features(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO feature_version(version, step_id_feature, step_id_preprocess)
                    VALUES ('1.0.0', 'v1', 'v1')
                    """
                )
            )
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        command_data = [
            {
                "name": "feature1",
                "value": 55.0,
                "fid": "g",
            },
            {
                "name": "feature2",
                "value": 22.0,
                "fid": "r",
            },
            {
                "name": "feature3",
                "value": 130.0,
                "fid": "g",
            },
            {
                "name": "feature1",
                "value": 694211.0,
                "fid": "g",
            },
        ]

        commands = [
            {
                "payload": json.dumps(
                    {
                        "collection": "object",
                        "type": "update_features",
                        "criteria": {"_id": "AL21XXX", "oid": ["ZTF04ululeea"]},
                        "data": {
                            "features_version": "1.0.0",
                            "features_group": "ztf",
                            "features": [data],
                        },
                    }
                )
            }
            for data in command_data
        ]

        for command in commands:
            self.producer.produce(command)
        self.producer.producer.flush(4)
        self.step.start()
        with self.db.session() as session:
            result = session.execute(
                text(
                    """
                    SELECT name, value FROM feature WHERE oid = 'ZTF04ululeea' AND name = 'feature1'
                """
                )
            )
            result = list(result)[0]
            assert result[0] == "feature1"
            assert result[1] == 694211.0

    def test_upsert_probabilities(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        command_data = [
            {
                "classifier_name": "lc_classifier",
                "classifier_version": "1.0.0",
                "class1": 0.6,
                "class2": 0.4,
            },
            {
                "classifier_name": "lc_classifier",
                "classifier_version": "1.0.0",
                "class1": 0.35,
                "class2": 0.65,
            },
        ]

        commands = [
            {
                "payload": json.dumps(
                    {
                        "collection": "object",
                        "type": "update_probabilities",
                        "criteria": {"_id": "AL21XXX", "oid": ["ZTF04ululeea"]},
                        "data": data,
                    }
                )
            }
            for data in command_data
        ]

        for command in commands:
            self.producer.produce(command)

        self.producer.producer.flush(4)
        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """
                    SELECT * FROM probability WHERE oid = 'ZTF04ululeea'
                    """
                )
            )
            result = list(result)
            assert len(result) == 2
            for row in result:
                if row[1] == "class2":
                    assert row[4] == 0.65 and row[5] == 1
                else:
                    assert row[4] == 0.35 and row[5] == 2

    def test_update_object_stats(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        command = {
            "collection": "magstats",
            "criteria": {"oid": "ZTF04ululeea"},
            "data": {
                "corrected": False,
                "firstmjd": 0.04650036190916984,
                "lastmjd": 0.9794581336685745,
                "magstats": [
                    {
                        "corrected": False,
                        "dm_first": None,
                        "dmdt_first": None,
                        "dt_first": None,
                        "fid": "r",
                        "firstmjd": 0.04650036190916984,
                        "lastmjd": 0.9794581336685745,
                        "magfirst": 0.45738787694615635,
                        "magfirst_corr": None,
                        "maglast": 0.8891703032055938,
                        "maglast_corr": None,
                        "magmax": 0.9954724098279284,
                        "magmax_corr": None,
                        "magmean": 0.6485098616306181,
                        "magmean_corr": None,
                        "magmedian": 0.6183493589106022,
                        "magmedian_corr": None,
                        "magmin": 0.29146111487295745,
                        "magmin_corr": None,
                        "magsigma": 0.24471928116997924,
                        "magsigma_corr": None,
                        "ndet": 9,
                        "ndubious": 0,
                        "saturation_rate": None,
                        "sid": "ZTF",
                        "sigmadm_first": None,
                        "stellar": False,
                    },
                ],
                "meandec": 0.4861642021396574,
                "meanra": 0.5267988555440914,
                "ndet": 20,
                "sigmadec": 0.00568264450571807,
                "sigmara": 0.0006830686562186637,
                "stellar": False,
            },
            "type": "upsert",
        }

        self.producer.produce({"payload": json.dumps(command)})

        command["data"]["magstats"] = [
            {
                "corrected": False,
                "dm_first": None,
                "dmdt_first": None,
                "dt_first": None,
                "fid": "g",
                "firstmjd": 0.13577030206791907,
                "lastmjd": 0.95383888383811,
                "magfirst": 0.6249465481253661,
                "magfirst_corr": None,
                "maglast": 0.894922004401134,
                "maglast_corr": None,
                "magmax": 0.894922004401134,
                "magmax_corr": None,
                "magmean": 0.4860666136917287,
                "magmean_corr": None,
                "magmedian": 0.6062813119154207,
                "magmedian_corr": None,
                "magmin": 0.03844908454164819,
                "magmin_corr": None,
                "magsigma": 0.2650409061639637,
                "magsigma_corr": None,
                "ndet": 11,
                "ndubious": 0,
                "saturation_rate": None,
                "sid": "ZTF",
                "sigmadm_first": None,
                "stellar": False,
            }
        ]
        self.producer.produce({"payload": json.dumps(command)})
        self.producer.producer.flush(1)
        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """
                    SELECT object.oid as oid, fid, meanra, meandec, magstat.ndet as ndet
                    FROM object JOIN magstat on object.oid = magstat.oid 
                    """
                )
            )
            assert len(list(result))

    def test_upsert_xmatch(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45),
                    ('ZTF05ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        commands = [
            {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": "ALX123", "oid": ["ZTF04ululeea", "ZTF05ululeea"]},
                "data": {
                    "xmatch": {
                        "catoid": "J239263.32+240338.4",
                        "dist": 0.009726,
                        "catid": "allwise",
                    }
                },
            },
            {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": "ALX134", "oid": ["ZTF05ululeea", "ZTF04ululeea"]},
                "data": {
                    "xmatch": {
                        "catoid": "J239263.32+240338.4",
                        "dist": 0.615544,
                        "catid": "allwise",
                    }
                },
            },
        ]

        for command in commands:
            self.producer.produce({"payload": json.dumps(command)})

        self.step.start()
        with self.db.session() as session:
            result = session.execute(
                text(
                    """
                    SELECT * FROM xmatch WHERE oid = 'ZTF04ululeea' 
                    """
                )
            )
        assert len(list(result)) > 0

    def test_forced_photometry_insertion(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES
                    ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45),
                    ('ZTF05ululeea', 1, 50001, 1.0, 0.9, 45, 45)
                    ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

            command = {
                "collection": "forced_photometry",
                "type": "update",
                "criteria": {
                    "_id": "candid-pid",
                    "candid": "candid-pid",
                },
                "data": {
                    "mag": 10.0,
                    "pid": 423432,
                    "oid": "ZTF04ululeea",
                    "corrected": False,
                    "dubious": False,
                    "e_mag": 10.0,
                    "ra": 1,
                    "e_ra": 1,
                    "dec": 1,
                    "e_dec": 1,
                    "isdiffpos": 1,
                    "fid": "g",
                    "mjd": 1.0,
                    "has_stamp": True,
                    "parent_candid": "12345678",
                    "extra_fields": {
                        "diffmaglim": 19.198299407958984,
                        "pdiffimfilename": "ztf_20231018117361_000585_zr_c12_o_q2_scimrefdiffimg.fits",
                        "programpi": "Kulkarni",
                        "programid": 1,
                        "tblid": 2,
                        "nid": 2481,
                        "rcid": 45,
                        "field": 585,
                        "xpos": 1414.1082763671875,
                        "ypos": 307.8822937011719,
                        "chipsf": 1.7330743074417114,
                        "magap": 19.151899337768555,
                        "sigmagap": 0.1842000037431717,
                        "distnr": 0.24700157344341278,
                        "magnr": 18.56399917602539,
                        "sigmagnr": 0.017000000923871994,
                        "chinr": 0.44600000977516174,
                        "sharpnr": 0.007000000216066837,
                        "sky": -0.3027240037918091,
                        "magdiff": -0.046737998723983765,
                        "fwhm": 2.6700000762939453,
                        "classtar": 0.984000027179718,
                        "mindtoedge": 307.8822937011719,
                        "magfromlim": 0.869318962097168,
                        "seeratio": 0.8929277658462524,
                        "aimage": 0.8220000267028809,
                        "bimage": 0.6970000267028809,
                        "aimagerat": 0.307865172624588,
                        "bimagerat": 0.26104867458343506,
                        "elong": 1.1793400049209595,
                        "nneg": 4,
                        "nbad": 0,
                        "rb": 0.9742857217788696,
                        "ssdistnr": -999.0,
                        "ssmagnr": -999.0,
                        "ssnamenr": "null",
                        "sumrat": 0.9896100759506226,
                        "magapbig": 19.161300659179688,
                        "sigmagapbig": 0.23800000548362732,
                        "ranr": 249.21548461914062,
                        "decnr": 20.66690444946289,
                        "sgmag1": 18.89550018310547,
                        "srmag1": 18.409900665283203,
                        "simag1": 18.60740089416504,
                        "szmag1": 18.634599685668945,
                        "sgscore1": 0.9938330054283142,
                        "distpsnr1": 0.09082309156656265,
                        "ndethist": 1636,
                        "ncovhist": 3829,
                        "jdstarthist": 2458600.0,
                        "jdendhist": 2460235.5,
                        "scorr": 10.018070220947266,
                        "tooflag": 0,
                        "objectidps1": 1.3280249332629504e17,
                        "objectidps2": 1.3280249332629504e17,
                        "sgmag2": -999.0,
                        "srmag2": 21.96809959411621,
                        "simag2": 20.82509994506836,
                        "szmag2": 19.954500198364258,
                        "sgscore2": 0.9708539843559265,
                        "distpsnr2": 11.368770599365234,
                        "objectidps3": 1.3280249332629504e17,
                        "sgmag3": 21.577600479125977,
                        "srmag3": 21.18670082092285,
                        "simag3": 20.908899307250977,
                        "szmag3": 20.631799697875977,
                        "sgscore3": 0.731939971446991,
                        "distpsnr3": 16.569791793823242,
                        "nmtchps": 7,
                        "rfid": 585120145,
                        "jdstartref": 2458159.0,
                        "jdendref": 2458253.0,
                        "nframesref": 15,
                        "rbversion": "t17_f5_c3",
                        "dsnrms": 9.402395248413086,
                        "ssnrms": 17.510025024414062,
                        "dsdiff": -8.107629776000977,
                        "magzpsci": 26.11090087890625,
                        "magzpsciunc": 3.0955998227000237e-05,
                        "magzpscirms": 0.050816699862480164,
                        "nmatches": 1230,
                        "clrcoeff": -0.03848319873213768,
                        "clrcounc": 8.465119753964245e-05,
                        "zpclrcov": -4.699999863078119e-06,
                        "zpmed": 26.298999786376953,
                        "clrmed": 0.5509999990463257,
                        "clrrms": 0.2764450013637543,
                        "neargaia": 0.08682537823915482,
                        "neargaiabright": 34.5357780456543,
                        "maggaia": 18.35614013671875,
                        "maggaiabright": 12.212721824645996,
                        "exptime": 30.0,
                        "drb": 0.999990701675415,
                        "drbversion": "d6_m7",
                        "brokerIngestTimestamp": 1697657219,
                        "surveyPublishTimestamp": 1697598668800.0,
                        "parent_candid": None,
                        "sciinpseeing": 4.046999931335449,
                        "scibckgnd": 64.53669738769531,
                        "scisigpix": 9.99390983581543,
                        "adpctdif1": 0.14711999893188477,
                        "adpctdif2": 0.11690100282430649,
                        "forcediffimfluxunc": 70.34915161132812,
                        "procstatus": "0",
                    },
                },
            }

            self.producer.produce({"payload": json.dumps(command)})
            self.producer.produce({"payload": json.dumps(command)})
            self.producer.producer.flush(1)

            self.step.start()
            with self.db.session() as session:
                result = session.execute(
                    text(
                        """
                    SELECT * FROM forced_photometry WHERE oid = 'ZTF04ululeea' 
                    """
                    )
                )
            assert True

    def test_upsert_metadata(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES
                    ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45)
                    ON CONFLICT DO NOTHING"""
                )
            )
            session.execute(
                text(
                    """
                    INSERT INTO detection
                    (candid, oid, mjd, fid, pid, isdiffpos, ra, dec, \
                     magpsf, sigmapsf, corrected, dubious, has_stamp, step_id_corr)
                     VALUES
                     (9192746, 'ZTF04ululeea', 55500, 1, 654321, 1, 45, 45, \
                        20.0, 1.0, false, false, false, 'ALERCE')
                    """
                )
            )
            session.commit()
            command = {
                "collection": "metadata",
                "type": "upsert",
                "criteria": {
                    "_id": "candid-pid",
                    "candid": "candid-pid",  # maybe add oid here?
                },
                "data": {
                    "oid": "ZTF04ululeea",
                    "candid": 9192746,
                    "rfid": 4000000,
                    "extra_fields": {
                        "jd": 2459966.7402431,
                        "fid": 1,
                        "pid": 2212240240015,
                        "diffmaglim": 19.8356990814209,
                        "pdiffimfilename": "/ztf/archive/sci/2023/0122/240231/ztf_20230122240231_000354_zg_c01_o_q1_scimrefdiffimg.fits.fz",
                        "programpi": "Kulkarni",
                        "programid": 1,
                        "candid": 9192746,
                        "isdiffpos": "t",
                        "tblid": 2,
                        "nid": 2212,
                        "rcid": 0,
                        "field": 354,
                        "xpos": 242.45599365234375,
                        "ypos": 481.0260009765625,
                        "mjdstartref": 55000,
                        "mjdendref": 60000,
                        "nframesref": 2,
                        "ra": 72.3385553,
                        "dec": -11.9220759,
                        "magpsf": 18.461200714111328,
                        "sigmapsf": 0.20323200523853302,
                        "chipsf": 5.37214994430542,
                        "magap": 18.402599334716797,
                        "sigmagap": 0.09950000047683716,
                        "distnr": 0.6387689709663391,
                        "magnr": 15.371999740600586,
                        "sigmagnr": 0.012000000104308128,
                        "chinr": 0.4580000042915344,
                        "sharpnr": 0.008999999612569809,
                        "sky": -1.4953199625015259,
                        "magdiff": -0.05859399959445,
                        "fwhm": 1.8398799896240234,
                        "classtar": 0.9340000152587891,
                        "mindtoedge": 242.45599365234375,
                        "magfromlim": 1.4330500364303589,
                        "seeratio": 2.0,
                        "aimage": 0.7059999704360962,
                        "bimage": 0.5709999799728394,
                        "aimagerat": 0.3837209939956665,
                        "bimagerat": 0.3103469908237457,
                        "elong": 1.236430048942566,
                        "nneg": 8,
                        "nbad": 0,
                        "rb": 0.46142899990081787,
                        "ssdistnr": None,
                        "ssmagnr": None,
                        "ssnamenr": None,
                        "sumrat": 0.6994069814682007,
                        "magapbig": 18.365100860595703,
                        "sigmagapbig": 0.11460000276565552,
                        "ranr": 72.3385822,
                        "decnr": -11.9218982,
                        "scorr": 10.3521,
                        "magzpsci": 26.209199905395508,
                        "magzpsciunc": 1.6734900782466866e-05,
                        "magzpscirms": 0.0390545018017292,
                        "clrcoeff": -0.04869139939546585,
                        "clrcounc": 3.019940049853176e-05,
                        "rbversion": "t17_f5_c3",
                        "unique1": False,
                    },
                },
            }
        self.producer.produce({"payload": json.dumps(command)})
        self.producer.produce({"payload": json.dumps(command)})
        self.producer.producer.flush()
        
        self.step.start()
        with self.db.session() as session:
            result = session.execute(
                text("""SELECT * FROM gaia_ztf WHERE oid='ZTF04ululeea'""")
            ).all()

        assert True
