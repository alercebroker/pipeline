import pickle
import pytest
from confluent_kafka.admin import AdminClient, NewTopic
from apf.producers import KafkaProducer
import pathlib
import os
from fastavro.utils import generate_many
import random

from tests.shared.sorting_hat_schema import SCHEMA


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return (
        pathlib.Path(pytestconfig.rootdir) / "tests/integration/docker-compose.yaml"
    ).absolute()


@pytest.fixture(scope="session")
def docker_compose_command():
    v2 = False
    if os.getenv("COMPOSE", "v1") == "v2":
        v2 = True
    return "docker compose" if v2 else "docker-compose"


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["sorting-hat"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(f"Can't create topic {topic}")
            print(e)
            return False
    produce_messages("sorting-hat")
    return True


@pytest.fixture(scope="session")
def kafka_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    port = docker_services.port_for("kafka", 9092)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_kafka(server)
    )
    return server


@pytest.fixture
def env_variables():
    env_variables_dict = {
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "sorting-hat",
        "CONSUMER_GROUP_ID": "prv-test",
        "METRICS_HOST": "localhost:9092",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    return env_variables_dict


def produce_messages(topic):
    producer = KafkaProducer(
        {
            "PARAMS": {"bootstrap.servers": "localhost:9092"},
            "TOPIC": topic,
            "SCHEMA": SCHEMA,
        }
    )
    messages = generate_many(SCHEMA, 10)
    producer.set_key_field("aid")
    random.seed(42)

    for message in messages:
        message["tid"] = "ZTF" if random.random() > 0.5 else "ATLAS"
        if str(message["tid"]).lower() == "ztf":
            message["extra_fields"] = extra_fields_generator()
        producer.produce(message)


def extra_fields_generator():
    prv_candidates = [
        {
            "jd": 2459951.7121181,
            "fid": 1,
            "pid": 2197212111315,
            "diffmaglim": 19.03019905090332,
            "pdiffimfilename": "/ztf/archive/sci/2023/0107/212106/ztf_20230107212106_000355_zg_c04_o_q2_scimrefdiffimg.fits.fz",
            "programpi": "Kulkarni",
            "programid": 1,
            "candid": None,
            "isdiffpos": None,
            "tblid": None,
            "nid": 2197,
            "rcid": 13,
            "field": 355,
            "xpos": None,
            "ypos": None,
            "ra": None,
            "dec": None,
            "magpsf": None,
            "sigmapsf": None,
            "chipsf": None,
            "magap": None,
            "sigmagap": None,
            "distnr": None,
            "magnr": None,
            "sigmagnr": None,
            "chinr": None,
            "sharpnr": None,
            "sky": None,
            "magdiff": None,
            "fwhm": None,
            "classtar": None,
            "mindtoedge": None,
            "magfromlim": None,
            "seeratio": None,
            "aimage": None,
            "bimage": None,
            "aimagerat": None,
            "bimagerat": None,
            "elong": None,
            "nneg": None,
            "nbad": None,
            "rb": None,
            "ssdistnr": None,
            "ssmagnr": None,
            "ssnamenr": None,
            "sumrat": None,
            "magapbig": None,
            "sigmagapbig": None,
            "ranr": None,
            "decnr": None,
            "scorr": None,
            "magzpsci": 26.258800506591797,
            "magzpsciunc": 3.7078898458275944e-05,
            "magzpscirms": 0.04231010004878044,
            "clrcoeff": -0.03463109955191612,
            "clrcounc": 8.699159661773592e-05,
            "rbversion": "t17_f5_c3",
        },
        {
            "jd": 2459951.7525347,
            "fid": 2,
            "pid": 2197252531315,
            "diffmaglim": 19.135099411010742,
            "pdiffimfilename": "/ztf/archive/sci/2023/0107/252512/ztf_20230107252512_000355_zr_c04_o_q2_scimrefdiffimg.fits.fz",
            "programpi": "Kulkarni",
            "programid": 1,
            "candid": None,
            "isdiffpos": None,
            "tblid": None,
            "nid": 2197,
            "rcid": 13,
            "field": 355,
            "xpos": None,
            "ypos": None,
            "ra": None,
            "dec": None,
            "magpsf": None,
            "sigmapsf": None,
            "chipsf": None,
            "magap": None,
            "sigmagap": None,
            "distnr": None,
            "magnr": None,
            "sigmagnr": None,
            "chinr": None,
            "sharpnr": None,
            "sky": None,
            "magdiff": None,
            "fwhm": None,
            "classtar": None,
            "mindtoedge": None,
            "magfromlim": None,
            "seeratio": None,
            "aimage": None,
            "bimage": None,
            "aimagerat": None,
            "bimagerat": None,
            "elong": None,
            "nneg": None,
            "nbad": None,
            "rb": None,
            "ssdistnr": None,
            "ssmagnr": None,
            "ssnamenr": None,
            "sumrat": None,
            "magapbig": None,
            "sigmagapbig": None,
            "ranr": None,
            "decnr": None,
            "scorr": None,
            "magzpsci": 26.197599411010742,
            "magzpsciunc": 1.8071899830829352e-05,
            "magzpscirms": 0.04117390140891075,
            "clrcoeff": 0.10742300003767014,
            "clrcounc": 2.9680899388040416e-05,
            "rbversion": "t17_f5_c3",
        },
        {
            "jd": 2459956.7757639,
            "fid": 1,
            "pid": 2202275760015,
            "diffmaglim": 19.776199340820312,
            "pdiffimfilename": "/ztf/archive/sci/2023/0112/275764/ztf_20230112275764_000354_zg_c01_o_q1_scimrefdiffimg.fits.fz",
            "programpi": "Kulkarni",
            "programid": 1,
            "candid": 2202275760015015000,
            "isdiffpos": "t",
            "tblid": 0,
            "nid": 2202,
            "rcid": 0,
            "field": 354,
            "xpos": 229.73500061035156,
            "ypos": 432.718994140625,
            "ra": 72.3385236,
            "dec": -11.9219861,
            "magpsf": 18.929399490356445,
            "sigmapsf": 0.18436799943447113,
            "chipsf": 3.6311099529266357,
            "magap": 18.518199920654297,
            "sigmagap": 0.1412999927997589,
            "distnr": 0.37338098883628845,
            "magnr": 15.371999740600586,
            "sigmagnr": 0.012000000104308128,
            "chinr": 0.4580000042915344,
            "sharpnr": 0.008999999612569809,
            "sky": 0.8764089941978455,
            "magdiff": -0.4111819863319397,
            "fwhm": 1.3357199430465698,
            "classtar": 0.9919999837875366,
            "mindtoedge": 229.73500061035156,
            "magfromlim": 1.2579799890518188,
            "seeratio": 2.0,
            "aimage": 0.5,
            "bimage": 0.4429999887943268,
            "aimagerat": 0.3743300139904022,
            "bimagerat": 0.33165600895881653,
            "elong": 1.1286699771881104,
            "nneg": 6,
            "nbad": 0,
            "rb": 0.5757139921188354,
            "ssdistnr": None,
            "ssmagnr": None,
            "ssnamenr": None,
            "sumrat": 0.743707001209259,
            "magapbig": 18.37179946899414,
            "sigmagapbig": 0.15039999783039093,
            "ranr": 72.3385822,
            "decnr": -11.9218982,
            "scorr": 8.87609,
            "magzpsci": 26.106399536132812,
            "magzpsciunc": 1.1533899851201568e-05,
            "magzpscirms": 0.03296779841184616,
            "clrcoeff": -0.06164040043950081,
            "clrcounc": 2.0050300008733757e-05,
            "rbversion": "t17_f5_c3",
        },
        {
            "jd": 2459966.7402431,
            "fid": 1,
            "pid": 2212240240015,
            "diffmaglim": 19.8356990814209,
            "pdiffimfilename": "/ztf/archive/sci/2023/0122/240231/ztf_20230122240231_000354_zg_c01_o_q1_scimrefdiffimg.fits.fz",
            "programpi": "Kulkarni",
            "programid": 1,
            "candid": 2212240240015015002,
            "isdiffpos": "t",
            "tblid": 2,
            "nid": 2212,
            "rcid": 0,
            "field": 354,
            "xpos": 242.45599365234375,
            "ypos": 481.0260009765625,
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
        },
    ]
    extra_fields = {"prv_candidates": pickle.dumps(prv_candidates)}
    return extra_fields
