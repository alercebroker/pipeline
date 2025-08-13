import logging
from apf.core.settings import config_from_yaml_file

from core.DB.database_connection import PSQLConnection
from correction_multistream_step.step import CorrectionMultistreamZTFStep


def set_logger(settings):
    level = logging.INFO
    if settings.get("LOGGING_DEBUG"):
        level = logging.DEBUG

    logger = logging.getLogger("alerce")
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)7s %(name)36s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.setLevel(level)

    logger.addHandler(handler)
    return logger


def step_creator():
    settings = config_from_yaml_file("/tests/test_utils/config.yaml")
    logger = set_logger(settings)

    db_sql = PSQLConnection(settings["PSQL_CONFIG"])

    step_params = {"config": settings, "db_sql": db_sql}

    if settings["FEATURE_FLAGS"]["SKIP_MJD_FILTER"]:
        logger.info(
            "This step won't filter detections by MJD. \
            Keep this in mind when using for ELAsTiCC"
        )

    return CorrectionMultistreamZTFStep(**step_params)


def validate_alert_fields_message(data):
    num_msg = 0
    for message in data:
        required_top_fields = [
            "oid",
            "measurement_id",
            "meanra",
            "meandec",
            "detections",
            "non_detections",
        ]  # List of columns expected in the outer part of any alert
        missing_top_fields = [field for field in required_top_fields if field not in message]

        if missing_top_fields:
            print(
                f"The following fields are missing at the top level of the message number {num_msg}: {missing_top_fields}"
            )
            return False

        unexpected_fields = [field for field in message if field not in required_top_fields]
        if unexpected_fields:
            print(f"Unexpected top-level fields found: {unexpected_fields}")
            return False
        num_msg = num_msg + 1
    print("All top-level fields are present")
    return True


def validate_non_detection_fields_message(data):
    num_msg = 0
    for message in data:
        if "non_detections" in message:
            if not isinstance(message["non_detections"], list):
                print(" The non detection field should be a list")
                return False

            if message["non_detections"]:
                required_non_det_fields = ["oid", "sid", "tid", "band", "mjd", "diffmaglim"]
                for i, non_det in enumerate(message["non_detections"]):
                    missing_non_det_fields = [
                        field for field in required_non_det_fields if field not in non_det
                    ]
                    if missing_non_det_fields:
                        print(
                            f"Non-detection at index {i} is missing fields: {missing_non_det_fields}"
                        )
                        return False

                    unexpected_non_det_fields = [
                        field for field in non_det if field not in required_non_det_fields
                    ]
                    if unexpected_non_det_fields:
                        print(
                            f"Unexpected fields found in non-detection at index {i} in message number {num_msg}: {unexpected_non_det_fields}"
                        )
                        return False

        num_msg += 1
    print("All non detection fields are present")
    return True


def validate_detection_fields_message(data):
    num_msg = 0
    for message in data:
        if "detections" in message:
            if not isinstance(message["detections"], list):
                print(" The detection field should be a list")
                return False

            if message["detections"]:
                required_det_fields = [
                    "oid",
                    "sid",
                    "tid",
                    "pid",
                    "band",
                    "measurement_id",
                    "mjd",
                    "ra",
                    "e_ra",
                    "dec",
                    "e_dec",
                    "mag",
                    "e_mag",
                    "mag_corr",
                    "e_mag_corr",
                    "e_mag_corr_ext",
                    "isdiffpos",
                    "parent_candid",
                    "has_stamp",
                    "corrected",
                    "dubious",
                    "stellar",
                    "forced",
                    "new",
                    "extra_fields",
                ]
                for i, det in enumerate(message["detections"]):
                    missing_det_fields = [
                        field for field in required_det_fields if field not in det
                    ]
                    if missing_det_fields:
                        print(f"Detection at index {i} is missing fields: {missing_det_fields}")
                        return False

                    unexpected_det_fields = [
                        field for field in det if field not in required_det_fields
                    ]
                    if unexpected_det_fields:
                        print(
                            f"Unexpected fields found in detection at index {i} in message number {num_msg}: {unexpected_det_fields}"
                        )
                        return False

        num_msg += 1
    print("All detection fields are present")
    return True


def message_validation(data):
    return (
        validate_alert_fields_message(data)
        and validate_non_detection_fields_message(data)
        and validate_detection_fields_message(data)
    )


# Function that receives all the messages, selects the one with the oid we have crafted to test the step
# and compares the number of detections to the expected number. Same for ndets
def output_expected_count(data, oid, expected_dets, expected_non_dets):
    matching_dicts = [item for item in data if item.get("oid") == oid][0]
    len_detections = len(matching_dicts["detections"])
    len_non_detections = len(matching_dicts["non_detections"])
    if len_detections != expected_dets or len_non_detections != expected_non_dets:
        print(f"The number of detections and non detections are not correct for oid {oid}")
        print(f"Expected {expected_dets} detections and {expected_non_dets} non detections")
        print(f"Got {len_detections} detections and {len_non_detections} non detections.")
        return False
    print(f"The number of detections and non detections are correct for oid {oid}")
    return True


# Consuming all messages at once due to simplicity with the offset
# Original json is of len 103 messages. There's oids with repetitions
if __name__ == "__main__":
    step = step_creator()
    logger = logging.getLogger(f"alerce.{step.__class__.__name__}")
    """Start running the step."""
    step._pre_consume()
    for message in step.consumer.consume():
        preprocessed_msg = step._pre_execute(message)
        breakpoint()
        if len(preprocessed_msg) == 0:
            logger.info("Message of len zero after pre_execute")
            continue
        try:
            result = step.execute(preprocessed_msg)
        except Exception as error:
            logger.debug("Error at execute")
            logger.debug(f"The message(s) that caused the error: {message}")
            raise error
        result = step._post_execute(result)
        result = step._pre_produce(result)
        step.producer.produce(result)

        # First thing to test! The number of output messages corresponds to the expected number
        # Since the number of output messages depends on the number of unique oids in the original input, we'll check against that
        original_step_oids = [item["oid"] for item in step.consumer.messages]
        len_expected_output = len(set(original_step_oids))
        len_messages_output = len(step.producer.pre_produce_message[0])
        assert len_expected_output == len_messages_output

        # Second thing to test! Verify that all the columns expected are present in the output! To do so, we'll pick the first message of the output and just
        # verify the presence of the columns we expect due to the step and database modeles! We won't care about the values for now
        messages_produce = step.producer.pre_produce_message[
            0
        ]  # All messages produced are saved in this
        assert message_validation(messages_produce) == True

        # Third thing to test! Verify that the number of detections and forced photometry corresponds to the expected number
        # This can be done since we know the exact message tested and the data inside the ddbb, so we already know the expected number
        assert output_expected_count(
            messages_produce, 1111111111, expected_dets=3, expected_non_dets=3
        )
        assert output_expected_count(
            messages_produce, 879453281, expected_dets=6, expected_non_dets=6
        )
