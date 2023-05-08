import logging
import os
import glob


def test_step(kafka_service, env_variables, caplog, kafka_consumer):
    from scripts.run_step import step

    caplog.set_level(logging.INFO)
    step.start()
    nmessages = 0
    total_consumed = len(glob.glob(os.path.join("data", "*.avro")))
    for _ in kafka_consumer.consume():
        nmessages += 1
    assert nmessages == total_consumed
