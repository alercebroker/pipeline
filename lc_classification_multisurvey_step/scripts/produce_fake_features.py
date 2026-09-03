"""Fill a local topic with synthetic feature_step messages, so the step has input.

For running the step by hand against the harness in `tests/integration` -- bring
the broker up, produce, then start the step:

    docker compose -f tests/integration/docker-compose.yml up -d
    python scripts/produce_fake_features.py --count 20
    CONFIG_FROM_YAML=yes CONFIG_YAML_PATH=$(pwd)/local_config_docker.yaml \\
        python scripts/run_step.py

The database still has to be seeded for the step to start; `tests/integration/
conftest.py` does that for the tests, and the README spells out the same two
statements for a manual run.

The generator is shared with the integration tests (`tests.integration.
fake_features`) so both send the same shape of message.
"""
import argparse
import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

from apf.producers import KafkaSchemalessProducer  # noqa: E402

from tests.integration.fake_features import SCHEMA_PATH, generate_messages  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--topic", default="ztf-features")
    parser.add_argument("--server", default="localhost:9092")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="produce messages with features=null, which the step drops",
    )
    args = parser.parse_args()

    messages = generate_messages(
        args.count, seed=args.seed, with_features=not args.no_features
    )
    producer = KafkaSchemalessProducer(
        {
            "TOPIC": args.topic,
            "PARAMS": {"bootstrap.servers": args.server},
            "SCHEMA_PATH": str(SCHEMA_PATH),
        }
    )
    for message in messages:
        producer.produce(message)
    producer.producer.flush()

    oids = [message["oid"] for message in messages]
    print(f"produced {len(messages)} message(s) to '{args.topic}' on {args.server}")
    print(f"oids: {oids[0]}..{oids[-1]}")


if __name__ == "__main__":
    main()
