"""Replay the real-examples fixture through the step, dry-run or to a real scribe.

The step normally consumes the multisurvey feature_step topic. That topic is
empty, so this drives the real step lifecycle (`GenericStep.start`) from a file
instead: the 992 objects in `tests/integration/data/real_examples.json.gz`,
rebuilt as feature_step messages and consumed through apf's `JSONConsumer`.
Everything after the consumer is the production path -- pre_execute, execute,
post_execute, produce_scribe, and the flush apf does before committing.

Two modes, chosen by `--send`:

- default (dry run): the scribe producer is replaced by one that records every
  command to `<out>/scribe_commands.jsonl`, one `{"key", "payload"}` per line,
  and nothing leaves the machine. The model, the taxonomy lookup and the row
  building are all real.
- `--send`: the yaml's `SCRIBE_PRODUCER_CONFIG` is used as is, so the commands
  go to the real scribe topic. The scribe then writes them to
  `multisurvey_ztf.probability`. Run the dry run first and look at the file.

Messages are built the way `test_real_data_equivalence.py` builds its frame:
feature names translated from the database spelling to the model's, rows from
a superseded computation dropped, every model column present (null where the
object has no row), and a single detection carrying the fixture's `lastmjd`
because that is the only thing the step reads from `detections`.

    python scripts/replay_fixture.py \
        --config local_config.yaml \
        --model-path /path/to/model/2.1.0 \
        --out /tmp/replay

`--config` is any yaml with `PSQL_CONFIG`, `MODEL_CONFIG` and
`SCRIBE_PRODUCER_CONFIG` blocks; `--model-path` overrides the yaml's
`model_path`, which in the local config is a placeholder.
"""
import argparse
import gzip
import json
import logging
import os
import pathlib
import sys

import pandas as pd
import yaml
from apf.producers import GenericProducer

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

# `get_class` resolves the recording producer by dotted path; make this module
# reachable under its own name even when it runs as __main__.
sys.modules.setdefault("replay_fixture", sys.modules[__name__])

from tests.integration.test_real_data_equivalence import (  # noqa: E402
    FIXTURE,
    _current_features,
    _model_column,
)

log = logging.getLogger("replay_fixture")


class RecordingScribeProducer(GenericProducer):
    """Stand-in for the scribe producer: appends each command to a jsonl file.

    Same shape the step hands the real producer -- the message is
    `{"payload": <json string>}` keyed by the oid -- decoded back so the file
    is readable. `flush` is a no-op; writes are synchronous.
    """

    def __init__(self, config):
        super().__init__(config=config)
        path = pathlib.Path(config["FILE_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "w", encoding="utf-8")
        self.count = 0

    def produce(self, message=None, **kwargs):
        key = kwargs.get("key")
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        record = {"key": key, "payload": json.loads(message["payload"])}
        self._handle.write(json.dumps(record) + "\n")
        self.count += 1

    def flush(self):
        self._handle.flush()


def load_fixture(path: pathlib.Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def model_feature_list(model_path: str) -> list:
    pickle = os.path.join(model_path, "hierarchical_random_forest_model.pkl")
    return list(pd.read_pickle(pickle)["feature_list"])


def build_message(entry: dict, model_features: list) -> dict:
    """One feature_step-shaped message for a fixture object.

    Only the fields the step reads: `oid` as the string the Avro schema types
    it as, the full `features` record in the model's spelling, and one
    detection whose mjd is the fixture's `lastmjd` (the step takes the max mjd
    over detections, so one entry reproduces it exactly).
    """
    values = {
        _model_column(name, band): value
        for name, band, value in _current_features(entry)
    }
    return {
        "oid": str(entry["oid"]),
        "detections": [{"mjd": float(entry["lastmjd"]), "forced": False}],
        "features": {name: values.get(name) for name in model_features},
    }


def write_batches(messages: list, batch_size: int, path: pathlib.Path) -> int:
    """`JSONConsumer` yields one top-level element per iteration and the step
    treats a list as a batch, so the file is a list of batches."""
    batches = [
        messages[start : start + batch_size]
        for start in range(0, len(messages), batch_size)
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(batches, handle)
    return len(batches)


def build_config(args, batches_path: pathlib.Path, out: pathlib.Path) -> dict:
    with open(args.config) as handle:
        base = yaml.safe_load(handle)

    model_config = dict(base["MODEL_CONFIG"])
    model_config["PARAMS"] = {**model_config.get("PARAMS", {}), "model_path": args.model_path}

    if args.send:
        scribe = dict(base["SCRIBE_PRODUCER_CONFIG"])
        # A stuck broker raises here instead of hanging the run forever.
        scribe.setdefault("FLUSH_TIMEOUT", args.flush_timeout)
    else:
        scribe = {
            "CLASS": "replay_fixture.RecordingScribeProducer",
            "FILE_PATH": str(out / "scribe_commands.jsonl"),
        }

    return {
        "CONSUMER_CONFIG": {
            "CLASS": "apf.consumers.JSONConsumer",
            "FILE_PATH": str(batches_path),
        },
        "SCRIBE_PRODUCER_CONFIG": scribe,
        "PSQL_CONFIG": base["PSQL_CONFIG"],
        "MODEL_CONFIG": model_config,
        "PRODUCER_CONFIG": {},
        "METRICS_CONFIG": {},
        "FEATURE_FLAGS": {"PROMETHEUS": False},
        "LOGGING_DEBUG": bool(base.get("LOGGING_DEBUG", False)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="yaml with PSQL/MODEL/SCRIBE blocks")
    parser.add_argument("--model-path", required=True, help="directory holding the BHRF pickle")
    parser.add_argument("--fixture", type=pathlib.Path, default=FIXTURE)
    parser.add_argument("--out", type=pathlib.Path, required=True, help="output directory")
    parser.add_argument("--batch-size", type=int, default=100, help="messages per batch (production consumes 100)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N objects")
    parser.add_argument("--flush-timeout", type=float, default=300.0, help="seconds to wait for delivery with --send")
    parser.add_argument("--send", action="store_true", help="produce to the yaml's real scribe topic")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    objects = load_fixture(args.fixture)["objects"]
    if args.limit is not None:
        objects = objects[: args.limit]
    model_features = model_feature_list(args.model_path)
    messages = [build_message(entry, model_features) for entry in objects]

    batches_path = args.out / "batches.json"
    n_batches = write_batches(messages, args.batch_size, batches_path)
    log.info(
        "%d objects from %s -> %d batches of up to %d in %s",
        len(messages), args.fixture.name, n_batches, args.batch_size, batches_path,
    )

    config = build_config(args, batches_path, args.out)
    scribe = config["SCRIBE_PRODUCER_CONFIG"]
    if args.send:
        log.warning(
            "SENDING to the real scribe: topic %s on %s",
            scribe.get("TOPIC"), scribe.get("PARAMS", {}).get("bootstrap.servers"),
        )
    else:
        log.info("dry run: commands recorded to %s", scribe["FILE_PATH"])

    from lc_classification_multisurvey_step.step import LateClassifierMultisurvey

    step = LateClassifierMultisurvey(config=config, level=logging.INFO)
    step.start()

    if not args.send:
        log.info("recorded %d commands", step.scribe_producer.count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
