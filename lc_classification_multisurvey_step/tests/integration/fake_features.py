"""Synthetic `feature_step` output messages, for running this step without one.

The multisurvey feature_step lives on another branch and its topic is empty, so
there is nothing upstream to consume. What this step actually needs from a
message is small -- an oid that parses as an int, a full features record, and at
least one detection to take `lastmjd` from -- so the messages are generated from
the schema rather than captured.

`fastavro.utils.generate_many` fills every field of `schemas/feature_step/
output.avsc` with random values, which keeps the messages schema-valid for free
as the schema changes. Three fields are then overwritten, because random values
for them are not merely unrealistic but wrong:

- `oid`: generated as a random 10-character string, which `filter_messages`
  drops (`int(oid)` raises), so every generated message would be discarded
  before reaching the model.
- `features`: typed `["null", "features_record_ztf"]`, so roughly half the
  generated messages carry no features at all and are dropped too. Forced to a
  full 209-field record -- the model's `feature_list` is 199 names and every one
  of them is in the schema, so a full record always covers it.
- `detections[].mjd`: generated anywhere in the double range, including
  negatives. `lastmjd` is written to the database, so keep it plausible.

The probabilities that come out are meaningless -- these are random features.
This exercises the wiring, not the science.
"""
import pathlib
import random

from fastavro.schema import load_schema
from fastavro.utils import generate_many

SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent.parent.parent,
    "schemas",
    "feature_step",
    "output.avsc",
)

# Arbitrary, but a bigint the way the multisurvey masterid is (design §4), and
# far from the ZTF oids in any real database so a stray local write is obvious.
FIRST_OID = 900000000000000000

# Plausible ZTF-era MJD, so `probability.lastmjd` looks like the real thing.
BASE_MJD = 60000.0


def _feature_names() -> list:
    """The 209 feature field names, read off the schema the messages are built from."""
    schema = load_schema(str(SCHEMA_PATH))
    features_union = next(f for f in schema["fields"] if f["name"] == "features")
    record = next(branch for branch in features_union["type"] if branch != "null")
    return [field["name"] for field in record["fields"]]


def generate_messages(
    count: int, *, seed: int = 42, first_oid: int = FIRST_OID, with_features: bool = True
) -> list:
    """`count` schema-valid feature_step messages, one per distinct oid.

    Deterministic for a given `seed`, so a failing run can be replayed.

    `with_features=False` returns the same messages with `features` set to null,
    which is the shape `filter_messages` drops -- the only way to build the
    "nothing to classify" case without hand-writing a message.
    """
    rng = random.Random(seed)
    names = _feature_names()
    schema = load_schema(str(SCHEMA_PATH))

    messages = []
    for index, message in enumerate(generate_many(schema, count)):
        oid = str(first_oid + index)
        message["oid"] = oid

        for offset, detection in enumerate(message["detections"]):
            detection["oid"] = oid
            detection["mjd"] = BASE_MJD + index + offset * 0.01
            detection["forced"] = False

        message["features"] = (
            {name: rng.uniform(-5.0, 5.0) for name in names} if with_features else None
        )
        messages.append(message)

    return messages
