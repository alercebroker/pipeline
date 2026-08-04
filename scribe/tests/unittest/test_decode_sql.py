import json
import unittest

from mongo_scribe.sql.command.decode import command_factory
from mongo_scribe.sql.command.exceptions import MongoDialectCommandException
from mongo_scribe.sql.command.commands import UpsertXmatchCommand


def _payload(command: dict) -> str:
    return json.dumps(command)


class SqlCommandFactoryTest(unittest.TestCase):
    def test_mongo_dialect_object_update_is_skipped(self):
        """The legacy Mongo-dialect object update -- a generic
        {"type": "update", "collection": "object"} with no xmatch, still emitted
        per-object by magstats_step for the retired MongoDB backend -- must raise
        the distinct MongoDialectCommandException so the step skips it quietly
        instead of WARN-flooding it as an invalid drop. See MONGODB-LEGACY.md."""
        msg = _payload(
            {
                "type": "update",
                "collection": "object",
                "criteria": {"_id": "ZTF18abdlcao"},
                "data": {"ndet": 1003, "meanra": 259.34, "meandec": -20.38},
                "options": {"upsert": True},
            }
        )
        with self.assertRaises(MongoDialectCommandException):
            command_factory(msg)

    def test_mongo_dialect_exception_is_a_valueerror(self):
        """Subclassing ValueError keeps it backward-compatible: any caller that
        still does `except ValueError` treats it as a (silently) dropped command
        rather than letting it crash the batch."""
        self.assertTrue(issubclass(MongoDialectCommandException, ValueError))

    def test_genuinely_unknown_command_still_raises_plain_valueerror(self):
        """A real unknown command must NOT be classified as a Mongo-dialect
        fossil -- it still raises the plain ValueError the step WARNs + logs the
        payload for, so latent command-build bugs keep surfacing."""
        msg = _payload(
            {"type": "mock", "collection": "object", "data": {"field": "value"}}
        )
        with self.assertRaisesRegex(ValueError, "Unrecognized command"):
            command_factory(msg)
        with self.assertRaises(ValueError) as ctx:
            command_factory(msg)
        self.assertNotIsInstance(ctx.exception, MongoDialectCommandException)

    def test_object_update_with_xmatch_still_routes(self):
        """Guards ordering: the xmatch branch must win over the new
        Mongo-dialect catch-all, so an object update carrying xmatch still builds
        an UpsertXmatchCommand rather than being skipped."""
        msg = _payload(
            {
                "type": "update",
                "collection": "object",
                "criteria": {"_id": "ZTF18abdlcao"},
                "data": {"xmatch": {"allwise": {"catoid": "W1", "dist": 0.1}}},
            }
        )
        self.assertIsInstance(command_factory(msg), UpsertXmatchCommand)


if __name__ == "__main__":
    unittest.main()
