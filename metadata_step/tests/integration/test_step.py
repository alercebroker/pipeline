from metadata_step.step import MetadataStep
from metadata_step.utils.database import PSQLConnection
from metadata_step.tests.data.messages import new_message_batch

def _populate_db(connection: PSQLConnection):
    pass

def _create_connection() -> PSQLConnection:
    pass

def test_step(psql_service):
    # go step by step
    db = _create_connection()
    _populate_db(db)
    step = MetadataStep({}, db)

    messages = new_message_batch()
    result = step.execute(messages)
    step.post_execute(result)

    # assert everything went in correctly
    assert True