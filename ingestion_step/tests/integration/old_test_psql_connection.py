from sorting_hat_step.database import PsqlConnection
from sqlalchemy import text


def test_psql_connection(psql_service):
    conn = PsqlConnection(
        {
            "USER": "postgres",
            "PASSWORD": "postgres",
            "HOST": "localhost",
            "PORT": 5432,
            "DB_NAME": "postgres",
        }
    )

    with conn.session() as session:
        result = session.execute(text("SELECT 1"))
        result = result.fetchone()
    assert result
    assert result[0] == 1
