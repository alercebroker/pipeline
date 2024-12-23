from typing import TypedDict

import psycopg2

DatabaseConfig = TypedDict(
    "DatabaseConfig",
    {"USER": str, "PASSWORD": str, "HOST": str, "PORT": str, "DB_NAME": str},
)


class PsqlDatabase:
    def __init__(self, config: DatabaseConfig) -> None:
        url = self.__format_db_url(config)
        self.conn = lambda: psycopg2.connect(url)

    def __format_db_url(self, config):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"
