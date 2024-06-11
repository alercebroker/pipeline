import psycopg


class PsqlDatabase:
    def __init__(self, config: dict) -> None:
        url = self.__format_db_url(config)
        self.conn = lambda: psycopg.connect(url)

    def __format_db_url(self, config):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"
