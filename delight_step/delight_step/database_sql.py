from contextlib import contextmanager
from typing import Callable, ContextManager



from sqlalchemy import create_engine, select, and_
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models import Probability


class PSQLConnection:
    def __init__(self, config: dict, echo=False) -> None:
        url = self.__format_db_url(config)
        self._engine = create_engine(url, echo=echo)
        self._session_factory = sessionmaker(
            self._engine,
        )

    def __format_db_url(self, config):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"

    @contextmanager
    def session(self) -> Callable[..., ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

            
def default_parser(data, **kwargs):
    return data 

def only_unique_oid_parser(data, **kwargs): # cambiar el nombre only unique oid parser
    """
    Toma la data de probabilidades
    Pueden venir varios datos para el mismo oid
    Recomendancion usar set para lista de oids

    Parseamos para obtener solo la lista de oids no repetidos
    [1, 2, 3, 4, 1, 5, 6, 1]
    [1, 2, 3, 4, 5, 6]
    """
    unique_oids = set()
    for d in data:
        unique_oids.add(d[0]) # confirmar el retorno de sql alchemi
    
    return list(unique_oids)

    
def get_sql_probabily(
    oids: list, db_sql: PSQLConnection, probability_filter: dict = None, parser: Callable = default_parser
):
    """
    Documentar la funcion
    Oid lista de oids a buscar
    db_sql sesion de sql alchemi
    caso si filter es none retorna la lista de oids
    filter diccionario con filtros de probabilidad a usar
    """
    if db_sql is None:
        return []
    with db_sql.session() as session:
        if probability_filter is None: # no necesita la session sql, fuera del with
            return oids  # Si no hay filtro, retornar directamente la lista de oids

        stmt = select(Probability).where(
            and_(
                Probability.oid.in_(oids),
                Probability.classifier_name == probability_filter['classifier_name'],
                Probability.class_name == probability_filter['class_name'],
                Probability.probability >= probability_filter['min_probability']
            )
        )

        result = session.execute(stmt).all()
        parsed = parser(result, probability_filter)
        return parsed

