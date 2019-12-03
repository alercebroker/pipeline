from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# create_engine url: dialect+driver://username:password@host:port/database
#engine = create_engine('postgresql+psycopg2://alerce:tiger@localhost/mydatabase', echo=True)
engine = create_engine('sqlite:///:memory:', echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)