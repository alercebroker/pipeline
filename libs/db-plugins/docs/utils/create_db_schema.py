from db_plugins.db.sql.models import Base
from eralchemy import render_er

render_er(Base, '../source/_static/images/diagram.png')
