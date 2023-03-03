import pandas as pd

# Should make this a separate package
from magstats_step.utils.multi_driver.connection import MultiDriverConnection

from typing import List

# Temporal, why?
def get_catalog(
    aids: List[str or int], table: str, driver: MultiDriverConnection
):
    filter_by = {"aid": {"$in": aids}}
    catalog = driver.query(table, engine="psql").find_all(
        filter_by=filter_by, paginate=False
    )
    catalog = pd.DataFrame(catalog)
    catalog.replace({np.nan: None}, inplace=True)
    return catalog

