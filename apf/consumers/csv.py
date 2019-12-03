from apf.consumers.generic import GenericConsumer

import io
import pandas as pd

class CSVConsumer(GenericConsumer):
    """CSV Consumer.

    Parameters
    ----------
    FILE_PATH: path
        CSV path location

    OTHER_ARGS: dict
        Parameters passed to :func:`pandas.read_csv`

    """
    def __init__(self,config):
        super().__init__(config)
        path = self.config.get("FILE_PATH",None)
        if path is None:
            raise Exception("FILE_PATH variable not set")

    def consume(self):
        df = pd.read_csv(self.config["FILE_PATH"], **self.config.get("OTHER_ARGS",{}))
        self.len = len(df)
        for index, row in df.iterrows():
            yield row.to_dict()
