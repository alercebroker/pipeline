from apf.producers.generic import GenericProducer
from pandas.io.json import json_normalize

class CSVProducer(GenericProducer):
    """CSV File Producer.

    Parameters
    ----------
    FILE_PATH: :py:class:`str`
        Output CSV File Path.
    """
    def __init__(self,config):
        super().__init__(config=config)

    def produce(self,message=None):
        """Produce Message to a CSV File.

        Doesn't add the header
        """
        serialized_message = json_normalize(message)
        serialized_message.to_csv(self.config["FILE_PATH"], mode='a+',index=False,header=False)
