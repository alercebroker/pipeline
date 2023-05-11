import pandas as pd

class NoClassifiedPostProcessor(object):

    class_name = "no_class"

    def __init__(self, messages_dataframe: pd.DataFrame, classifications: pd.DataFrame) -> None:
        self.messages = messages_dataframe
        self.classifications = classifications

    def get_modified_classifications(self) -> pd.DataFrame:
        pass
        # get the aids not classified.
        # difference
        # add a columb to each of messages

        # add a column to messages "no_class"
        # the value is 0 if aid is in "classifications"
        # the value is 1 if aid is not
        pass
