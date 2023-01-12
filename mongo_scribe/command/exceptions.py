class NoDataProvidedException(Exception):
    def __init__(self):
        super().__init__("The command must contain data to insert or update")


class UpdateWithNoCriteriaException(Exception):
    def __init__(self):
        super().__init__("The Update command must contain an update criteria")


class MisformattedCommandExcepction(Exception):
    def __init__(self):
        super().__init__("Received a misformatted message")
