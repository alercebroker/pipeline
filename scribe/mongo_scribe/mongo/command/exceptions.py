class NoDataProvidedException(ValueError):
    """
    Exception to raise when the command doesn't have any data
    """

    def __init__(self):
        super().__init__("The command must contain data to insert or update")


class UpdateWithNoCriteriaException(ValueError):
    """
    Exception to raise when an update command doesn't have a filter (or criteria)
    """

    def __init__(self):
        super().__init__("The Update command must contain an update criteria")


class WrongFormatCommandException(ValueError):
    """
    Exception to raise when a dictionary doesn't have valid command fields
    """

    def __init__(self):
        super().__init__("Received a badly formatted message")


class NonExistentCollectionException(ValueError):
    """
    Exception to raise when trying to obtain a non-existent collection
    """

    def __init__(self, collection: str):
        super().__init__(f"Collection {collection} doesn't exist")


class NoCollectionProvidedException(ValueError):
    """
    Exception to raise when a command doesn't provide a collection to write on
    """

    def __init__(self):
        super().__init__("No collection provided in the command")


class NoClassifierInfoProvidedException(ValueError):
    """
    Exception to raise when a command doesn't provide the classifier when
    updating probabilities
    """

    def __init__(self):
        super().__init__("No classifier info provided in the command")


class NoAlerceIdentificationProvidedException(ValueError):
    """
    Exception to raise when a command does not provide an ALeRCE ID when
    trying to update probabilities
    """

    def __init__(self):
        super().__init__("No AID provided in the command")


class NoFeatureProvidedException(ValueError):
    """
    Exception to raise when a command doesn't provide the features list
    when updating features
    """

    def __init__(self):
        super().__init__("No features provided in the command")


class NoFeatureVersionProvidedException(ValueError):
    """
    Exception to raise when a command doesn't provide the feature_version
    when updating features
    """

    def __init__(self):
        super().__init__("No features_version provided in the command")


class NoFeatureGroupProvidedException(ValueError):
    """
    Exception to raise when a command doesn't provide the feature_group
    when updating features
    """

    def __init__(self):
        super().__init__("No features_group provided in the command")
