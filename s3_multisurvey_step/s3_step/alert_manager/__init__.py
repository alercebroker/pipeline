from .lsst import LSSTAlertManager
from .ztf import ZTFAlertManager

"""
idea for the config
config.yaml
    survet_id: lsst
    consumer_config:
        ----
    bucket_config:
        region: us-east-1
        bucket_name:

"""


def manager_selector(survey_id: str):
    """
    Selects the appropriate alert manager based on the survey ID.
    
    Parameters
    ----------
    survey_id : str
        The ID of the survey (e.g., 'lsst', 'ztf').
    
    Returns
    -------
    BaseAlertManager
        An instance of the appropriate alert manager.
    """
    if survey_id == LSSTAlertManager.survey_id:
        return LSSTAlertManager
    elif survey_id == ZTFAlertManager.survey_id:
        return ZTFAlertManager
    else:
        raise ValueError(f"Unsupported survey ID: {survey_id}")