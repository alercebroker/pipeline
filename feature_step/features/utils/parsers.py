"""Re-export shim: this module defines nothing, it only re-exports names.

The parser, scribe and output functions that used to live here are defined under
features/surveys (common.py, ztf/, lsst/). They are re-exported so that the
tests and scripts importing them from features.utils.parsers keep working; the
other imports keep every name the old module bound importable from it.
"""
import logging
import numpy as np
import pandas as pd
from lc_classifier.features.core.base import AstroObject, query_ao_table
from lc_classifier.utils import mag2flux, mag_err_2_flux_err
from typing import List, Dict, Optional
import os
import json
from features.surveys.common import last_mjd_of, prepare_features_for_db, wise_magnitudes  # shared survey helpers
from features.surveys.lsst import LSSTSurvey  # lsst survey class, name kept importable
from features.surveys.ztf import ZTFSurvey  # ztf survey class, name kept importable
from features.surveys.common import DETECTION_KEYS_MAP  # moved to common, re-exported for lsst and importers
from features.surveys.ztf.astro_object import (  # ztf parser moved, names re-exported for tests and scripts
    add_mag_and_flux_columns,  # ztf magnitude and flux rows
    detections_to_astro_object,  # ztf epochs -> AstroObject
    get_bogus_flags_for_each_detection,  # rb and procstatus per epoch
    get_new_references_from_message,  # references carried in the message
    get_reference_for_each_detection,  # distnr and rfid per epoch
)  # end of the ztf parser imports
from features.surveys.ztf.output import parse_output  # ztf output messages, re-exported
from features.surveys.ztf.scribe import fid_mapper_for_db, parse_scribe_payload  # ztf scribe, re-exported
from features.surveys.ztf.scribe import prepare_ao_features_for_db  # ztf feature rows for the db, body moved to ztf/scribe
from features.surveys.lsst.astro_object import (  # lsst parser moved, names re-exported for tests and scripts
    add_mag_and_flux_lsst,  # lsst magnitude and flux rows
    detections_to_astro_object_lsst,  # lsst sources -> AstroObject
    flux_err_2_mag_err,  # flux error -> magnitude error
    fluxnjy2mag,  # nJy flux -> magnitude
)  # end of the lsst parser imports
from features.surveys.lsst.output import parse_output_lsst  # lsst output messages, re-exported
from features.surveys.lsst.scribe import fid_mapper_for_db_lsst, parse_scribe_payload_lsst  # lsst scribe, re-exported
from features.surveys.lsst.scribe import prepare_ao_features_for_db_lsst  # lsst feature rows for the db, body moved to lsst/scribe
