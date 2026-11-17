"""lsst band table, in its own module so the lsst submodules can use it without importing LSSTSurvey."""
from features.surveys.base import SurveyBands  # band table type


class LSSTBands(SurveyBands):
    """lsst band table whose db code lookup is the multisurvey fid_mapper_for_db_lsst.

    fid_for_db turns the band into a stripped string before the lookup, so " g "
    gives 1 and a non-string band gives 0 without raising. suffix and letter stay
    the plain table lookups: the multisurvey output adds no suffix to a padded band.
    """

    def fid_for_db(self, band):
        """Return the db code of a band after str() and strip(); None or unknown bands give 0."""
        if band is None:  # missing band
            return 0 #reemplazamos valor nulo por 0
        band = str(band).strip()  # band as a stripped string
        return super().fid_for_db(band)  # codes of the lsst band table, anything else 0


LSST_BANDS = LSSTBands(  # band table of lsst
    {"u": 6, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5},  # single bands
    {"u,g": 61, "g,r": 12, "r,i": 23, "i,z": 34, "z,y": 45},  # band pairs
)  # end of band table
