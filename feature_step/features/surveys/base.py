"""Survey contract and band table shared by every survey package."""


class SurveyBands:
    """Band table of a survey: letters, db codes and feature-name suffixes."""

    def __init__(self, singles, pairs, letters=None):
        """Store single bands, band pairs and the letter table (default: singles)."""
        self.singles = singles  # single band letter -> db code
        self.pairs = pairs  # band pair such as "g,r" -> db code
        self.letters = singles if letters is None else letters  # letter table used by letter()
        self._codes = {**singles, **pairs}  # all codes known to fid_for_db and suffix

    def fid_for_db(self, band):
        """Return the db code of a band; None or unknown bands give 0."""
        return self._codes.get(band, 0)  # unknown or None -> 0

    def suffix(self, band):
        """Return the feature-name suffix "_<code>" of a band; None or unknown give ''."""
        if band in self._codes:  # known band
            return "_" + str(self._codes[band])  # suffix with the db code
        return ""  # None or unknown band

    def letter(self, fid):
        """Return the band letter of a numeric fid, or None if unknown."""
        return self.singles_inverse.get(fid)  # reverse lookup

    @property
    def singles_inverse(self):
        """Return the numeric fid -> band letter dict (today's band_map_inverse)."""
        return {code: letter for letter, code in self.letters.items()}  # invert the letter table


class Survey:
    """Contract every survey implements; attributes live on the subclasses."""

    name: str  # survey name used in SURVEY
    sid: int  # survey id
    tid: int  # type id
    id_column: str = "measurement_id"  # column holding the detection id
    needs_references: bool  # whether references are read from the db
    bands: SurveyBands  # band table of the survey

    def make_preprocessor(self):
        """Return the lightcurve preprocessor of the survey."""
        raise NotImplementedError  # subclass must implement

    def make_extractor(self):
        """Return the feature extractor of the survey."""
        raise NotImplementedError  # subclass must implement

    def epochs_from_message(self, message: dict) -> list:
        """Return the detections of a message as a list of dicts."""
        raise NotImplementedError  # subclass must implement

    def forced_from_message(self, message: dict) -> list:
        """Return the forced photometry of a message as a list of dicts."""
        raise NotImplementedError  # subclass must implement

    def build_astro_object(self, detections, forced, xmatches, references_db):
        """Return the AstroObject built from detections, forced photometry and xmatches."""
        raise NotImplementedError  # subclass must implement

    def scribe_commands(self, astro_objects, features_version, features_group, feature_name_lut):
        """Return the scribe commands for the computed astro objects."""
        raise NotImplementedError  # subclass must implement

    def output_messages(self, astro_objects, messages, candids):
        """Return the output messages for the computed astro objects."""
        raise NotImplementedError  # subclass must implement
