from dataclasses import dataclass
from random import Random
from typing import Any, Literal

HASH_CONSTANT_MUL = 72348927498  # Random number
HASH_CONSTANT_INV_MUL = pow(HASH_CONSTANT_MUL, -1, 2**63 - 1)
HASH_CONSTANT_XOR = 23894729048  # Random number


def simple_hash(x: int) -> int:
    x = x * HASH_CONSTANT_MUL
    x = x ^ HASH_CONSTANT_XOR
    return (x % (2**63 - 1)) + 1


def inv_simple_hash(x: int) -> int:
    x = x - 1
    x = x ^ HASH_CONSTANT_XOR
    x = x * HASH_CONSTANT_INV_MUL
    return x % (2**63 - 1)


ObjectTypes = Literal["dia", "ss"]


@dataclass
class ObjectInfo:
    oid: int
    otype: ObjectTypes
    obj: dict[str, Any]


class LsstAlertGenerator:
    bands = ("u", "g", "r", "i", "z", "y")

    _id: int = 0
    mjd: float = 60_000.0

    rng: Random
    new_obj_rate: float
    new_prv_source_rate: float
    new_fp_rate: float

    lightcurves: dict[int, list[dict[str, Any]]]
    fps: dict[int, list[dict[str, Any]]]

    prv_object_types: dict[ObjectTypes, list[int]]
    prv_objects: dict[int, dict[str, Any]]

    def __init__(
        self,
        rng: Random | None = None,
        new_obj_rate: float = 0.1,
        new_prv_source_rate: float = 0.1,
        new_fp_rate: float = 0.1,
    ):
        self.rng = rng if rng is not None else Random()
        self.new_obj_rate = new_obj_rate
        self.new_prv_source_rate = new_prv_source_rate
        self.new_fp_rate = new_fp_rate
        self.lightcurves = {}
        self.fps = {}
        self.prv_object_types = {"dia": [], "ss": []}
        self.prv_objects = {}

    def random_band(self):
        return self.rng.choice(self.bands)

    def noneable(self, val: Any, none_rate: float = 0.01):
        return None if self.rng.random() < none_rate else val

    def new_id(self) -> int:
        id = simple_hash(self._id)
        self._id += 1

        return id

    def new_object(self) -> ObjectInfo:
        otype: ObjectTypes = self.rng.choice(["dia", "ss"])
        oid = self.new_id()

        if otype == "dia":
            obj = self.random_dia_object(oid)
        elif otype == "ss":
            obj = self.random_ss_object(oid)

        self.lightcurves[oid] = []
        self.fps[oid] = []
        self.prv_object_types[otype].append(oid)
        self.prv_objects[oid] = obj

        return ObjectInfo(oid, otype, obj)

    def get_object(self) -> ObjectInfo | None:
        object_type: ObjectTypes = self.rng.choice(["dia", "ss"])

        try:
            object_id = self.rng.choice(self.prv_object_types[object_type])
        except IndexError:
            return None

        obj = self.prv_objects[object_id]

        return ObjectInfo(object_id, object_type, obj)

    def generate_alert(self):
        if self.rng.random() < self.new_obj_rate:
            obj_info = self.new_object()
        else:
            obj_info = self.get_object()
            if obj_info is None:
                obj_info = self.new_object()

        mjd = self.mjd
        self.mjd += self.rng.uniform(0.0, 10.0)

        source = self.random_source(obj_info, mjd)

        if self.rng.random() < self.new_prv_source_rate:
            n_new_prv_sources = self.rng.randint(1, 3)
            for _ in range(n_new_prv_sources):
                new_prv_source = self.random_source(
                    obj_info,
                    mjd - self.rng.uniform(-10_000, 0),
                    parent_id=source["diaSourceId"],
                )
                self.lightcurves[obj_info.oid].append(new_prv_source)

        prv_sources = sorted(
            self.lightcurves[obj_info.oid], key=lambda x: x["midpointMjdTai"]
        )[-30:]
        self.lightcurves[obj_info.oid] = prv_sources

        if self.rng.random() < self.new_fp_rate and obj_info.otype == "dia":
            n_new_fp_sources = self.rng.randint(1, 3)
            for _ in range(n_new_fp_sources):
                new_fp_source = self.random_forced_source(
                    obj_info, mjd - self.rng.uniform(-10_000, 0)
                )
                self.fps[obj_info.oid].append(new_fp_source)

        forced_sources = sorted(
            self.fps[obj_info.oid], key=lambda x: x["midpointMjdTai"]
        )[-30:]
        self.fps[obj_info.oid] = forced_sources

        n_non_detections = self.rng.randint(0, 30)
        non_detections = sorted(
            [
                self.random_non_detection(mjd + self.rng.uniform(-10_000, 0))
                for _ in range(n_non_detections)
            ],
            key=lambda x: x["midpointMjdTai"],
        )

        self.lightcurves[obj_info.oid].append(source)

        return {
            "alertId": self.new_id(),
            "diaSource": source,
            "prvDiaSources": self.noneable(prv_sources),
            "prvDiaForcedSources": self.noneable(forced_sources)
            if obj_info.otype == "dia"
            else None,
            "prvDiaNondetectionLimits": self.noneable(non_detections),
            "diaObject": obj_info.obj if obj_info.otype == "dia" else None,
            "ssObject": obj_info.obj if obj_info.otype == "ss" else None,
            "cutoutDifference": None,
            "cutoutScience": None,
            "cutoutTemplate": None,
        }

    def random_source(
        self, obj_info: ObjectInfo, mjd: float, parent_id: int | None = None
    ) -> dict[str, Any]:
        if obj_info.otype == "ss":
            ra = self.rng.uniform(0, 360)
            dec = self.rng.uniform(-90, 90)
        else:
            ra = obj_info.obj["ra"] + self.rng.uniform(-0.01, +0.01)
            dec = obj_info.obj["dec"] + self.rng.uniform(-0.01, +0.01)

        return {
            "diaSourceId": self.new_id(),
            "visit": self.rng.randint(1, 2**63 - 1),
            "detector": self.rng.randint(0, 2**31 - 1),
            "diaObjectId": obj_info.oid if obj_info.otype == "dia" else 0,
            "ssObjectId": obj_info.oid if obj_info.otype == "ss" else 0,
            "parentDiaSourceId": parent_id,
            "midpointMjdTai": mjd,
            "ra": ra,
            "raErr": self.noneable(self.rng.uniform(0, 0.1)),
            "dec": dec,
            "decErr": self.noneable(self.rng.uniform(0, 0.1)),
            "ra_dec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "x": self.rng.uniform(0, 4000),
            "xErr": self.noneable(self.rng.uniform(0, 5)),
            "y": self.rng.uniform(0, 4000),
            "yErr": self.noneable(self.rng.uniform(0, 5)),
            "x_y_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "centroid_flag": self.noneable(self.rng.choice([True, False])),
            "apFlux": self.noneable(self.rng.uniform(-1000, 1000)),
            "apFluxErr": self.noneable(self.rng.uniform(0, 100)),
            "apFlux_flag": self.noneable(self.rng.choice([True, False])),
            "apFlux_flag_apertureTruncated": self.noneable(
                self.rng.choice([True, False])
            ),
            "is_negative": self.noneable(self.rng.choice([True, False])),
            "snr": self.noneable(self.rng.uniform(-10, 100)),
            "psfFlux": self.noneable(self.rng.uniform(-1000, 1000)),
            "psfFluxErr": self.noneable(self.rng.uniform(0, 100)),
            "psfRa": self.noneable(self.rng.uniform(0, 360)),
            "psfRaErr": self.noneable(self.rng.uniform(0, 0.1)),
            "psfDec": self.noneable(self.rng.uniform(-90, 90)),
            "psfDecErr": self.noneable(self.rng.uniform(0, 0.1)),
            "psfFlux_psfRa_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "psfFlux_psfDec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "psfRa_psfDec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "psfLnL": self.noneable(self.rng.uniform(-1000, 1000)),
            "psfChi2": self.noneable(self.rng.uniform(0, 1000)),
            "psfNdata": self.noneable(self.rng.randint(0, 1000)),
            "psfFlux_flag": self.noneable(self.rng.choice([True, False])),
            "psfFlux_flag_edge": self.noneable(self.rng.choice([True, False])),
            "psfFlux_flag_noGoodPixels": self.noneable(self.rng.choice([True, False])),
            "trailFlux": self.noneable(self.rng.uniform(-1000, 1000)),
            "trailFluxErr": self.noneable(self.rng.uniform(0, 100)),
            "trailRa": self.noneable(self.rng.uniform(0, 360)),
            "trailRaErr": self.noneable(self.rng.uniform(0, 0.1)),
            "trailDec": self.noneable(self.rng.uniform(-90, 90)),
            "trailDecErr": self.noneable(self.rng.uniform(0, 0.1)),
            "trailLength": self.noneable(self.rng.uniform(0, 100)),
            "trailLengthErr": self.noneable(self.rng.uniform(0, 10)),
            "trailAngle": self.noneable(self.rng.uniform(0, 360)),
            "trailAngleErr": self.noneable(self.rng.uniform(0, 10)),
            "trailFlux_trailRa_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailFlux_trailDec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailFlux_trailLength_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailFlux_trailAngle_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailRa_trailDec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailRa_trailLength_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailRa_trailAngle_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailDec_trailLength_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailDec_trailAngle_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailLength_trailAngle_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "trailLnL": self.noneable(self.rng.uniform(-1000, 1000)),
            "trailChi2": self.noneable(self.rng.uniform(0, 1000)),
            "trailNdata": self.noneable(self.rng.randint(0, 1000)),
            "trail_flag_edge": self.noneable(self.rng.choice([True, False])),
            "dipoleMeanFlux": self.noneable(self.rng.uniform(-1000, 1000)),
            "dipoleMeanFluxErr": self.noneable(self.rng.uniform(0, 100)),
            "dipoleFluxDiff": self.noneable(self.rng.uniform(-1000, 1000)),
            "dipoleFluxDiffErr": self.noneable(self.rng.uniform(0, 100)),
            "dipoleRa": self.noneable(self.rng.uniform(0, 360)),
            "dipoleRaErr": self.noneable(self.rng.uniform(0, 0.1)),
            "dipoleDec": self.noneable(self.rng.uniform(-90, 90)),
            "dipoleDecErr": self.noneable(self.rng.uniform(0, 0.1)),
            "dipoleLength": self.noneable(self.rng.uniform(0, 100)),
            "dipoleLengthErr": self.noneable(self.rng.uniform(0, 10)),
            "dipoleAngle": self.noneable(self.rng.uniform(0, 360)),
            "dipoleAngleErr": self.noneable(self.rng.uniform(0, 10)),
            "dipoleMeanFlux_dipoleFluxDiff_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleMeanFlux_dipoleRa_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "dipoleMeanFlux_dipoleDec_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleMeanFlux_dipoleLength_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleMeanFlux_dipoleAngle_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleFluxDiff_dipoleRa_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "dipoleFluxDiff_dipoleDec_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleFluxDiff_dipoleLength_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleFluxDiff_dipoleAngle_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleRa_dipoleDec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "dipoleRa_dipoleLength_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "dipoleRa_dipoleAngle_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "dipoleDec_dipoleLength_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "dipoleDec_dipoleAngle_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "dipoleLength_dipoleAngle_Cov": self.noneable(
                self.rng.uniform(-0.01, 0.01)
            ),
            "dipoleLnL": self.noneable(self.rng.uniform(-1000, 1000)),
            "dipoleChi2": self.noneable(self.rng.uniform(0, 1000)),
            "dipoleNdata": self.noneable(self.rng.randint(0, 1000)),
            "forced_PsfFlux_flag": self.noneable(self.rng.choice([True, False])),
            "forced_PsfFlux_flag_edge": self.noneable(self.rng.choice([True, False])),
            "forced_PsfFlux_flag_noGoodPixels": self.noneable(
                self.rng.choice([True, False])
            ),
            "snapDiffFlux": self.noneable(self.rng.uniform(-1000, 1000)),
            "snapDiffFluxErr": self.noneable(self.rng.uniform(0, 100)),
            "fpBkgd": self.noneable(self.rng.uniform(-100, 100)),
            "fpBkgdErr": self.noneable(self.rng.uniform(0, 10)),
            "ixx": self.noneable(self.rng.uniform(0, 100)),
            "ixxErr": self.noneable(self.rng.uniform(0, 10)),
            "iyy": self.noneable(self.rng.uniform(0, 100)),
            "iyyErr": self.noneable(self.rng.uniform(0, 10)),
            "ixy": self.noneable(self.rng.uniform(-100, 100)),
            "ixyErr": self.noneable(self.rng.uniform(0, 10)),
            "ixx_iyy_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "ixx_ixy_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "iyy_ixy_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "ixxPSF": self.noneable(self.rng.uniform(0, 100)),
            "iyyPSF": self.noneable(self.rng.uniform(0, 100)),
            "ixyPSF": self.noneable(self.rng.uniform(-100, 100)),
            "shape_flag": self.noneable(self.rng.choice([True, False])),
            "shape_flag_no_pixels": self.noneable(self.rng.choice([True, False])),
            "shape_flag_not_contained": self.noneable(self.rng.choice([True, False])),
            "shape_flag_parent_source": self.noneable(self.rng.choice([True, False])),
            "extendedness": self.noneable(self.rng.uniform(0, 1)),
            "reliability": self.noneable(self.rng.uniform(0, 1)),
            "band": self.noneable(self.random_band()),
            "dipoleFitAttempted": self.noneable(self.rng.choice([True, False])),
            "pixelFlags": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_bad": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_cr": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_crCenter": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_edge": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_nodata": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_nodataCenter": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_interpolated": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_interpolatedCenter": self.noneable(
                self.rng.choice([True, False])
            ),
            "pixelFlags_offimage": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_saturated": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_saturatedCenter": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_suspect": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_suspectCenter": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_streak": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_streakCenter": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_injected": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_injectedCenter": self.noneable(self.rng.choice([True, False])),
            "pixelFlags_injected_template": self.noneable(
                self.rng.choice([True, False])
            ),
            "pixelFlags_injected_templateCenter": self.noneable(
                self.rng.choice([True, False])
            ),
        }

    def random_forced_source(self, obj_info: ObjectInfo, mjd: float) -> dict[str, Any]:
        return {
            "diaForcedSourceId": self.new_id(),
            "diaObjectId": obj_info.oid,
            "ra": obj_info.obj["ra"] + self.rng.uniform(-0.01, +0.01),
            "dec": obj_info.obj["dec"] + self.rng.uniform(-0.01, +0.01),
            "visit": self.rng.randint(1, 2**63 - 1),
            "detector": self.rng.randint(0, 2**31 - 1),
            "psfFlux": self.noneable(self.rng.uniform(-1000, 1000)),
            "psfFluxErr": self.noneable(self.rng.uniform(0, 100)),
            "midpointMjdTai": mjd,
            "band": self.noneable(self.random_band()),
        }

    def random_non_detection(self, mjd: float) -> dict[str, Any]:
        return {
            "ccdVisitId": self.rng.randint(1, 2**63 - 1),
            "midpointMjdTai": mjd,
            "band": self.random_band(),
            "diaNoise": self.rng.uniform(0, 100),
        }

    def random_dia_object(self, object_id: int) -> dict[str, Any]:
        obj = {
            "diaObjectId": object_id,
            "ra": self.rng.uniform(0, 360),
            "raErr": self.noneable(self.rng.uniform(0, 0.1)),
            "dec": self.rng.uniform(-90, 90),
            "decErr": self.noneable(self.rng.uniform(0, 0.1)),
            "ra_dec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "radecMjdTai": self.noneable(self.rng.uniform(50000, 70000)),
            "pmRa": self.noneable(self.rng.uniform(-100, 100)),
            "pmRaErr": self.noneable(self.rng.uniform(0, 10)),
            "pmDec": self.noneable(self.rng.uniform(-100, 100)),
            "pmDecErr": self.noneable(self.rng.uniform(0, 10)),
            "parallax": self.noneable(self.rng.uniform(-10, 10)),
            "parallaxErr": self.noneable(self.rng.uniform(0, 10)),
            "pmRa_pmDec_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "pmRa_parallax_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "pmDec_parallax_Cov": self.noneable(self.rng.uniform(-0.01, 0.01)),
            "pmParallaxLnL": self.noneable(self.rng.uniform(-1000, 1000)),
            "pmParallaxChi2": self.noneable(self.rng.uniform(0, 1000)),
            "pmParallaxNdata": self.noneable(self.rng.randint(0, 1000)),
            "nearbyObj1": self.noneable(self.rng.randint(1, 2**63 - 1)),
            "nearbyObj1Dist": self.noneable(self.rng.uniform(0, 1000)),
            "nearbyObj1LnP": self.noneable(self.rng.uniform(-100, 0)),
            "nearbyObj2": self.noneable(self.rng.randint(1, 2**63 - 1)),
            "nearbyObj2Dist": self.noneable(self.rng.uniform(0, 1000)),
            "nearbyObj2LnP": self.noneable(self.rng.uniform(-100, 0)),
            "nearbyObj3": self.noneable(self.rng.randint(1, 2**63 - 1)),
            "nearbyObj3Dist": self.noneable(self.rng.uniform(0, 1000)),
            "nearbyObj3LnP": self.noneable(self.rng.uniform(-100, 0)),
        }
        for band in self.bands:
            obj[f"{band}_psfFluxMean"] = self.noneable(self.rng.uniform(-1000, 1000))
            obj[f"{band}_psfFluxMeanErr"] = self.noneable(self.rng.uniform(0, 100))
            obj[f"{band}_psfFluxSigma"] = self.noneable(self.rng.uniform(0, 100))
            obj[f"{band}_psfFluxChi2"] = self.noneable(self.rng.uniform(0, 1000))
            obj[f"{band}_psfFluxNdata"] = self.noneable(self.rng.randint(0, 1000))
            obj[f"{band}_fpFluxMean"] = self.noneable(self.rng.uniform(-1000, 1000))
            obj[f"{band}_fpFluxMeanErr"] = self.noneable(self.rng.uniform(0, 100))
            obj[f"{band}_fpFluxSigma"] = self.noneable(self.rng.uniform(0, 100))
            obj[f"{band}_psfFluxErrMean"] = self.noneable(self.rng.uniform(0, 100))
        return obj

    def random_ss_object(self, object_id: int) -> dict[str, Any]:
        obj = {
            "ssObjectId": object_id,
            "discoverySubmissionDate": self.noneable(self.rng.uniform(50000, 70000)),
            "firstObservationDate": self.noneable(self.rng.uniform(50000, 70000)),
            "arc": self.noneable(self.rng.uniform(0, 10000)),
            "numObs": self.noneable(self.rng.randint(0, 1000)),
            "MOID": self.noneable(self.rng.uniform(0, 10)),
            "MOIDTrueAnomaly": self.noneable(self.rng.uniform(0, 360)),
            "MOIDEclipticLongitude": self.noneable(self.rng.uniform(0, 360)),
            "MOIDDeltaV": self.noneable(self.rng.uniform(0, 100)),
            "medianExtendedness": self.noneable(self.rng.uniform(0, 1)),
        }
        for band in self.bands:
            obj[f"{band}_H"] = self.noneable(self.rng.uniform(10, 30))
            obj[f"{band}_G12"] = self.noneable(self.rng.uniform(-0.5, 0.8))
            obj[f"{band}_HErr"] = self.noneable(self.rng.uniform(0, 1))
            obj[f"{band}_G12Err"] = self.noneable(self.rng.uniform(0, 0.5))
            obj[f"{band}_H_{band}_G12_Cov"] = self.noneable(self.rng.uniform(-0.1, 0.1))
            obj[f"{band}_Chi2"] = self.noneable(self.rng.uniform(0, 1000))
            obj[f"{band}_Ndata"] = self.noneable(self.rng.randint(0, 1000))
        return obj
