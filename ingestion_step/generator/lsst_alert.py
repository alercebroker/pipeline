import pickle
from dataclasses import dataclass
from random import Random
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# Constants for simple_hash/inv_simple_hash, used for generating unique IDs.
HASH_CONSTANT_MUL = 72348927498  # Random number
HASH_CONSTANT_INV_MUL = pow(HASH_CONSTANT_MUL, -1, 2**63 - 1)
HASH_CONSTANT_XOR = 23894729048  # Random number

DEFAULT_ERR = 0.01  # Used as fallback for missing error values.


def simple_hash(x: int) -> int:
    """Generates a pseudo-unique positive integer for IDs."""
    x = x * HASH_CONSTANT_MUL
    x = x ^ HASH_CONSTANT_XOR
    return (x % (2**63 - 1)) + 1


def inv_simple_hash(x: int) -> int:
    """Inverse of simple_hash, for debugging or mapping back."""
    x = x - 1
    x = x ^ HASH_CONSTANT_XOR
    x = x * HASH_CONSTANT_INV_MUL
    return x % (2**63 - 1)


ObjectTypes = Literal["dia", "ss"]


class ObjectStats:
    """Tracks stadistics for each object"""

    oid: int
    sid: int
    tid: int
    n_det: int
    n_fphot: int
    ras: list[float]
    decs: list[float]
    e_ras: list[float | None]
    e_decs: list[float | None]
    first_mjd: float
    last_mjd: float
    binary_data: Any

    def __init__(self, oid: int, sid: int):
        self.oid = oid
        self.sid = sid
        self.tid = 0
        self.n_det = 0
        self.n_fphot = 0
        self.ras = []
        self.decs = []
        self.e_ras = []
        self.e_decs = []
        self.first_mjd = float("inf")
        self.last_mjd = -float("inf")

    def to_objstats_dict(self) -> dict[str, Any]:
        """Returns a dict format, similar to how it is stored in the DB"""
        sigmas_ra = self._sigmas(self.e_ras)
        sigmas_dec = self._sigmas(self.e_decs)
        return {
            "oid": self.oid,
            "sid": self.sid,
            "tid": self.tid,
            "n_det": self.n_det,
            "n_forced": self.n_fphot,
            "meanra": np.average(self.ras, weights=sigmas_ra),
            "meandec": np.average(self.decs, weights=sigmas_dec),
            "sigmara": self._sigma(sigmas_ra),
            "sigmadec": self._sigma(sigmas_dec),
            "firstmjd": self.first_mjd,
            "lastmjd": self.last_mjd,
            "deltamjd": self.last_mjd - self.first_mjd,
        }

    @staticmethod
    def _sigmas(errors: list[float | None]):
        processed_errors = []
        for error in errors:
            if error is not None and not np.isnan(error):
                processed_errors.append(float(error))
            else:
                processed_errors.append(DEFAULT_ERR)

        sigmas = np.array(processed_errors, dtype=np.float64) / 3600.0  # Arcsec to deg
        return sigmas**-2

    @staticmethod
    def _sigma(sigmas: NDArray[np.floating[Any]]) -> float:
        sigma = np.sqrt(1 / np.sum(sigmas))
        sigma = sigma * 3600.0  # Deg to arcsec

        return sigma


@dataclass
class ObjectInfo:
    oid: int
    otype: ObjectTypes
    obj: dict[str, Any]


class LsstAlertGenerator:
    """Generator for LSST like alerts"""

    bands = ("u", "g", "r", "i", "z", "y")
    binary_data: Any

    _id: int = 0
    mjd: float = 60_000.0

    rng: Random
    new_obj_rate: float
    new_prv_source_rate: float
    new_fp_rate: float

    lightcurves: dict[int, list[dict[str, Any]]]  # oid -> list[sources]
    fps: dict[int, list[dict[str, Any]]]  # oid -> list[forced_sources]

    prv_object_types: dict[ObjectTypes, list[int]]  # dia|ss -> list[dia|ss]
    prv_objects: dict[int, dict[str, Any]]  # oid -> object

    objstats: dict[int, ObjectStats]  # oid -> objstats

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
        self.objstats = {}
        with open("./generator/binary_cutouts.pkl", "rb") as f:
            binary_data = pickle.load(f)
        self.binary_data = binary_data

    def get_objstats(self, oid: int) -> ObjectStats:
        return self.objstats[oid]

    def get_all_objstats_dicts(self) -> list[dict[str, Any]]:
        """Return al objstats as a list of dicts"""
        return [objstat.to_objstats_dict() for objstat in self.objstats.values()]

    def generate_alert(self):
        """
        Generates alerts matching the Lsst schema.
        Keeps lightcurves and objstats updated.
        """
        if self.rng.random() < self.new_obj_rate:
            obj_info = self._new_object()
        else:
            obj_info = self._get_object()
            if obj_info is None:
                obj_info = self._new_object()

        mjd = self.mjd
        self.mjd += self.rng.uniform(0.5, 10.0)

        source = self._random_dia_source(obj_info, mjd)

        objstats = self.objstats[obj_info.oid]

        objstats.n_det += 1
        objstats.ras.append(source["ra"])
        objstats.decs.append(source["dec"])
        objstats.e_ras.append(source["raErr"])
        objstats.e_decs.append(source["decErr"])
        objstats.first_mjd = min(objstats.first_mjd, source["midpointMjdTai"])
        objstats.last_mjd = max(objstats.last_mjd, source["midpointMjdTai"])

        if self.rng.random() < self.new_prv_source_rate:
            n_new_prv_sources = self.rng.randint(1, 3)
            for _ in range(n_new_prv_sources):
                base_mjd = (
                    self.lightcurves[obj_info.oid][-5]["midpointMjdTai"]
                    if len(self.lightcurves[obj_info.oid]) > 5
                    else mjd - 100.0
                )
                new_prv_source = self._random_dia_source(
                    obj_info,
                    self.rng.uniform(base_mjd, mjd),
                    parent_id=source["diaSourceId"],
                )
                self.lightcurves[obj_info.oid].append(new_prv_source)
                objstats.n_det += 1
                objstats.ras.append(new_prv_source["ra"])
                objstats.decs.append(new_prv_source["dec"])
                objstats.e_ras.append(new_prv_source["raErr"])
                objstats.e_decs.append(new_prv_source["decErr"])
                objstats.first_mjd = min(
                    objstats.first_mjd, new_prv_source["midpointMjdTai"]
                )
                objstats.last_mjd = max(
                    objstats.last_mjd, new_prv_source["midpointMjdTai"]
                )

        prv_sources = sorted(
            self.lightcurves[obj_info.oid], key=lambda x: x["midpointMjdTai"]
        )[-30:]
        self.lightcurves[obj_info.oid] = prv_sources

        if self.rng.random() < self.new_fp_rate and obj_info.otype == "dia":
            n_new_fp_sources = self.rng.randint(1, 3)
            for _ in range(n_new_fp_sources):
                base_mjd = (
                    self.fps[obj_info.oid][-5]["midpointMjdTai"]
                    if len(self.fps[obj_info.oid]) > 5
                    else mjd - 100.0
                )
                new_fp_source = self._random_forced_source(
                    obj_info,
                    self.rng.uniform(base_mjd, mjd),
                )
                self.fps[obj_info.oid].append(new_fp_source)
                objstats.n_fphot += 1

        forced_sources = sorted(
            self.fps[obj_info.oid], key=lambda x: x["midpointMjdTai"]
        )[-30:]
        self.fps[obj_info.oid] = forced_sources

        self.lightcurves[obj_info.oid].append(source)

        self.objstats[obj_info.oid] = objstats

        alert = {
            "diaSourceId": source["diaSourceId"],
            "observation_reason": self._noneable("TEST OBSERVATION REASON"),
            "target_name": self._noneable("TEST TARGET NAME"),
            "diaSource": source,
            "prvDiaSources": prv_sources[:-1] if len(prv_sources) > 1 else None,
            "prvDiaForcedSources": forced_sources
            if obj_info.otype == "dia" or len(forced_sources) > 0
            else None,
            "diaObject": obj_info.obj if obj_info.otype == "dia" else None,
            "ssSource": self._random_ss_source(
                obj_info, dia_source_id=source["diaSourceId"]
            )
            if obj_info.otype == "ss"
            else None,
            # "ssObject": obj_info.obj if obj_info.otype == "ss" else None,
            "mpc_orbits": self._random_mpc_orbit() if obj_info.otype == "ss" else None,
            "cutoutDifference": None,
            "cutoutScience": None,
            "cutoutTemplate": None,
        }

        alert.update(self.binary_data)
        return alert

    def _random_band(self):
        return self.rng.choice(self.bands)

    def _noneable(self, val: Any, none_rate: float = 0.01):
        return None if self.rng.random() < none_rate else val

    def _new_id(self) -> int:
        id = simple_hash(self._id)
        self._id += 1

        return id

    def _new_object(self) -> ObjectInfo:
        otype: ObjectTypes = self.rng.choice(["dia", "ss"])
        oid = self._new_id()

        if otype == "dia":
            obj = self._random_dia_object(oid)
        elif otype == "ss":
            obj = self._random_ss_object(oid)

        self.lightcurves[oid] = []
        self.fps[oid] = []
        self.prv_object_types[otype].append(oid)
        self.prv_objects[oid] = obj
        self.objstats[oid] = ObjectStats(oid=oid, sid=1 if otype == "dia" else 2)

        return ObjectInfo(oid, otype, obj)

    def _get_object(self) -> ObjectInfo | None:
        object_type: ObjectTypes = self.rng.choice(["dia", "ss"])

        try:
            object_id = self.rng.choice(self.prv_object_types[object_type])
        except IndexError:
            return None

        obj = self.prv_objects[object_id]

        return ObjectInfo(object_id, object_type, obj)

    def _random_dia_source(
        self, obj_info: ObjectInfo, mjd: float, parent_id: int | None = None
    ) -> dict[str, Any]:
        return {
            "diaSourceId": self._new_id(),
            "visit": self.rng.randint(1, 2**63 - 1),
            "detector": self.rng.randint(0, 2**31 - 1),
            "diaObjectId": obj_info.oid if obj_info.otype == "dia" else None,
            "ssObjectId": obj_info.oid if obj_info.otype == "ss" else None,
            "parentDiaSourceId": parent_id,
            "midpointMjdTai": mjd,
            "ra": obj_info.obj["ra"] + self.rng.uniform(-0.01, +0.01)
            if obj_info.otype == "dia"
            else self.rng.uniform(0, 360),
            "raErr": self._noneable(self.rng.uniform(0, 0.1)),
            "dec": obj_info.obj["dec"] + self.rng.uniform(-0.01, +0.01)
            if obj_info.otype == "dia"
            else self.rng.uniform(-90, 90),
            "decErr": self._noneable(self.rng.uniform(0, 0.1)),
            "ra_dec_Cov": self._noneable(self.rng.uniform(-0.01, 0.01)),
            "x": self.rng.uniform(0, 4000),
            "xErr": self._noneable(self.rng.uniform(0, 5)),
            "y": self.rng.uniform(0, 4000),
            "yErr": self._noneable(self.rng.uniform(0, 5)),
            "centroid_flag": self._noneable(self.rng.choice([True, False])),
            "apFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "apFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "apFlux_flag": self._noneable(self.rng.choice([True, False])),
            "apFlux_flag_apertureTruncated": self._noneable(
                self.rng.choice([True, False])
            ),
            "isNegative": self._noneable(self.rng.choice([True, False])),
            "snr": self._noneable(self.rng.uniform(-10, 100)),
            "psfFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "psfFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "psfLnL": self._noneable(self.rng.uniform(-1000, 1000)),
            "psfChi2": self._noneable(self.rng.uniform(0, 1000)),
            "psfNdata": self._noneable(self.rng.randint(0, 1000)),
            "psfFlux_flag": self._noneable(self.rng.choice([True, False])),
            "psfFlux_flag_edge": self._noneable(self.rng.choice([True, False])),
            "psfFlux_flag_noGoodPixels": self._noneable(self.rng.choice([True, False])),
            "trailFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "trailFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "trailRa": self._noneable(self.rng.uniform(0, 360)),
            "trailRaErr": self._noneable(self.rng.uniform(0, 0.1)),
            "trailDec": self._noneable(self.rng.uniform(-90, 90)),
            "trailDecErr": self._noneable(self.rng.uniform(0, 0.1)),
            "trailLength": self._noneable(self.rng.uniform(0, 100)),
            "trailLengthErr": self._noneable(self.rng.uniform(0, 10)),
            "trailAngle": self._noneable(self.rng.uniform(0, 360)),
            "trailAngleErr": self._noneable(self.rng.uniform(0, 10)),
            "trailChi2": self._noneable(self.rng.uniform(0, 1000)),
            "trailNdata": self._noneable(self.rng.randint(0, 1000)),
            "trail_flag_edge": self._noneable(self.rng.choice([True, False])),
            "dipoleMeanFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "dipoleMeanFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "dipoleFluxDiff": self._noneable(self.rng.uniform(-1000, 1000)),
            "dipoleFluxDiffErr": self._noneable(self.rng.uniform(0, 100)),
            "dipoleLength": self._noneable(self.rng.uniform(0, 100)),
            "dipoleAngle": self._noneable(self.rng.uniform(0, 360)),
            "dipoleChi2": self._noneable(self.rng.uniform(0, 1000)),
            "dipoleNdata": self._noneable(self.rng.randint(0, 1000)),
            "scienceFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "scienceFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "forced_PsfFlux_flag": self._noneable(self.rng.choice([True, False])),
            "forced_PsfFlux_flag_edge": self._noneable(self.rng.choice([True, False])),
            "forced_PsfFlux_flag_noGoodPixels": self._noneable(
                self.rng.choice([True, False])
            ),
            "templateFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "templateFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "ixx": self._noneable(self.rng.uniform(0, 100)),
            "iyy": self._noneable(self.rng.uniform(0, 100)),
            "ixy": self._noneable(self.rng.uniform(-100, 100)),
            "ixxPSF": self._noneable(self.rng.uniform(0, 100)),
            "iyyPSF": self._noneable(self.rng.uniform(0, 100)),
            "ixyPSF": self._noneable(self.rng.uniform(-100, 100)),
            "shape_flag": self._noneable(self.rng.choice([True, False])),
            "shape_flag_no_pixels": self._noneable(self.rng.choice([True, False])),
            "shape_flag_not_contained": self._noneable(self.rng.choice([True, False])),
            "shape_flag_parent_source": self._noneable(self.rng.choice([True, False])),
            "extendedness": self._noneable(self.rng.uniform(0, 1)),
            "reliability": self._noneable(self.rng.uniform(0, 1)),
            "band": self._noneable(self._random_band()),
            "isDipole": self._noneable(self.rng.choice([True, False])),
            "dipoleFitAttempted": self._noneable(self.rng.choice([True, False])),
            "timeProcessedMjdTai": self.rng.uniform(50000, 70000),  # realistic MJD TAI
            "timeWithdrawnMjdTai": None,  # always present, default None
            "bboxSize": self._noneable(self.rng.randint(10, 100)),
            "pixelFlags": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_bad": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_cr": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_crCenter": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_edge": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_nodata": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_nodataCenter": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_interpolated": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_interpolatedCenter": self._noneable(
                self.rng.choice([True, False])
            ),
            "pixelFlags_offimage": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_saturated": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_saturatedCenter": self._noneable(
                self.rng.choice([True, False])
            ),
            "pixelFlags_suspect": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_suspectCenter": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_streak": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_streakCenter": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_injected": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_injectedCenter": self._noneable(self.rng.choice([True, False])),
            "pixelFlags_injected_template": self._noneable(
                self.rng.choice([True, False])
            ),
            "pixelFlags_injected_templateCenter": self._noneable(
                self.rng.choice([True, False])
            ),
            "glint_trail": self._noneable(self.rng.choice([True, False])),
        }

    def _random_ss_source(
        self, obj_info: ObjectInfo, dia_source_id: int
    ) -> dict[str, Any]:
        return {
            "diaSourceId": dia_source_id,
            "ssObjectId": obj_info.oid,
            "designation": self._noneable(f"DES{self.rng.randint(10000, 99999)}"),
            "eclLambda": self.rng.uniform(0, 360),
            "eclBeta": self.rng.uniform(-90, 90),
            "galLon": self.rng.uniform(0, 360),
            "galLat": self.rng.uniform(-90, 90),
            "elongation": self._noneable(self.rng.uniform(0, 180)),
            "phaseAngle": self._noneable(self.rng.uniform(0, 180)),
            "topoRange": self._noneable(self.rng.uniform(0, 50)),
            "topoRangeRate": self._noneable(self.rng.uniform(-10, 10)),
            "helioRange": self._noneable(self.rng.uniform(0, 50)),
            "helioRangeRate": self._noneable(self.rng.uniform(-10, 10)),
            "ephRa": self._noneable(self.rng.uniform(0, 360)),
            "ephDec": self._noneable(self.rng.uniform(-90, 90)),
            "ephVmag": self._noneable(self.rng.uniform(10, 30)),
            "ephRate": self._noneable(self.rng.uniform(0, 10)),
            "ephRateRa": self._noneable(self.rng.uniform(-5, 5)),
            "ephRateDec": self._noneable(self.rng.uniform(-5, 5)),
            "ephOffset": self._noneable(self.rng.uniform(0, 1)),
            "ephOffsetRa": self._noneable(self.rng.uniform(-0.1, 0.1)),
            "ephOffsetDec": self._noneable(self.rng.uniform(-0.1, 0.1)),
            "ephOffsetAlongTrack": self._noneable(self.rng.uniform(-0.1, 0.1)),
            "ephOffsetCrossTrack": self._noneable(self.rng.uniform(-0.1, 0.1)),
            "helio_x": self._noneable(self.rng.uniform(-50, 50)),
            "helio_y": self._noneable(self.rng.uniform(-50, 50)),
            "helio_z": self._noneable(self.rng.uniform(-50, 50)),
            "helio_vx": self._noneable(self.rng.uniform(-5, 5)),
            "helio_vy": self._noneable(self.rng.uniform(-5, 5)),
            "helio_vz": self._noneable(self.rng.uniform(-5, 5)),
            "helio_vtot": self._noneable(self.rng.uniform(0, 10)),
            "topo_x": self._noneable(self.rng.uniform(-50, 50)),
            "topo_y": self._noneable(self.rng.uniform(-50, 50)),
            "topo_z": self._noneable(self.rng.uniform(-50, 50)),
            "topo_vx": self._noneable(self.rng.uniform(-5, 5)),
            "topo_vy": self._noneable(self.rng.uniform(-5, 5)),
            "topo_vz": self._noneable(self.rng.uniform(-5, 5)),
            "topo_vtot": self._noneable(self.rng.uniform(0, 10)),
            "diaDistanceRank": self._noneable(self.rng.randint(1, 5)),
        }

    def _random_forced_source(self, obj_info: ObjectInfo, mjd: float) -> dict[str, Any]:
        return {
            "diaForcedSourceId": self._new_id(),
            "diaObjectId": obj_info.oid,
            "ra": obj_info.obj["ra"] + self.rng.uniform(-0.01, +0.01),
            "dec": obj_info.obj["dec"] + self.rng.uniform(-0.01, +0.01),
            "visit": self.rng.randint(1, 2**63 - 1),
            "detector": self.rng.randint(0, 2**31 - 1),
            "psfFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "psfFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "midpointMjdTai": mjd,
            "scienceFlux": self._noneable(self.rng.uniform(-1000, 1000)),
            "scienceFluxErr": self._noneable(self.rng.uniform(0, 100)),
            "band": self._noneable(self._random_band()),
            "timeProcessedMjdTai": self.rng.uniform(50000, 70000),
            "timeWithdrawnMjdTai": None,
        }

    # def _random_non_detection(self, mjd: float) -> dict[str, Any]:
    #     return {
    #         "ccdVisitId": self.rng.randint(1, 2**63 - 1),
    #         "midpointMjdTai": mjd,
    #         "band": self._random_band(),
    #         "diaNoise": self.rng.uniform(0, 100),
    #     }

    def _random_dia_object(self, object_id: int) -> dict[str, Any]:
        obj = {
            "diaObjectId": object_id,
            "validityStartMjdTai": self.rng.uniform(50000, 70000),
            "ra": self.rng.uniform(0, 360),
            "raErr": self._noneable(self.rng.uniform(0, 0.1)),
            "dec": self.rng.uniform(-90, 90),
            "decErr": self._noneable(self.rng.uniform(0, 0.1)),
            "ra_dec_Cov": self._noneable(self.rng.uniform(-0.01, 0.01)),
        }
        bands = self.bands
        for band in bands:
            obj[f"{band}_psfFluxMean"] = self._noneable(self.rng.uniform(-1000, 1000))
            obj[f"{band}_psfFluxMeanErr"] = self._noneable(self.rng.uniform(0, 100))
            obj[f"{band}_psfFluxSigma"] = self._noneable(self.rng.uniform(0, 100))
            obj[f"{band}_psfFluxNdata"] = self._noneable(self.rng.randint(0, 1000))
            obj[f"{band}_fpFluxMean"] = self._noneable(self.rng.uniform(-1000, 1000))
            obj[f"{band}_fpFluxMeanErr"] = self._noneable(self.rng.uniform(0, 100))
            obj[f"{band}_scienceFluxMean"] = self._noneable(
                self.rng.uniform(-1000, 1000)
            )
            obj[f"{band}_scienceFluxMeanErr"] = self._noneable(self.rng.uniform(0, 100))
            obj[f"{band}_psfFluxMin"] = self._noneable(self.rng.uniform(-1000, 1000))
            obj[f"{band}_psfFluxMax"] = self._noneable(self.rng.uniform(-1000, 1000))
            obj[f"{band}_psfFluxMaxSlope"] = self._noneable(self.rng.uniform(-100, 100))
            obj[f"{band}_psfFluxErrMean"] = self._noneable(self.rng.uniform(0, 100))
        obj["firstDiaSourceMjdTai"] = self._noneable(self.rng.uniform(50000, 70000))
        obj["lastDiaSourceMjdTai"] = self._noneable(self.rng.uniform(50000, 70000))
        obj["nDiaSources"] = self.rng.randint(1, 1000)
        return obj

    def _random_mpc_orbit(self) -> dict[str, Any]:
        return {
            "id": self.rng.randint(0, 9999999),
            "designation": f"DES{self.rng.randint(10000, 99999)}",
            "packed_primary_provisional_designation": f"K{self.rng.randint(10, 99)}A{self.rng.randint(10, 99)}B",
            "unpacked_primary_provisional_designation": f"20{self.rng.randint(10, 99)} AB",
            "mpc_orb_jsonb": self._noneable("{}"),
            "created_at": None,
            "updated_at": None,
            "orbit_type_int": self._noneable(self.rng.randint(0, 10)),
            "u_param": self._noneable(self.rng.randint(0, 9)),
            "nopp": self._noneable(self.rng.randint(0, 10)),
            "arc_length_total": self._noneable(self.rng.uniform(0, 10000)),
            "arc_length_sel": self._noneable(self.rng.uniform(0, 10000)),
            "nobs_total": self._noneable(self.rng.randint(0, 10000)),
            "nobs_total_sel": self._noneable(self.rng.randint(0, 10000)),
            "a": self._noneable(self.rng.uniform(0.5, 50)),
            "q": self._noneable(self.rng.uniform(0.5, 50)),
            "e": self._noneable(self.rng.uniform(0, 1)),
            "i": self._noneable(self.rng.uniform(0, 180)),
            "node": self._noneable(self.rng.uniform(0, 360)),
            "argperi": self._noneable(self.rng.uniform(0, 360)),
            "peri_time": self._noneable(self.rng.uniform(0, 10000)),
            "yarkovsky": self._noneable(self.rng.uniform(-1, 1)),
            "srp": self._noneable(self.rng.uniform(-1, 1)),
            "a1": self._noneable(self.rng.uniform(-1, 1)),
            "a2": self._noneable(self.rng.uniform(-1, 1)),
            "a3": self._noneable(self.rng.uniform(-1, 1)),
            "dt": self._noneable(self.rng.uniform(-1, 1)),
            "mean_anomaly": self._noneable(self.rng.uniform(0, 360)),
            "period": self._noneable(self.rng.uniform(0, 10000)),
            "mean_motion": self._noneable(self.rng.uniform(0, 1)),
            "a_unc": self._noneable(self.rng.uniform(0, 1)),
            "q_unc": self._noneable(self.rng.uniform(0, 1)),
            "e_unc": self._noneable(self.rng.uniform(0, 1)),
            "i_unc": self._noneable(self.rng.uniform(0, 1)),
            "node_unc": self._noneable(self.rng.uniform(0, 1)),
            "argperi_unc": self._noneable(self.rng.uniform(0, 1)),
            "peri_time_unc": self._noneable(self.rng.uniform(0, 1)),
            "yarkovsky_unc": self._noneable(self.rng.uniform(0, 1)),
            "srp_unc": self._noneable(self.rng.uniform(0, 1)),
            "a1_unc": self._noneable(self.rng.uniform(0, 1)),
            "a2_unc": self._noneable(self.rng.uniform(0, 1)),
            "a3_unc": self._noneable(self.rng.uniform(0, 1)),
            "dt_unc": self._noneable(self.rng.uniform(0, 1)),
            "mean_anomaly_unc": self._noneable(self.rng.uniform(0, 1)),
            "period_unc": self._noneable(self.rng.uniform(0, 1)),
            "mean_motion_unc": self._noneable(self.rng.uniform(0, 1)),
            "epoch_mjd": self._noneable(self.rng.uniform(50000, 70000)),
            "h": self._noneable(self.rng.uniform(10, 30)),
            "g": self._noneable(self.rng.uniform(-0.5, 0.8)),
            "not_normalized_rms": self._noneable(self.rng.uniform(0, 10)),
            "normalized_rms": self._noneable(self.rng.uniform(0, 10)),
            "earth_moid": self._noneable(self.rng.uniform(0, 10)),
            "fitting_datetime": None,
        }

    def _random_ss_object(self, object_id: int) -> dict[str, Any]:
        obj = {
            "ssObjectId": object_id,
            "designation": self._noneable(f"DES{self.rng.randint(10000, 99999)}"),
            "nObs": self.rng.randint(0, 1000),
            "arc": self.rng.uniform(0, 10000),
            "firstObservationMjdTai": self._noneable(self.rng.uniform(50000, 70000)),
            "MOIDEarth": self._noneable(self.rng.uniform(0, 10)),
            "MOIDEarthDeltaV": self._noneable(self.rng.uniform(0, 100)),
            "MOIDEarthEclipticLongitude": self._noneable(self.rng.uniform(0, 360)),
            "MOIDEarthTrueAnomaly": self._noneable(self.rng.uniform(0, 360)),
            "MOIDEarthTrueAnomalyObject": self._noneable(self.rng.uniform(0, 360)),
            "tisserand_J": self._noneable(self.rng.uniform(2, 4)),
            "extendednessMax": self._noneable(self.rng.uniform(0, 1)),
            "extendednessMedian": self._noneable(self.rng.uniform(0, 1)),
            "extendednessMin": self._noneable(self.rng.uniform(0, 1)),
        }
        bands = self.bands
        for band in bands:
            obj[f"{band}_nObs"] = self._noneable(self.rng.randint(0, 1000))
            obj[f"{band}_H"] = self._noneable(self.rng.uniform(10, 30))
            obj[f"{band}_HErr"] = self._noneable(self.rng.uniform(0, 1))
            obj[f"{band}_G12"] = self._noneable(self.rng.uniform(-0.5, 0.8))
            obj[f"{band}_G12Err"] = self._noneable(self.rng.uniform(0, 0.5))
            obj[f"{band}_H_{band}_G12_Cov"] = self._noneable(
                self.rng.uniform(-0.1, 0.1)
            )
            obj[f"{band}_nObsUsed"] = self._noneable(self.rng.randint(0, 1000))
            obj[f"{band}_Chi2"] = self._noneable(self.rng.uniform(0, 1000))
            obj[f"{band}_phaseAngleMin"] = self._noneable(self.rng.uniform(0, 180))
            obj[f"{band}_phaseAngleMax"] = self._noneable(self.rng.uniform(0, 180))
            obj[f"{band}_slope_fit_failed"] = self._noneable(
                self.rng.choice([True, False])
            )
        return obj
