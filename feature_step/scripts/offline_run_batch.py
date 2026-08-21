#!/usr/bin/env python
"""Batched, multi-process offline runner: oids -> features + BHRF probabilities.

The per-oid path (`classify.classify_oid`) is the *validated* one, but it drives
every reader one oid at a time: 4 single-oid SQL queries plus one Xwave HTTP
round trip per object. The end-to-end benchmark put ~68% of the 2.30 s/oid in
those round trips and only ~0.58 s in actual compute. This runner keeps the
compute path byte-identical and batches everything around it.

Two levels of batching, which do different jobs:

  * work unit (--unit-size, default 5000 oids) -- the CHECKPOINT granularity.
    One unit produces one parquet shard plus one manifest, written atomically
    (.tmp then os.replace). A completed unit is never redone on a rerun.
  * minibatch (--minibatch, default 500 oids) -- the ROUND-TRIP granularity.
    One minibatch = 4 SQL queries and 1 Xwave call for all 500 oids, instead of
    2000 queries and 500 calls. 500 matches xmatch.DEFAULT_BATCH_SIZE.

Units are contiguous oid ranges, deliberately NOT n_det strata: stratifying by
light-curve length makes the long units stragglers and leaves most cores idle at
the tail. Contiguous ranges give every unit the same expected cost for free.

Parallelism is processes, not threads (feature extraction is CPU-bound Python /
numba, so the GIL would serialize threads). Note the two traps this script
handles for you:

  * BLAS/OpenMP/JAX each spawn N threads PER process, so N processes give N^2
    threads fighting over N cores. The env vars at the top of this file pin them
    to 1 and MUST be set before numpy is imported.
  * with --start-method fork (the default, Linux) the BHRF model is loaded once
    in the parent and shared copy-on-write. Under spawn each worker loads its
    own copy of the pickle.

Failures are isolated per oid and per unit. A unit that dies (e.g. Xwave down)
is simply not marked done, so rerunning the same command picks it up. A single
bad OBJECT does not kill its unit -- but the unit then completes and marks
itself done, so the resume logic will never look at it again. Every failed oid
is therefore written to errors/unit_*.jsonl (the manifest only samples 20), and
retrying them is a second, small run over that list:

    cat <out-dir>/errors/*.jsonl | jq -r .oid > retry.txt
    python offline_run_batch.py --oid-file retry.txt --out-dir <out-dir>-retry

A separate --out-dir is required, not optional: the oid list differs, so the
run fingerprint will not match the original shards -- which is the guard doing
its job, not an obstacle.

Xwave/DB errors are retried with backoff rather than silently degraded -- a
missing AllWISE crossmatch changes the predicted class, so falling back to
"no WISE" would quietly corrupt results.

Output is parquet shards, always. --load-db additionally upserts each finished
unit into <schema>.probability, <schema>.xmatch, and <schema>.feature with
--features: ONE statement-batch per table per unit, ~2% of the unit's wall
clock, not one transaction per oid. The write lands BEFORE the manifest, so a
unit only counts as done once its rows are committed.

The xmatch rows are the crossmatch link the live step sends to the scribe
(step.produce_xmatch_to_scribe). This run stands in for that step, so without
them the objects it processes end up classified with no record of which AllWISE
source they matched or how far away it was -- data Xwave already gave us and
that the features were built from.

--load-db is a start-of-run decision, not something to switch on midway. Resume
skips a unit when its MANIFEST exists, never by looking at the database, so any
unit finished while the flag was off stays on disk forever and is never loaded.

Typical use:

    export MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl
    export XMATCH_URL=http://127.0.0.1:8081

    # 1. materialize the oid list once (n_det >= 6 -> ~7.45M of the 130M)
    python feature_step/scripts/offline_run_batch.py --min-n-det 6 \
        --save-oids feature_step/features/offline/oids/run_ndet6.npy --plan-only

    # 2. scaling probe: where does throughput stop scaling?
    python feature_step/scripts/offline_run_batch.py \
        --oid-file feature_step/features/offline/oids/run_ndet6.npy \
        --out-dir /data/bhrf_run --workers 16 --max-units 16

    # 3. the real run (resumable: rerun the same command after any interruption)
    python feature_step/scripts/offline_run_batch.py \
        --oid-file feature_step/features/offline/oids/run_ndet6.npy \
        --out-dir /data/bhrf_run --workers 64 --features \
        --load-db --write-credentials feature_step/features/offline/write_credentials.json
"""
# --- thread pinning: MUST happen before numpy/BLAS/JAX are imported ---
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for _p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
           PIPE / "libs" / "apf", PIPE / "libs" / "xmatch_client",
           PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(_p))

import numpy as np
import sqlalchemy as sa
import pandas as pd
from sqlalchemy import text


def _silence_library_warnings() -> None:
    """Mute the known-noisy pandas/numpy deprecations from lc_classifier and the
    Squidward mapper. They fire per extractor per object, so over millions of
    oids they are gigabytes of log that hide the lines an operator needs. Scoped
    to these two categories only -- --warnings turns them back on."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

from features.offline import db, xmatch
from features.offline.classify import classify_astro_object, load_squidward_model
from features.offline.classifier_taxonomy_lut import CLASSIFIER_VERSION
from features.offline.feature_lut import default_version_name
from features.offline.lc_features import compute_astro_object
from features.offline.message import build_message
from features.offline.feature_writer import write_features
from features.offline.xmatch import persist_matches
from features.offline.probability_writer import (
    CLASSIFIER_IDS, build_probability_rows, write_probabilities)
from features.utils.parsers import prepare_ao_features_for_db

DEFAULT_CREDENTIALS = str(PIPE / "feature_step" / "features" / "offline" / "credentials.json")
SID_ZTF = 0

# Worker-process singletons, built once by _init_worker and reused for every unit
# that process handles. Under fork these are inherited from the parent (so the
# model pickle is shared copy-on-write); under spawn each worker builds its own.
_W: dict = {}


# --------------------------------------------------------------------------- #
#  oid selection
# --------------------------------------------------------------------------- #
def physical_cores() -> int:
    """Physical cores, NOT os.cpu_count().

    os.cpu_count() reports hardware threads: on a dual-socket EPYC 7662 with SMT2
    that is 256 for 128 real cores, so sizing the pool from it oversubscribes
    2:1. Feature extraction is CPU-bound with the BLAS threads already pinned to
    1, so the second hyperthread of a core buys almost nothing and the extra
    context switching costs. Count distinct (physical id, core id) pairs.
    """
    try:
        cores, phys, core = set(), None, None
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    cores.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            cores.add((phys, core))
        if cores:
            return len(cores)
    except OSError:
        pass   # not Linux, or /proc unavailable
    return os.cpu_count() or 4


def default_workers() -> int:
    """Leave two cores for the parent, the DB driver and the OS."""
    return max(1, physical_cores() - 2)


def default_start_method() -> str:
    """fork on Linux, spawn on macOS.

    fork is what makes the model shareable copy-on-write, so it is the right
    choice for the real (Linux) run. On macOS it is not merely discouraged: a
    forked child here dies with SIGSEGV before running a single line of Python,
    because the parent has already initialised numpy/Accelerate and Apple's BLAS
    is not fork-safe. Measured on this repo: fork -> exitcode -11 whether or not
    the model is preloaded; spawn -> clean. It is also INTERMITTENT, so a fork
    run that passes a smoke test can still die in production.
    """
    return "spawn" if platform.system() == "Darwin" else "fork"


def select_oids(credentials: str, min_n_det: int, limit=None) -> np.ndarray:
    """oids of <schema>.object with n_det >= min_n_det, ascending.

    Ascending order is load-bearing twice: it is what makes the contiguous work
    units cheap (a unit's oids land on adjacent index pages instead of
    scattering across the heap), and it is what makes the run fingerprint
    reproducible, so a rerun resumes instead of starting over.

    numpy does the sorting, not Postgres. The only index that serves the filter
    is on n_det, so a full cut (n_det >= 2, ~26M rows) plans as a bitmap heap
    scan over the 8 partitions followed by an external sort -- which roughly
    doubles the cost of the query. Sorting 26M int64 client-side is 212 MB and a
    couple of seconds.

    With a limit it is the other way round: ORDER BY oid LIMIT n walks the
    (oid, sid) primary key in order and stops at n, touching nothing else. It
    also has to stay, because it is what makes "the first n oids" mean the same
    thing on every run -- the verification scripts compare across runs.
    """
    sql = f"""
        SELECT oid FROM {db.SCHEMA}.object
        WHERE sid = :sid AND n_det >= :min_n_det
    """
    if limit:
        sql += " ORDER BY oid LIMIT :limit"
    params = {"sid": db.SID, "min_n_det": min_n_det}
    if limit:
        params["limit"] = limit
    engine = db._make_engine(credentials)
    with engine.connect() as conn:
        conn = conn.execution_options(stream_results=True)
        chunks = [c["oid"].to_numpy(dtype=np.int64)
                  for c in pd.read_sql_query(text(sql), conn, params=params,
                                             chunksize=1_000_000)]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(chunks)
    out.sort()
    return out


def load_oids(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.int64)
    return np.loadtxt(p, dtype=np.int64, ndmin=1)


def make_units(oids: np.ndarray, unit_size: int) -> list:
    """Split the (sorted) oid array into contiguous work units."""
    return [(i, oids[s:s + unit_size])
            for i, s in enumerate(range(0, len(oids), unit_size))]


# --------------------------------------------------------------------------- #
#  batched fetch
# --------------------------------------------------------------------------- #
def _by_oid(frame: pd.DataFrame) -> tuple:
    """(  {oid: sub-frame},  empty-frame-with-the-right-columns  ).

    The empty frame is the stand-in for oids the query returned nothing for; it
    keeps build_message / the parser on the same code path as the single-oid
    reader, which always hands them a frame rather than None.
    """
    if frame is None or len(frame) == 0:
        empty = frame if frame is not None else pd.DataFrame()
        return {}, empty
    return ({int(k): v for k, v in frame.groupby("oid", sort=False)},
            frame.iloc[0:0])


def _retry(fn, attempts: int, what: str, base_sleep: float = 2.0):
    """Run fn with exponential backoff; re-raise the last error if all fail.

    Deliberately raises instead of degrading: an empty AllWISE frame is a valid
    -- but WRONG -- input that silently changes the predicted class.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            wait = base_sleep * 2 ** (attempt - 1)
            print(f"  [pid {os.getpid()}] {what} failed ({type(exc).__name__}: {exc}); "
                  f"retry {attempt}/{attempts - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_minibatch(oids: list, cfg: dict) -> tuple:
    """One round of batched reads for `oids`: 4 SQL queries, 1 Xwave call.

    Returns ({oid: (message, refs, allwise)}, matches). The first is only the
    oids that have detections; oids with no detections are absent (nothing to
    classify). The second is the raw Xwave match list -- empty when the AllWISE
    frames came from the DB instead -- handed back so the caller can persist the
    crossmatch it already paid for rather than recomputing it later.
    """
    cred, retries = cfg["credentials"], cfg["retries"]

    dets_by, dets_empty = _by_oid(_retry(
        lambda: db.fetch_detections(cred, oids), retries, "detections"))
    forced_by, forced_empty = _by_oid(_retry(
        lambda: db.fetch_forced_photometry(cred, oids), retries, "forced"))
    ps1_by, ps1_empty = _by_oid(_retry(
        lambda: db.fetch_ps1(cred, oids), retries, "ps1"))
    refs_by, refs_empty = _by_oid(_retry(
        lambda: db.fetch_references(cred, oids), retries, "references"))

    # Messages first: the crossmatch cone centre is the message's meanra/meandec,
    # so it can only be built after the detections are assembled.
    messages = {}
    for oid in oids:
        dets = dets_by.get(oid)
        if dets is None or len(dets) == 0:
            continue  # no detections -> nothing to classify
        messages[oid] = build_message(oid, dets, forced_by.get(oid, forced_empty),
                                      ps1_by.get(oid, ps1_empty))

    # One crossmatch call for the whole minibatch (vs one per oid).
    allwise_by, allwise_empty = {}, pd.DataFrame(columns=["oid", "W1", "W2", "W3", "W4"])
    matches = []   # raw Xwave matches, handed back so the unit can persist them
    if messages:
        if cfg["xmatch_url"]:
            mb_oids = list(messages)
            ras = [messages[o]["meanra"] for o in mb_oids]
            decs = [messages[o]["meandec"] for o in mb_oids]
            matches = _retry(
                lambda: xmatch.compute_matches(mb_oids, ras, decs,
                                               base_url=cfg["xmatch_url"]),
                retries, "xmatch")
            allwise_by, _ = _by_oid(xmatch.matches_to_allwise_df(matches))
        else:
            allwise_by, _ = _by_oid(_retry(
                lambda: db.fetch_allwise(cred, oids), retries, "allwise"))

    return ({oid: (msg, refs_by.get(oid, refs_empty),
                   allwise_by.get(oid, allwise_empty))
             for oid, msg in messages.items()},
            matches)


# --------------------------------------------------------------------------- #
#  per-oid compute (unchanged from the validated single-oid path)
# --------------------------------------------------------------------------- #
def process_oid(oid: int, message: dict, refs, allwise, cfg: dict):
    """-> (probability_rows, feature_rows | None); (None, None) if unclassifiable."""
    ao = compute_astro_object(message, refs, allwise, cfg["min_detections"],
                              preprocessor=_W["preprocessor"], extractor=_W["extractor"])
    if ao is None:
        return None, None

    dto = classify_astro_object(ao, message, _W["model"])
    # build_message emits forced epochs inside `detections` (forced=True), so this
    # max is already max(detections, forced) -- same value as classify._lc_lastmjd.
    mjds = [d["mjd"] for d in message["detections"]]
    lastmjd = max(mjds) if mjds else None
    prob_rows = build_probability_rows(dto, oid, lastmjd, _W["taxonomy"],
                                       version=CLASSIFIER_VERSION, sid=SID_ZTF)

    feat_rows = None
    if cfg["features"]:
        # Reuse the AstroObject we already extracted rather than calling
        # compute_db_features, which would run the whole extractor a second time.
        feat_rows = prepare_ao_features_for_db(ao, _W["feature_lut"])
        feat_rows = feat_rows.copy()
        feat_rows["oid"] = int(oid)
        feat_rows["sid"] = SID_ZTF
        feat_rows["version"] = _W["feature_version_id"]
        feat_rows = feat_rows.drop(columns=["name"])[
            ["oid", "sid", "feature_id", "band", "version", "value"]]
    return prob_rows, feat_rows


# --------------------------------------------------------------------------- #
#  worker
# --------------------------------------------------------------------------- #
def _init_worker(cfg: dict):
    # Ctrl-C is handled by the parent; workers must not each raise their own.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor

    if not cfg["warnings"]:
        _silence_library_warnings()
    _W["cfg"] = cfg
    # Built once per process: the extractor is heavy, and numba/jax pay their JIT
    # on first use, so per-unit construction would dominate a short unit.
    _W["preprocessor"] = ZTFLightcurvePreprocessor(drop_bogus=True)
    _W["extractor"] = ZTFFeatureExtractor()
    # Under fork the parent's already-loaded model is inherited (copy-on-write);
    # under spawn there is nothing to inherit, so load it here.
    _W["model"] = _MODEL if _MODEL is not None else load_squidward_model()[0]
    _W["taxonomy"] = db.fetch_taxonomy_maps(cfg["credentials"], CLASSIFIER_IDS,
                                            schema=cfg["schema"])
    # Both ids come from cfg, where main() resolved them against the DB once
    # (see the preflight there) -- never from the local fixture.
    _W["feature_lut"] = cfg.get("feature_lut")
    _W["feature_version_id"] = cfg.get("feature_version_id")


def process_unit(unit) -> dict:
    """Run one work unit end to end and write its shard. Returns a manifest dict."""
    index, oids = unit
    cfg = _W["cfg"]
    out_dir = Path(cfg["out_dir"])
    t0 = time.perf_counter()

    oids = [int(o) for o in oids]
    prob_rows, feat_frames, unit_matches = [], [], []
    n_ok = n_errors = n_no_allwise = n_no_detections = n_unclassifiable = 0
    # `failed` keeps EVERY failed oid and lands in errors/unit_*.jsonl; the
    # manifest only carries a capped sample of it. The unit still completes and
    # still writes its manifest, so the resume logic will skip it forever --
    # without the full list those oids are counted, unnamed and unrecoverable.
    failed = []

    for start in range(0, len(oids), cfg["minibatch"]):
        mb = oids[start:start + cfg["minibatch"]]
        # A failure here aborts the UNIT (not the run): the unit stays unmarked
        # and a rerun picks it up, rather than writing a shard with a silent hole.
        inputs, mb_matches = fetch_minibatch(mb, cfg)
        if cfg.get("load_db"):
            unit_matches.extend(mb_matches)
        for oid in mb:
            got = inputs.get(oid)
            if got is None:
                n_no_detections += 1   # nothing to classify; not a failure
                continue
            # An oid the crossmatch was ASKED about and that came back empty.
            # Counted here, after the no-detections skip above, because an oid
            # with no detections never reaches the cone search and so is not a
            # miss. Without this number, "no counterpart" and "the crossmatch
            # never ran" are indistinguishable in everything the run leaves
            # behind: no WISE rows in <schema>.feature, no row in
            # <schema>.xmatch, either way. Expect ~14%
            # (WISE_NULL_CLASSIFICATION_IMPACT.md puts recovery at 86%).
            if len(got[2]) == 0:
                n_no_allwise += 1
            try:
                p_rows, f_rows = process_oid(oid, *got, cfg)
            except Exception as exc:
                n_errors += 1
                failed.append({"oid": oid, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not p_rows:
                n_unclassifiable += 1   # too few real detections; not a failure
                continue
            prob_rows.extend(p_rows)
            if f_rows is not None and len(f_rows):
                feat_frames.append(f_rows)
            n_ok += 1

    feats = (pd.concat(feat_frames, ignore_index=True) if feat_frames
             else pd.DataFrame(columns=["oid", "sid", "feature_id",
                                        "band", "version", "value"]))
    # The shards are the primary output UNLESS --load-db is on, in which case the
    # database holds the same rows and nothing ever reads them back: resume keys
    # off the manifest, and no loader exists that could consume them. That is
    # ~68 GB across the full run, so it is worth being able to say no.
    if not cfg.get("no_shards"):
        _write_shard(out_dir / "probabilities" / f"unit_{index:07d}.parquet",
                     pd.DataFrame(prob_rows))
        if cfg["features"]:
            _write_shard(out_dir / "features" / f"unit_{index:07d}.parquet", feats)

    # --- optional load into the DB -----------------------------------------
    # ONE call per table per unit, not per oid: the writers batch every row into
    # one statement per page, which is ~2% of a unit's wall clock. Per-oid would
    # be ~15M transactions across the run instead of ~3k, for the same rows.
    #
    # Placed BEFORE the manifest on purpose. The manifest is the done-marker, so
    # a unit is only finished once its rows are committed; if this raises, the
    # unit stays unmarked and the rerun redoes it -- safe, because the upsert is
    # idempotent. Writing the manifest first would strand the unit as "done"
    # with nothing in the database.
    n_db_prob = n_db_feat = n_db_xmatch = 0
    if cfg.get("load_db"):
        wc = cfg["write_credentials"]
        if prob_rows:
            n_db_prob = write_probabilities(
                prob_rows, wc, schema=cfg["schema"], execute=True)["written"]
        if cfg["features"] and len(feats):
            n_db_feat = write_features(
                feats, wc, schema=cfg["schema"], execute=True)["written"]
        # The crossmatch link rows: in the live pipeline writing these IS the
        # feature step's job (step.produce_xmatch_to_scribe), and this run
        # stands in for that step. Guarded on the list being non-empty because
        # without --xmatch-url the AllWISE frames come from the DB and there is
        # nothing to write -- persist_matches([]) would open a connection per
        # unit to do nothing.
        if unit_matches:
            n_db_xmatch = persist_matches(
                unit_matches, wc, schema=cfg["schema"], execute=True)["written"]

    # Also before the manifest: its presence must mean the error list is complete.
    _write_jsonl(out_dir / "errors" / f"unit_{index:07d}.jsonl", failed)

    manifest = {
        "unit": index, "oid_lo": oids[0], "oid_hi": oids[-1], "n_oids": len(oids),
        "n_ok": n_ok, "n_errors": n_errors,
        # n_skipped is the total of the three reasons below, kept so existing
        # readers of older manifests keep working. Only n_errors is worth a retry.
        "n_skipped": n_errors + n_no_detections + n_unclassifiable,
        "n_no_detections": n_no_detections,
        "n_unclassifiable": n_unclassifiable,
        "n_no_allwise": n_no_allwise,
        "prob_rows": len(prob_rows),
        "feat_rows": int(sum(len(f) for f in feat_frames)),
        "db_prob_rows": n_db_prob, "db_feat_rows": n_db_feat,
        "db_xmatch_rows": n_db_xmatch,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": worker_peak_rss_mb(),
        "errors": failed[:20],   # sample; errors/unit_*.jsonl has all of them
    }
    _write_json(out_dir / "manifests" / f"unit_{index:07d}.json", manifest)
    return manifest


def shards_would_be_lost(no_shards: bool, load_db: bool) -> bool:
    """True when --no-shards would throw the run's output away.

    The shards are only redundant because the database holds the same rows.
    Without --load-db they are the ONLY output, and dropping them means
    computing 26M objects and keeping nothing but the counters.
    """
    return no_shards and not load_db


def rss_mb(ru_maxrss: int, system: str) -> float:
    """getrusage's ru_maxrss -> MB.

    The field is kilobytes on Linux and BYTES on macOS/BSD. Reading it raw makes
    a Mac look 1024x worse than it is, which is exactly the kind of number that
    gets a run cancelled for no reason.
    """
    return round(ru_maxrss / (1024 * 1024) if system == "Darwin" else ru_maxrss / 1024, 1)


def worker_peak_rss_mb() -> float:
    """Peak RSS of THIS worker process since it started.

    Under fork the 1.7 GB model is shared copy-on-write, but refcounting dirties
    the pages it touches, so the sharing degrades as the model is used. Whether
    it degrades a little or all the way to one private copy per worker is the
    difference between a run that finishes and a machine that gets OOM-killed --
    and until this landed in the manifest it was only visible to somebody
    watching `top` while the probe happened to be running.
    """
    import resource
    return rss_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, platform.system())


def exit_code(n_failed_units: int, n_oid_errors: int) -> int:
    """Process exit status for a finished run.

    Non-zero when UNITS were lost: those oids produced nothing and the only
    thing that recovers them is somebody rerunning the command, which a
    supervisor or cron entry will never do if the run reports success.

    Per-oid errors deliberately do NOT fail the run. At 26M objects a handful
    of bad ones is an expected outcome; they are counted, named in
    errors/unit_*.jsonl and retried explicitly (SERVER_RUNBOOK.md §10). Folding
    them in would make the exit code non-zero on essentially every run, which
    is the same as having no exit code at all.
    """
    return 1 if n_failed_units else 0


def run_fingerprint(oids: np.ndarray, unit_size: int) -> dict:
    """Identify the (oid list, unit size) a set of shards belongs to.

    A unit index only means something relative to a specific oid array: unit 37
    IS oids[185000:190000]. Resuming against a DIFFERENT array -- a changed
    --min-n-det, or a re-SELECT after the table grew -- silently maps finished
    indices onto different oids, skipping objects that were never processed.
    The digest makes that mismatch loud instead of silent.
    """
    return {
        "n_oids": int(len(oids)),
        "unit_size": int(unit_size),
        "oid_sha1": hashlib.sha1(np.ascontiguousarray(oids).tobytes()).hexdigest(),
        "oid_lo": int(oids[0]) if len(oids) else None,
        "oid_hi": int(oids[-1]) if len(oids) else None,
    }


def check_or_write_fingerprint(out_dir: Path, fp: dict, force: bool) -> None:
    """Refuse to resume a run whose oid list no longer matches these shards."""
    path = out_dir / "run.json"
    if not path.exists():
        _write_json(path, fp)
        return
    old = json.loads(path.read_text())
    if old == fp:
        return
    diff = {k: (old.get(k), fp.get(k)) for k in fp if old.get(k) != fp.get(k)}
    msg = ("\nREFUSING TO RESUME: this output directory was built from a "
           "different oid list.\n"
           f"  {out_dir}\n"
           + "".join(f"  {k}: stored={o!r} now={n!r}\n" for k, (o, n) in diff.items())
           + "  Unit indices from the stored run point at different oids now, so\n"
             "  resuming would skip objects that were never processed.\n"
             "  Fix: pass the SAME --oid-file used before (that is what --save-oids\n"
             "  is for), or use a fresh --out-dir. --force-resume overrides.\n")
    if not force:
        raise SystemExit(msg)
    print(msg + "  --force-resume given; continuing anyway.\n")
    _write_json(path, fp)


def clean_stale_tmp(out_dir: Path) -> int:
    """Remove .tmp files left by a hard kill. They are never read (the atomic
    rename means only the final name is visible), but they accumulate."""
    n = 0
    for pattern in ("*/*.parquet.tmp", "*/*.json.tmp"):
        for stale in out_dir.glob(pattern):
            stale.unlink()
            n += 1
    return n


def _write_shard(path: Path, frame: pd.DataFrame) -> None:
    """Atomic parquet write: a crash mid-write can never leave a half shard that
    a later run would mistake for finished output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_jsonl(path: Path, records: list) -> None:
    """Atomically write one JSON object per line. No records -> no file, so
    `ls errors/` names exactly the units that hit trouble."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)   # the manifest is the done-marker, so it lands LAST


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
_MODEL = None


def main():
    ap = argparse.ArgumentParser(
        description="Batched multi-process offline feature + BHRF classification runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--min-n-det", type=int,
                     help="select oids from <schema>.object with n_det >= this.")
    src.add_argument("--oid-file", help="read oids from a .npy or newline .txt file.")
    ap.add_argument("--save-oids", help="write the selected oid array to this .npy.")
    ap.add_argument("--limit", type=int, help="cap the selection (calibration runs).")
    ap.add_argument("--out-dir", help="output root for shards + manifests.")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="worker processes. Default is PHYSICAL cores - 2 "
                         "(%(default)s here); os.cpu_count() would count "
                         "hyperthreads and oversubscribe a CPU-bound run.")
    ap.add_argument("--unit-size", type=int, default=5000,
                    help="oids per work unit = checkpoint granularity.")
    ap.add_argument("--minibatch", type=int, default=xmatch.DEFAULT_BATCH_SIZE,
                    help="oids per batched DB/Xwave round trip.")
    ap.add_argument("--max-units", type=int, help="stop after N units (scaling probe).")
    ap.add_argument("--min-detections", type=int, default=1)
    ap.add_argument("--features", action="store_true",
                    help="also write feature shards (~199 rows/oid).")
    ap.add_argument("--load-db", action="store_true", dest="load_db",
                    help="also upsert each finished unit into <schema>.probability "
                         "(and <schema>.feature with --features). Requires "
                         "--write-credentials. Off by default: the parquet shards are "
                         "the primary output, and a probe run should not touch the DB.")
    ap.add_argument("--no-shards", action="store_true", dest="no_shards",
                    help="skip the parquet shards; requires --load-db. The DB then "
                         "holds the same rows and nothing reads the shards back "
                         "(~68 GB saved across a full run).")
    ap.add_argument("--write-credentials", dest="write_credentials",
                    help="credentials JSON with INSERT rights, required by --load-db "
                         "(--credentials may be read-only).")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per DB/Xwave call before failing the unit.")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
    ap.add_argument("--schema", default=db.SCHEMA)
    ap.add_argument("--xmatch-url", default=os.getenv("XMATCH_URL", xmatch.DEFAULT_XMATCH_URL),
                    dest="xmatch_url",
                    help="Xwave URL (default: %(default)s; or set XMATCH_URL). Pass '' to "
                         "force the DB read instead -- which yields NO WISE at all, since "
                         "<schema>.allwise is empty for ZTF, so only do that deliberately.")
    ap.add_argument("--start-method", default=default_start_method(),
                    choices=["fork", "spawn"],
                    help="fork shares the loaded model copy-on-write (Linux); "
                         "spawn reloads it per worker but is the only safe "
                         "option on macOS. Default: %(default)s on this host.")
    ap.add_argument("--stall-timeout", type=float, default=1800.0,
                    help="abort if no unit completes for this many seconds.")
    ap.add_argument("--force-resume", action="store_true",
                    help="resume even if the oid list no longer matches the "
                         "shards already in --out-dir (dangerous: unit indices "
                         "would point at different oids).")
    ap.add_argument("--warnings", action="store_true",
                    help="keep pandas/numpy deprecation warnings (off: they are "
                         "per-object and drown a large run's log).")
    ap.add_argument("--plan-only", action="store_true",
                    help="select oids, report the plan, write --save-oids, and exit.")
    args = ap.parse_args()

    # Validate BEFORE the oid selection: --min-n-det scans a 130M-row table, and
    # finding out about a missing flag after that wastes minutes.
    if args.load_db and not args.write_credentials:
        ap.error("--load-db requires --write-credentials (--credentials may be read-only)")
    if shards_would_be_lost(args.no_shards, args.load_db):
        ap.error("--no-shards without --load-db would discard the run's only output")

    # Progress on a multi-hour run is usually watched through a redirected log,
    # where Python's default block buffering would hold it back for minutes.
    sys.stdout.reconfigure(line_buffering=True)

    # --- oid selection -----------------------------------------------------
    t0 = time.perf_counter()
    if args.oid_file:
        oids = load_oids(args.oid_file)
        print(f"loaded {len(oids):,} oids from {args.oid_file}")
    else:
        print(f"selecting oids with n_det >= {args.min_n_det} ...")
        oids = select_oids(args.credentials, args.min_n_det, args.limit)
        print(f"selected {len(oids):,} oids in {time.perf_counter() - t0:.1f}s")
    if args.limit:
        oids = oids[:args.limit]
    oids = np.sort(oids)

    if args.save_oids:
        Path(args.save_oids).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_oids, oids)
        print(f"saved oid array -> {args.save_oids}")

    units = make_units(oids, args.unit_size)
    print(f"host: {physical_cores()} physical cores / {os.cpu_count()} hw threads, "
          f"start-method={args.start_method}")
    print(f"plan: {len(oids):,} oids -> {len(units):,} units of {args.unit_size} "
          f"({args.minibatch}-oid minibatches, {args.workers} workers)")
    if args.plan_only:
        return 0
    if not args.out_dir:
        ap.error("--out-dir is required unless --plan-only")

    # --- resume: a unit with a manifest is done ----------------------------
    # The manifest is written LAST (after both shards), so its presence means the
    # unit's output is complete. A unit killed mid-flight leaves no manifest and
    # is simply redone -- recovery is per UNIT, never per worker or per core.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_or_write_fingerprint(out_dir, run_fingerprint(oids, args.unit_size),
                               args.force_resume)
    stale = clean_stale_tmp(out_dir)
    if stale:
        print(f"cleaned {stale} stale .tmp file(s) from an interrupted run")
    done = {int(p.stem.split("_")[1])
            for p in (out_dir / "manifests").glob("unit_*.json")}
    todo = [u for u in units if u[0] not in done]
    if done:
        print(f"resume: {len(done):,} units already done, {len(todo):,} to go")
    if args.max_units:
        todo = todo[:args.max_units]
        print(f"--max-units: running {len(todo)} unit(s) this pass")
    if not todo:
        print("nothing to do.")
        return 0

    cfg = {
        "credentials": args.credentials, "schema": args.schema,
        "load_db": args.load_db, "write_credentials": args.write_credentials,
        "no_shards": args.no_shards,
        "xmatch_url": args.xmatch_url or None, "out_dir": str(out_dir),
        "minibatch": args.minibatch, "min_detections": args.min_detections,
        "features": args.features, "retries": args.retries,
        "warnings": args.warnings,
    }
    # --- feature LUT ids: resolved against the DB, once, in the parent ----
    # <schema>.feature has NO foreign key to feature_name_lut / feature_version_lut,
    # so ids taken from the local fixture are never validated by the database: a
    # fixture that drifted would stamp millions of rows with ids that resolve to
    # something else, and nothing would reject them. Resolving here also fails the
    # run BEFORE forking N workers, with the real error instead of a
    # BrokenProcessPool from N initializers raising at once.
    if args.features:
        try:
            fver = _pkg_version("feature-step")
        except PackageNotFoundError:
            fver = default_version_name()   # running from source, not installed
        cfg["feature_lut"] = db.fetch_feature_name_lut(args.credentials,
                                                       schema=args.schema)
        cfg["feature_version_id"] = db.fetch_feature_version_id(
            args.credentials, fver, schema=args.schema)
        print(f"feature LUT from DB: {len(cfg['feature_lut'])} names; "
              f"version {fver} -> id {cfg['feature_version_id']}")

    # Fail before forking N workers if the write credentials cannot connect.
    if args.load_db:
        with db._make_engine(args.write_credentials).connect() as _conn:
            _conn.execute(sa.text("SELECT 1"))
        tables = [f"{args.schema}.probability"]
        if args.xmatch_url:
            tables.append(f"{args.schema}.xmatch")
        if args.features:
            tables.append(f"{args.schema}.feature")
        print("load-db: ON -> " + ", ".join(tables))

    if not args.warnings:
        _silence_library_warnings()   # the parent loads the model + logs too

    # --- load the model in the PARENT so fork shares it copy-on-write ------
    global _MODEL
    if args.start_method == "fork":
        # Only worth doing under fork: spawn cannot inherit it, so preloading
        # here would just cost the parent a copy nobody uses.
        print("loading BHRF model in the parent (shared copy-on-write by fork)...")
        _MODEL, mname, mversion = load_squidward_model()
        print(f"  model: {mname} version={mversion}")

    # Drop the parent's pooled connections: psycopg2 sockets are not fork-safe,
    # and a child must never inherit one. (db keys its engine cache by pid, so
    # children build their own regardless -- this just closes the parent's.)
    db.dispose_engines()

    ctx = mp.get_context(args.start_method)
    n_units = len(todo)
    t_run = time.perf_counter()
    agg = {"n_ok": 0, "n_skipped": 0, "n_errors": 0, "n_no_allwise": 0,
           "n_no_detections": 0, "n_unclassifiable": 0,
           "prob_rows": 0, "feat_rows": 0, "db_prob_rows": 0, "db_feat_rows": 0,
           "db_xmatch_rows": 0}
    n_failed = 0

    # ProcessPoolExecutor, NOT multiprocessing.Pool. When a worker dies abruptly
    # (a segfault in a native library, the OOM killer), Pool silently starts a
    # replacement and keeps doing so forever -- a run can spin all night, print
    # nothing, and finish nothing. The executor raises BrokenProcessPool instead,
    # which turns that silent hang into an immediate, diagnosable failure.
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                 initializer=_init_worker, initargs=(cfg,)) as ex:
            futures = {ex.submit(process_unit, u): u[0] for u in todo}
            pending, i = set(futures), 0
            last_progress = time.monotonic()
            while pending:
                finished, pending = wait(pending, timeout=args.stall_timeout,
                                         return_when=FIRST_COMPLETED)
                if not finished:
                    stalled = time.monotonic() - last_progress
                    print(f"WARNING: no unit completed in {stalled / 60:.1f} min "
                          f"({len(pending)} still running)", flush=True)
                    if stalled > args.stall_timeout:
                        for fut in pending:
                            fut.cancel()
                        raise SystemExit(
                            f"\nABORTING: stalled for {stalled / 60:.1f} min with no "
                            f"progress. Finished units are checkpointed; rerun to "
                            f"resume. Check the DB and the Xwave service.")
                    continue
                last_progress = time.monotonic()
                for fut in finished:
                    i += 1
                    try:
                        man = fut.result()
                    except Exception as exc:
                        # One unit failing is survivable: it stays unmarked, so a
                        # rerun retries it. Only a broken pool is fatal.
                        if isinstance(exc, BrokenProcessPool):
                            raise
                        n_failed += 1
                        print(f"[{i}/{n_units}] unit {futures[fut]:>7} FAILED: "
                              f"{type(exc).__name__}: {exc} -- left unmarked, "
                              f"rerun to retry", flush=True)
                        continue
                    for k in agg:
                        agg[k] += man[k]
                    elapsed = time.perf_counter() - t_run
                    rate = agg["n_ok"] / elapsed if elapsed else 0.0
                    eta = (n_units - i) * (elapsed / i) if i else 0.0
                    print(f"[{i}/{n_units}] unit {man['unit']:>7} "
                          f"ok={man['n_ok']:>5} skip={man['n_skipped']:>4} "
                          f"err={man['n_errors']:>3} {man['elapsed_s']:>7.1f}s | "
                          f"{rate:6.1f} oid/s  ETA {eta / 3600:5.2f}h", flush=True)
    except BrokenProcessPool as exc:
        raise SystemExit(
            f"\nABORTING: a worker process died abruptly ({exc}).\n"
            f"  start method: {args.start_method}\n"
            "  A worker killed before it can raise a Python exception is almost\n"
            "  always a native crash (fork-unsafe BLAS) or the OOM killer.\n"
            "  Try --start-method spawn, or fewer --workers if memory-bound.\n"
            "  Finished units are checkpointed; rerun to resume.")
    except KeyboardInterrupt:
        print("\ninterrupted -- finished units are checkpointed; "
              "rerun the same command to resume.")
        return 130

    elapsed = time.perf_counter() - t_run
    print("\n" + "=" * 70)
    print(f"  units          : {n_units:,}")
    print(f"  classified     : {agg['n_ok']:,}")
    print(f"  skipped        : {agg['n_skipped']:,}  "
          f"(no detections: {agg['n_no_detections']:,}, "
          f"unclassifiable: {agg['n_unclassifiable']:,}, "
          f"errors: {agg['n_errors']:,})")
    if agg["n_errors"]:
        print(f"  -> retry them  : cat {out_dir}/errors/*.jsonl | jq -r .oid > retry.txt")
    _asked = agg["n_ok"] + agg["n_errors"]
    print(f"  no AllWISE     : {agg['n_no_allwise']:,}"
          f"  ({100 * agg['n_no_allwise'] / _asked if _asked else 0:.1f}% of the oids "
          f"the crossmatch was asked about; ~14% expected)")
    if n_failed:
        print(f"  FAILED units   : {n_failed:,}  <- left unmarked; rerun to retry")
    print(f"  probability rows: {agg['prob_rows']:,}")
    if args.features:
        print(f"  feature rows   : {agg['feat_rows']:,}")
    if args.load_db:
        print(f"  upserted to DB : {agg['db_prob_rows']:,} probability, "
              f"{agg['db_feat_rows']:,} feature, {agg['db_xmatch_rows']:,} xmatch")
    print(f"  elapsed        : {elapsed/3600:.2f} h "
          f"({agg['n_ok']/elapsed if elapsed else 0:.1f} oid/s, "
          f"{elapsed*args.workers/max(agg['n_ok'],1):.3f} core-s/oid)")
    print(f"  output         : {out_dir}")
    print("=" * 70)
    return exit_code(n_failed, agg["n_errors"])


if __name__ == "__main__":
    sys.exit(main())
