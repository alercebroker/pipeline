#!/usr/bin/env python
"""Take a freshly cloned repo to the point where offline_run_batch.py can start.

SERVER_RUNBOOK.md explains what the steps mean and why; this runs them. Every
step is idempotent: it checks first and only does the expensive thing when it is
actually missing, so re-running after a fix costs seconds and re-running on a
ready machine costs nothing.

Two things it deliberately does NOT do, because it cannot:

  * write the credentials files -- they carry live passwords, so they are
    created by hand (the script checks they exist, connect, and have the
    privileges the run needs);
  * grant those privileges -- that needs a superuser on the database host.

Everything else -- the 1.7 GB model download and its md5, the LUT/taxonomy
seeds, Xwave, and the one-off oid list -- it handles.

    python scripts/offline_setup.py                    # check, and fix what it can
    python scripts/offline_setup.py --check-only       # report only, change nothing
    python scripts/offline_setup.py --min-n-det 6      # a smaller catalogue cut

Exit code is 0 only when the machine can start a run.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import namedtuple
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]  # .../pipeline
for _p in (PIPE / "feature_step", PIPE / "lc_classifier", PIPE / "libs" / "idmapper",
           PIPE / "libs" / "xmatch_client", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(_p))

OFFLINE = PIPE / "feature_step" / "features" / "offline"

# The deployed BHRF artifact. The md5 is the whole point of pinning it: a
# truncated download fails later inside the model loader with an error that says
# nothing about the download.
MODEL_URL = ("https://alerce-models.s3.amazonaws.com/squidward/2.1.0/"
             "hierarchical_random_forest_model.pkl")
MODEL_MD5 = "95e8e9f18fde62f22025e31a88ad81fa"

# Both big artefacts default beside the code, which whoever cloned the repo can
# write. An earlier default put the model under /data, which is root-owned on a
# normal host: the script did not report that, it crashed trying to mkdir it.
# Both are gitignored. Override either with the flags, or MODEL_PATH.
DEFAULT_MODEL_PATH = OFFLINE / "models" / "hierarchical_random_forest_model.pkl"
DEFAULT_OID_FILE = OFFLINE / "oids" / "run.npy"

# n_det >= 2 keeps ~26.3M of the 130M objects (>= 6 would keep ~7.5M). Measured
# on object_part_0 and extrapolated across the 8 hash partitions.
DEFAULT_MIN_N_DET = 2

# Every table the run reads. The grants were once issued for the three write
# tables only and the run still could not start: without SELECT on these there
# are no light curves, no oid list and no LUTs. Checked explicitly because a
# missing one surfaces mid-run, per worker, as a permission error on a query.
READ_TABLES = ("object", "detection", "ztf_detection",
               "forced_photometry", "ztf_forced_photometry",
               "ztf_ps1", "ztf_reference", "xmatch", "allwise",
               "feature_name_lut", "feature_version_lut", "taxonomy")
WRITE_TABLES = ("feature", "probability", "xmatch")

OK, DONE, MISSING, FAIL = "OK", "HECHO", "FALTA", "ERROR"
_BLOCKING = (MISSING, FAIL)

Result = namedtuple("Result", "name status detail")


def is_ready(results) -> bool:
    """Ready means nothing is missing and nothing failed.

    DONE is not a blocker: it means this pass did the work.
    """
    return not any(r.status in _BLOCKING for r in results)


def safe_step(name: str, fn):
    """Run a step; turn any unexpected error into a reported row.

    The script's whole job is to say what is wrong. A step that raises reports
    exactly one problem and abandons every check after it -- which is how a
    non-writable MODEL_PATH hid the state of the oid list, Xwave and the seeds.
    """
    try:
        return fn()
    except Exception as exc:
        return Result(name, FAIL, f"{type(exc).__name__}: {str(exc)[:140]}")


def missing_privileges(privs: dict) -> list:
    """{name: has_it} -> the names that are False, in order."""
    return [name for name, ok in privs.items() if not ok]


def verify_md5(path, expected: str) -> bool:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


# --------------------------------------------------------------------------- #
#  steps
# --------------------------------------------------------------------------- #
def step_imports() -> Result:
    """The C extensions (P4J, mhps) are built from source; a failure here is an
    install problem, not a configuration one."""
    try:
        import P4J  # noqa: F401
        import mhps  # noqa: F401
        from lc_classifier.features.composites.ztf import ZTFFeatureExtractor  # noqa: F401
        # The same dotted path classify.DEFAULT_MODEL_CLASS resolves at runtime,
        # so this fails here rather than after the oid list is already built.
        from alerce_classifiers.squidward.model import (  # noqa: F401
            SquidwardFeaturesClassifier)
    except Exception as exc:
        return Result("dependencias", FAIL,
                      f"{type(exc).__name__}: {exc} -- revisar el paso 3 del runbook")
    return Result("dependencias", OK, "P4J, mhps, lc_classifier, alerce_classifiers")


def step_credentials(path: Path, need_write: bool) -> Result:
    """Connect, then check every privilege the run depends on.

    Reads are checked even for the write user, because one account can now do
    the whole run and the first attempt at that failed: it had INSERT on the
    three output tables and SELECT on nothing else, so it could not read a
    single light curve.
    """
    from features.offline import db
    from sqlalchemy import text

    label = path.name
    if not path.exists():
        return Result(label, MISSING, f"crear {path} a mano (lleva password)")
    try:
        with db._make_engine(str(path)).connect() as c:
            user = c.execute(text("SELECT current_user")).scalar()
            privs = {"USAGE en el schema": c.execute(text(
                "SELECT has_schema_privilege(current_user, :s, 'USAGE')"),
                {"s": db.SCHEMA}).scalar()}
            for t in READ_TABLES:
                privs[f"SELECT {t}"] = c.execute(text(
                    "SELECT has_table_privilege(current_user, :t, 'SELECT')"),
                    {"t": f"{db.SCHEMA}.{t}"}).scalar()
            if need_write:
                for t in WRITE_TABLES:
                    privs[f"INSERT/UPDATE {t}"] = c.execute(text(
                        "SELECT has_table_privilege(current_user, :t, 'INSERT') "
                        "AND has_table_privilege(current_user, :t, 'UPDATE')"),
                        {"t": f"{db.SCHEMA}.{t}"}).scalar()
    except Exception as exc:
        return Result(label, FAIL, f"{type(exc).__name__}: {str(exc)[:120]}")

    absent = missing_privileges(privs)
    if absent:
        return Result(label, MISSING,
                      f"{user} conecta pero le falta: {', '.join(absent[:4])}"
                      f"{' ...' if len(absent) > 4 else ''} "
                      "-- ver los GRANT del paso 4 del runbook")
    what = "lee y escribe" if need_write else "lee"
    return Result(label, OK, f"{user} {what} todo lo que la corrida necesita")


def step_seeds(credentials: Path) -> Result:
    """No FK ties <schema>.feature to the LUTs, so ids that resolve to nothing
    are accepted silently. Checking here is the only thing that catches it."""
    from features.offline import db
    try:
        names = len(db.fetch_feature_name_lut(str(credentials)))
        version = db.fetch_feature_version_id(str(credentials), "27.5.7a31")
        heads = sorted(db.fetch_taxonomy_maps(str(credentials), [5, 6, 7, 8, 9]))
    except Exception as exc:
        return Result("LUTs sembrados", MISSING,
                      f"{type(exc).__name__}: {str(exc)[:100]} -- aplicar "
                      "ztf_feature_luts_seed.sql y ztf_classifier_taxonomy_seed.sql")
    if not names or heads != [5, 6, 7, 8, 9]:
        return Result("LUTs sembrados", MISSING,
                      f"{names} feature names, clasificadores {heads} -- faltan seeds")
    return Result("LUTs sembrados", OK,
                  f"{names} feature names, version 27.5.7a31 -> id {version}, "
                  f"clasificadores {heads}")


def step_model(path: Path, check_only: bool) -> Result:
    if path.exists():
        if verify_md5(path, MODEL_MD5):
            return Result("modelo BHRF 2.1.0", OK, str(path))
        if check_only:
            return Result("modelo BHRF 2.1.0", FAIL,
                          f"{path} existe pero el md5 no cuadra (descarga cortada)")
        path.unlink()

    if check_only:
        return Result("modelo BHRF 2.1.0", MISSING, f"{path} no existe")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"    bajando {MODEL_URL} (1.7 GB) ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if not verify_md5(tmp, MODEL_MD5):
        tmp.unlink()
        return Result("modelo BHRF 2.1.0", FAIL, "md5 no cuadra tras bajarlo")
    os.replace(tmp, path)
    return Result("modelo BHRF 2.1.0", DONE, f"bajado y verificado -> {path}")


def step_xmatch(url: str) -> Result:
    """Xwave is mandatory: <schema>.allwise is empty, so without it every WISE
    colour comes out NaN and the classifications carry a documented bias."""
    import urllib.error
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            code = r.status
    except urllib.error.HTTPError as exc:
        code = exc.code          # answering at all is what matters
    except Exception as exc:
        return Result("Xwave", FAIL,
                      f"{url} no responde ({type(exc).__name__}) -- sin esto la "
                      "corrida clasifica sin WISE y sesgada")
    return Result("Xwave", OK, f"{url} -> HTTP {code}")


def step_oids(credentials: Path, out: Path, min_n_det: int, check_only: bool) -> Result:
    import numpy as np
    if out.exists():
        n = len(np.load(out, mmap_mode="r"))
        return Result("lista de oids", OK, f"{n:,} oids en {out}")
    if check_only:
        return Result("lista de oids", MISSING, f"{out} no existe")

    import offline_run_batch as R
    print(f"    seleccionando oids con n_det >= {min_n_det} "
          "(un scan de object, tarda) ...", flush=True)
    oids = R.select_oids(str(credentials), min_n_det)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npy.tmp")
    # Through a file handle on purpose: np.save(path) appends ".npy" unless the
    # name already ends in it, so passing the staging name writes
    # "run.npy.tmp.npy" and the rename below finds nothing -- throwing away the
    # scan that just took an hour.
    with open(tmp, "wb") as fh:
        np.save(fh, oids)
    os.replace(tmp, out)
    return Result("lista de oids", DONE, f"{len(oids):,} oids (n_det >= {min_n_det}) -> {out}")


# --------------------------------------------------------------------------- #
def main():
    from features.offline import xmatch as X

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true",
                    help="report what is missing without downloading or querying anything heavy.")
    ap.add_argument("--min-n-det", type=int, default=DEFAULT_MIN_N_DET,
                    help="catalogue cut for the oid list (default %(default)s).")
    ap.add_argument("--credentials", type=Path, default=OFFLINE / "credentials.json")
    ap.add_argument("--write-credentials", type=Path,
                    default=OFFLINE / "credentials.json",
                    help="defaults to --credentials: one account holds every "
                         "privilege the run needs, so a second file is optional.")
    ap.add_argument("--model-path", type=Path,
                    default=Path(os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH),
                    help="default: %(default)s (gitignored).")
    ap.add_argument("--oid-file", type=Path, default=DEFAULT_OID_FILE,
                    help="default: %(default)s (gitignored).")
    ap.add_argument("--xmatch-url", default=os.environ.get("XMATCH_URL")
                    or X.DEFAULT_XMATCH_URL)
    args = ap.parse_args()

    print(f"setup offline ZTF -- repo {PIPE}\n")
    results = [safe_step("dependencias", step_imports)]
    # One account can hold every privilege the run needs, and pointing both
    # flags at the same file is the recommended setup -- so check it once
    # instead of reporting the same result twice.
    same = (args.credentials.resolve() == args.write_credentials.resolve()
            if args.credentials.exists() and args.write_credentials.exists()
            else False)
    read = safe_step(args.credentials.name,
                     lambda: step_credentials(args.credentials, need_write=same))
    results.append(read)
    if not same:
        results.append(safe_step(
            args.write_credentials.name,
            lambda: step_credentials(args.write_credentials, need_write=True)))
    # The remaining DB steps are pointless without a working connection, so they
    # are skipped rather than reported as a pile of derived failures.
    if read.status == OK:
        results.append(safe_step("LUTs sembrados",
                                 lambda: step_seeds(args.credentials)))
    results.append(safe_step("modelo BHRF 2.1.0",
                             lambda: step_model(args.model_path, args.check_only)))
    results.append(safe_step("Xwave", lambda: step_xmatch(args.xmatch_url)))
    if read.status == OK:
        results.append(safe_step("lista de oids", lambda: step_oids(
            args.credentials, args.oid_file, args.min_n_det, args.check_only)))

    width = max(len(r.name) for r in results)
    print()
    for r in results:
        print(f"  [{r.status:^6}] {r.name:<{width}}  {r.detail}")

    ready = is_ready(results)
    print("\n" + "=" * 70)
    if ready:
        print("  listo para correr. El comando esta en el paso 8 del SERVER_RUNBOOK.md")
    else:
        print("  NO listo. Resolver lo marcado arriba y volver a correr este script.")
    print("=" * 70)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
