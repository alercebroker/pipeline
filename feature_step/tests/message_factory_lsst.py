import pandas as pd
import random
from test_utils.mockdata.extra_fields.elasticc import generate_extra_fields
from features.utils.parsers import fid_mapper_for_db
import copy
import os
import json
import pathlib

random.seed(8798, version=2)

ELASTICC_BANDS = ["u", "g", "r", "i", "z", "y"]
# measurement_id es candid


def _default_jsons_dir() -> str:
    # tests/ -> ../jsons
    here = pathlib.Path(__file__).resolve()
    return str(here.parent.parent / "jsons")


def load_messages_from_jsons(jsons_dir: str | None = None, limit: int | None = None) -> list[dict]:
    jsons_dir = jsons_dir or _default_jsons_dir()
    if not os.path.isdir(jsons_dir):
        return []
    files = [
        os.path.join(jsons_dir, f)
        for f in os.listdir(jsons_dir)
        if f.endswith(".json") and f.startswith("messages_")
    ]
    # ordenar por mtime descendente para priorizar los más recientes
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if limit is not None:
        files = files[:limit]

    messages: list[dict] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    messages.append(data[0])
                elif isinstance(data, dict):
                    messages.append(data)
        except Exception:
            # ignorar archivos corruptos
            continue
    return messages


def _ensure_message_keys(msg: dict) -> dict:
    msg = dict(msg)
    msg.setdefault("sources", [])
    msg.setdefault("previous_sources", [])
    msg.setdefault("forced_sources", [])
    msg.setdefault("dia_object", [])
    msg.setdefault("timestamp", 0)
    if "measurement_id" not in msg:
        # derivar measurement_id desde sources si existe
        if msg.get("sources"):
            msg["measurement_id"] = [
                src.get("measurement_id", i) for i, src in enumerate(msg["sources"])
            ]
        else:
            msg["measurement_id"] = []

    # Si no hay forced_sources, rellenar con algunas sources al azar
    if len(msg.get("forced_sources")) == 0:
        candidates = [
            s for s in (msg.get("sources", []) + msg.get("previous_sources", [])) if isinstance(s, dict)
        ]
        if candidates:
            k = random.randint(1, min(3, len(candidates)))
            #msg["forced_sources"] = copy.deepcopy(random.sample(candidates, k))

    return msg


def generate_input_batch_lsst(n: int, bands: list[str], offset=0, survey="LSST") -> list[dict]:
    """
    Carga un batch de mensajes LSST desde ../jsons en lugar de generarlos sintéticamente.
    Ignora los parámetros n y bands; devuelve todos los mensajes encontrados.
    """
    loaded = load_messages_from_jsons(limit = 5)
    batch = [_ensure_message_keys(m) for m in loaded]
    return batch

