def get_prv_candidates_len(row):
    dia_sources_len = len(row["extra_fields"].get("prvDiaSoruces", []))
    prv_candidates_len = len(row["extra_fields"].get("prv_candidates"))
    return dia_sources_len + prv_candidates_len


def get_prv_forced_len(row):
    return len(row["extra_fields"].get("prvDiaForcedSources", []))
