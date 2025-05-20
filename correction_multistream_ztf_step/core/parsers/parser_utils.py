import pandas as pd
import numpy as np 

def get_fid(fid_as_int: int):
    fid = {1: "g", 2: "r", 0: None, 12: "gr", 3: "i"}
    try:
        return fid[fid_as_int]
    except KeyError:
        return fid_as_int

def parse_output(result: dict):
    result["detections"] = pd.DataFrame(result["detections"]).groupby("oid")

    try:  # At least one non-detection
        result["non_detections"] = pd.DataFrame(result["non_detections"]).groupby("oid")
    except KeyError:  # to reproduce expected error for missing non-detections in loop
        result["non_detections"] = pd.DataFrame(columns=["oid"]).groupby("oid")
    output = []

    for oid, dets in result["detections"]:

        dets = dets.replace(
            {np.nan: None, pd.NA: None, -np.inf: None}
        )  # Avoid NaN in the final results or infinite
        for field in [
            "e_ra",
            "e_dec",
        ]:  # Replace the e_ra/e_dec converted to None back to float nan per avsc formatting
            dets[field] = dets[field].apply(lambda x: x if pd.notna(x) else float("nan"))
        unique_measurement_ids = result["measurement_ids"][oid]
        unique_measurement_ids_long = [int(id_str) for id_str in unique_measurement_ids]

        detections_result = dets.to_dict("records")

        # Force the detection' parent candid back to integer
        for detections in detections_result:
            detections["measurement_id"] = int(detections["measurement_id"])
            parent_candid = detections.get("parent_candid")
            if parent_candid is not None and pd.notna(parent_candid):
                detections["parent_candid"] = int(parent_candid)
            else:
                detections["parent_candid"] = None

        output_message = {
            "oid": oid,
            "measurement_id": unique_measurement_ids_long,
            "meanra": result["coords"][oid]["meanra"],
            "meandec": result["coords"][oid]["meandec"],
            "detections": detections_result,
        }

        try:
            output_message["non_detections"] = (
                result["non_detections"].get_group(oid).to_dict("records")
            )
        except KeyError:
            output_message["non_detections"] = []
        output.append(output_message)
    return output