
def refresh_mean_coordinates(messages: list[dict], objstats: dict) -> list[dict]:
    """
    Updates the original list of messages received by the step, with the data calculated in magstats (which in theory is the same)
    """

    updated_messages = []

    for msg in messages:
        oid = msg.get("oid")
        sid = msg.get("sid")
        
        new_msg = {}

        obj_by_oid = objstats.get(oid)
        obj_by_sid = obj_by_oid.get(sid)

        for key, value in msg.items():
            new_msg[key] = value

            if key == "measurement_id":
                new_msg["meanra"] = obj_by_sid["meanra"]
                new_msg["meandec"] = obj_by_sid["meandec"]

        updated_messages.append(new_msg)

    return updated_messages