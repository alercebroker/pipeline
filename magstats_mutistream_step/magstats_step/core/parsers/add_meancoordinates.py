def refresh_mean_coordinates(messages: list[dict], magstats: dict[str, dict]) -> list[dict]:
    """
    Updates the original list of messages received by the step, with the data calculated in magstats (which in theory is the same)
    """
    updated_messages = []

    for msg in messages:
        oid = msg.get("oid")
        magstats_data = magstats.get(oid)

        # Remove the meanra and mean dec in the message, using pop to remove
        msg.pop("meanra", None)
        msg.pop("meandec", None)

        # Insert the mean positions calculated during magstats
        msg["meanra"] = magstats_data["meanra"]
        msg["meandec"] = magstats_data["meandec"]

        updated_messages.append(msg)

    return updated_messages


def refresh_mean_coordinates(messages: list[dict], magstats: dict[str, dict]) -> list[dict]:
    """
    Updates the original list of messages received by the step, with the data calculated in magstats (which in theory is the same)
    """
    updated_messages = []

    for msg in messages:
        oid = msg.get("oid")
        magstats_data = magstats.get(oid)

        new_msg = {}
        for key, value in msg.items():
            new_msg[key] = value

            if key == "measurement_id": # Insert meanra and meandec right after measurement_id to preserve original message order
                new_msg["meanra"] = magstats_data["meanra"]
                new_msg["meandec"] = magstats_data["meandec"]

        updated_messages.append(new_msg)

    return updated_messages