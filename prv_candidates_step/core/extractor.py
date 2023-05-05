from . import strategy


class PreviousCandidatesExtractor:
    def __init__(self, alerts):
        self._alerts = alerts

    def extract_all(self):
        messages = []
        for alert in self._alerts:
            survey = alert["sid"].lower()

            alert["has_stamp"] = bool(alert.pop("stamps", False))
            try:
                module = getattr(strategy, survey)
            except AttributeError:
                messages.append({"aid": alert["aid"], "detections": [alert], "non_detections": [], "forced_photometries": []})
            else:
                messages.append(module.extract_detections_and_non_detections(alert))
        return messages
