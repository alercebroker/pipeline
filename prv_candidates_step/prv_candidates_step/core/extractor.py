import logging

from . import strategy


class PreviousCandidatesExtractor:
    def __init__(self, alerts):
        self._alerts = alerts
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")

    def extract_all(self):
        messages = []
        for alert in self._alerts:
            survey = alert["sid"].lower()

            alert.update(
                {
                    "has_stamp": bool(alert.pop("stamps", False)),
                    "forced": False,
                    "parent_candid": None,
                }
            )
            msg = "{} previous detections (including forced photometry) and {} previous non-detections for {}"
            try:
                module = getattr(strategy, survey)
            except AttributeError:
                messages.append(
                    {
                        "oid": alert["oid"],
                        "detections": [alert],
                        "non_detections": [],
                        "candid": alert["candid"],
                    }
                )
                self.logger.debug(msg.format(0, 0, alert["oid"]))
            else:
                out = module.extract_detections_and_non_detections(alert)
                messages.append({"candid": alert["candid"], **out})
                self.logger.debug(
                    msg.format(
                        len(out["detections"]) - 1,
                        len(out["non_detections"]),
                        alert["oid"],
                    )
                )
        return messages
