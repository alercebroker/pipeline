import logging
import asyncio

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
                    {"aid": alert["aid"], "detections": [alert], "non_detections": []}
                )
                self.logger.debug(msg.format(0, 0, alert["aid"]))
            else:
                out = asyncio.run(module.extract_detections_and_non_detections(alert))
                messages.append(out)
                self.logger.debug(
                    msg.format(
                        len(out["detections"]) - 1,
                        len(out["non_detections"]),
                        alert["aid"],
                    )
                )
        return messages
