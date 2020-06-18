import logging
import io
import pandas as pd
from apf.core.step import GenericStep
import sys
import requests
import operator
from apf.db.sql.models import Classifier, Class, Classification, AstroObject
from apf.db.sql import get_or_create, get_session, update, add_to_database


class EarlyClassifier(GenericStep):
    """EarlyClassifier Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)
        self.session = get_session(config["DB_CONFIG"])
        fby = {
            "name": "early"
        }
        self.classifier, created = get_or_create(self.session, Classifier, fby)
        add_to_database(self.session, self.classifier)

    def execute(self, message):
        oid = message['objectId']

        metadata_stream = io.StringIO()
        metadata = message['candidate']
        metadata_df = pd.Series(metadata)
        metadata_df['oid'] = oid
        metadata_df = metadata_df.to_frame().transpose()
        metadata_df.to_csv(metadata_stream, index=False)

        print(metadata_df)
        return
        template = message["cutoutTemplate"]["stampData"]
        science = message["cutoutScience"]["stampData"]
        difference = message["cutoutDifference"]["stampData"]
        files = {
            'cutoutScience': io.BytesIO(science),
            'cutoutTemplate': io.BytesIO(template),
            'cutoutDifference':  io.BytesIO(difference),
            'metadata': metadata_stream.getvalue()
        }
        work = False
        retries = 0
        while not work and retries < self.config["n_retry"]:
            try:
                resp = requests.post(self.config["clf_api"], files=files)
                work = True
            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    "Connection failed ({}), retrying...".format(str(e)))
                retries += 1

        if not work:
            self.logger.error(
                "Connection does not respond")
            sys.exit("Connection error")

        probabilities = resp.json()
        print(probabilities)
        if "status" not in probabilities:
            predicted_class = max(probabilities.items(), key=operator.itemgetter(1))[0]
            fby = {"oid": message["objectId"]}
            astro_object, _ = get_or_create(self.session, AstroObject, fby)
            classifications = [astro_object]
            fby = {"name": predicted_class}
            class_instance, _ = get_or_create(self.session, Class, fby)
            fby = {"class_name": predicted_class, "classifier_name": self.classifier.name,
                   "astro_object": astro_object.oid}
            args = {
                "probability": probabilities[predicted_class],
                "probabilities": probabilities
            }
            classification, created = get_or_create(
                self.session, Classification, fby, **args)
            if not created:
                update(classification, args)
            classifications.append(class_instance)
            classifications.append(classification)
            add_to_database(self.session, classifications)
        else:
            self.logger.debug(
                "Object {} has stamps with too many NaN values".format(message["objectId"]))
