import numpy as np
import pandas as pd
import os
import datetime
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from atlas_stamp_classifier.inference import AtlasStampClassifier
from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator
from db_plugins.db.mongo.helpers.update_probs import create_or_update_probabilities_bulk
import gzip
import io
import warnings
from astropy.io.fits import open as fits_open
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from typing import List
import logging


class AtlasStampClassifierStep(GenericStep):
    """AtlasStampClassifierStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    def __init__(self, consumer=None, config=None, level=logging.INFO, db_connection=None, producer=None, **step_args):
        super().__init__(consumer, config=config, level=level)
        self.model_version = os.getenv("MODEL_VERSION", "1.0.0")
        self.model_name = os.getenv("MODEL_NAME", "atlas_stamp_classifier")

        self.logger.info("Loading model")
        self.model = AtlasStampClassifier(model_dir=None, batch_size=50)
        self.driver = db_connection or new_DBConnection(MongoDatabaseCreator)
        self.driver.connect(config['DB_CONFIG'])

        prod_config = self.config.get("PRODUCER_CONFIG", None)
        self.producer = producer or KafkaProducer(prod_config)

        self.fits_fields = [
            'FILTER',
            'AIRMASS',
            'SEEING',
            'SUNELONG',
            'MANGLE'
        ]

    def format_output_message(self, predictions: pd.DataFrame, stamps: pd.DataFrame) -> List[dict]:
        def classifications(x):
            output = []
            for predicted_class, predicted_prob in x.iteritems():
                aux_dict = {
                    "classifier_name": self.model_name,
                    "model_version": self.model_version,
                    "class_name": predicted_class,
                    "probability": predicted_prob,
                }
                output.append(aux_dict)
            return output

        response = pd.DataFrame(
            {
                "classifications": predictions.apply(classifications, axis=1),
                "model_version": [self.model_version] * len(predictions),
            }
        )
        response["brokerPublishTimestamp"] = int(
            datetime.datetime.now().timestamp() * 1000
        )
        response = response.join(stamps)
        response.replace({np.nan: None}, inplace=True)
        response.index.name = 'oid'
        response.reset_index(inplace=True)

        return response.to_dict(orient="records")

    def extract_image_from_fits(self, stamp_byte, with_metadata=False):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            with gzip.open(io.BytesIO(stamp_byte), 'rb') as f:
                with fits_open(io.BytesIO(f.read()), memmap=False, ignore_missing_simple=True) as hdul:
                    im = hdul[0].data
                    header = hdul[0].header
            if not with_metadata:
                return im
            metadata = []
            for f in self.fits_fields:
                metadata.append(header[f])

            # Extract RA/DEC

            header['CTYPE1'] += '-SIP'
            header['CTYPE2'] += '-SIP'
            header['RADESYSa'] = header['RADECSYS']

            del_fields = [
                'CNPIX1',
                'CNPIX2',
                'RADECSYS',
                'RADESYS',
                'RP_SCHIN',
                'CDANAM',
                'CDSKEW'
            ]
            for field in del_fields:
                if field in header.keys():
                    del header[field]

            w = WCS(header, relax=True)
            w.sip = None

            pv = []

            for i in [1, 2]:
                for j in np.arange(30):
                    pv_name = 'PV' + str(i) + '_' + str(j)
                    if pv_name in header.keys():
                        pv_val = (i, j, header[pv_name])
                        pv.append(pv_val)

            w.wcs.set_pv(pv)

            w.wcs.ctype = ['RA---TPV', 'DEC--TPV']

            xy_center = [
                header['NAXIS1'] * 0.5 + 0.5,
                header['NAXIS2'] * 0.5 + 0.5
            ]
            radec_center = w.wcs_pix2world(xy_center[0], xy_center[1], 1)
            metadata.append(radec_center[0])
            metadata.append(radec_center[1])
        return im, metadata

    def message_to_df(self, messages):
        rows = []
        for message in messages:
            oid = message.get('oid')
            candid = message.get('candid')
            mjd = message.get('mjd')
            stamps = message.get('stamps')
            science_fits = stamps['science']
            difference_fits = stamps['difference']

            science, metadata = self.extract_image_from_fits(science_fits, with_metadata=True)
            difference = self.extract_image_from_fits(difference_fits, with_metadata=False)

            row_df = pd.DataFrame(
                data=[candid, mjd, science, difference] + metadata,
                index=[oid],
                columns=['candid', 'mjd', 'red', 'diff'] + self.fits_fields + ['ra', 'dec'])
            rows.append(row_df)
        return pd.concat(rows, axis=0)

    def save_predictions(self, predictions: pd.DataFrame):
        # list of dict probabilities and its class
        probabilities = predictions.to_dict(orient="records")

        aids = predictions.index.to_list()  # list of identifiers
        create_or_update_probabilities_bulk(
            self.driver, self.model_name, self.model_version, aids, probabilities
        )

    def produce(self, output_messages):
        for message in output_messages:
            aid = message["oid"]
            self.producer.produce(message, key=str(aid))

    def add_class_metrics(self, predictions: pd.DataFrame) -> None:
        classification = predictions.idxmax(axis=1).tolist()
        self.metrics["class"] = classification

    def execute(self, messages: List[dict]):
        self.logger.info(f"Processing {len(messages)} messages.")
        self.logger.info("Getting batch alert data")
        stamps_dataframe = self.message_to_df(messages)
        self.logger.info(f"Found {len(stamps_dataframe)} stamp pairs.")

        self.logger.info("Doing inference")
        predictions = self.model.predict_proba(stamps_dataframe)

        self.logger.info("Inserting/Updating results on database")
        self.save_predictions(predictions)  # should predictions be in normalized form?

        self.logger.info("Producing messages")
        output = self.format_output_message(predictions, stamps_dataframe)
        self.produce(output)
        self.add_class_metrics(predictions)
