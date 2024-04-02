from message_factory import generate_input_batch
from features.utils.parsers import detections_to_astro_objects
from features.utils.parsers import parse_scribe_payload, parse_output
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
import pandas as pd


messages = generate_input_batch(10, ["g", "r"], survey="ZTF")

astro_objects = []
candids = {}
for message in messages:
    if not message["oid"] in candids:
        candids[message["oid"]] = []
    candids[message["oid"]].extend(message["candid"])
    msg_detections = map(
        lambda x: {
            **x,
            "index_column": str(x["candid"]) + "_" + x["oid"]},
        message.get("detections", []),
    )
    xmatch_data = message['xmatches']
    ao = detections_to_astro_objects(list(msg_detections), xmatch_data)
    print(ao.detections.shape)
    astro_objects.append(ao)

lightcurve_preprocessor = ZTFLightcurvePreprocessor()
feature_extractor = ZTFFeatureExtractor()

lightcurve_preprocessor.preprocess_batch(astro_objects)
feature_extractor.compute_features_batch(
    astro_objects, progress_bar=True)

output = parse_output(astro_objects, messages, candids)

print([len(om['features']) for om in output])
print(output[3]['features'])


aos = pd.read_pickle('../../training/lc_classifier_ztf/feature_computation/data_231206_ao_features/astro_objects_batch_None_0000.pkl')
print(aos[0].features)
commands = parse_scribe_payload(aos, "1.0.0")

