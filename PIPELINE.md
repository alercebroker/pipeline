# An in-depth guide to the ALeRCE pipeline

Every step works as a "event driven microservice", which listens to a topic and must receive one or more messages in order to perform their operations. The pipeline steps are developed under the [APF Framework](https://github.com/alercebroker/APF). This framework gives some strong guidelines on the structure of the step, in addition of providing some hooks which facilitates the parsing of the information received by the step and the further deployment as well.  

## Pipeline Diagram (detailed)

![Image](https://user-images.githubusercontent.com/20263599/229163793-f0cefe89-6a2b-4dee-a111-20da2eec3461.png)

## Steps specification

### Sorting Hat

The Sorting Hat is a gateway step which assigns an ALeRCE ID (aid) to every incoming alert. This `aid` can be newly generated or an already existing one, depending on the result of a query performed on the object database. The query is based on whether the alert `oid` is already present in an object in the database, or if it lies within 1.5 arcsec of an object in the database.  
At the present time, this step must be deployed once per survey, with the same settings except of the topic it consumes and produces to. So you might find more than one instance of sorting hat deployed.

#### About the ALeRCE ID generation

The `aid` encodes the truncated position of the alert into a string. It always starts with AL and the last two digits of the current year.

#### Output Information

[schemas/sorting_hat_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/sorting_hat_step/output.avsc)

### Stamp Classifier Step

Applies a machine learning model to determine the classification of an alert based on its stamps. This should have an instance per survey, since the ML models are survey dependent.

#### Output Information

This step doesn't publish to any Kafka topic related to the pipeline.

[schemas/stam_classifier_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/stamp_classifier_step/output.avsc)

### Previous Candidates Step

Parses the `extra_fields` of the detections to retrieve the binary field `prv_candidates` if the alert comes from ZTF. This will add to the main alert the previous detections and (possibly) non-detections. For other surveys, only the main alert is passed among the detections and the non-detections are empty. 

This step returns a list with the processed objects, including their `aid`, detections (main alert and previous detections), and non-detections.

#### Output Information

[schemas/prv_candidate_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/prv_candidate_step/output.avsc)

### Lightcurve Step 

This step retrieves from the database and then returns all stored detections and non-detections of an object, together with the new detections/non-detections coming from the previous step.

Repeated objects within the same batch of messages are merged.

#### Output Schema

The same output schema of the **Previous candidates step**.

[schemas/lightcurve_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/lightcurve_step/output.avsc)

### Correction Step

Process detections and applies correction to pass from difference magnitude to apparent magnitude.

It also calculates the object mean RA and declination based on the position of its detections.

#### Output Schema

[schemas/correction_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/correction_step/output.avsc)

### Magstats Step

Generates/updates the object that is to be stored in the DB.

#### Output Schema

[schemas/magstats_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/magstats_step/output.avsc)

### Xmatch Step

This step performs a crossmatch with ALLWISE catalog using CDS crossmatch service and sends the result to the xmatch topic of the Kafka server.

[schemas/xmatch_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/xmatch_step/output.avsc)

### Features Step

Obtain the features of a given lightcurve, using FeatureExtractors.

#### Output Schema

[schemas/feature_step/output.avsc](https://github.com/alercebroker/pipeline/blob/main/schemas/feature_step/output.avsc)

### LC Classifier

Perform a inference given a set of features. This might not get a classification if the object lacks of certain features.

#### Output Schema

This step doesn't publish to any Kafka topic.
