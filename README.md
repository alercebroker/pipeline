# Pipeline

## About the pipeline

### Pipeline topology (simplified)

![Image](https://user-images.githubusercontent.com/20263599/228923692-27d46532-955f-4f8c-9cfe-94489300fb59.png)

### Steps explanation

 - [Sorting Hat](https://github.com/alercebroker/sorting_hat_step): Assigns an ALeRCE ID (aid) to the alerts received by it. This step performs a crossmatch to check if the alerted object must be assigned to a previously created `aid` or need to create a new `aid` that will contain this alert.
 - [Previous Candidates](https://github.com/alercebroker/prv_candidates_step): Decodes the `extra_fields` (only in ZTF alerts for now) and obtains the previous candidates in the alert. Returns an AVRO package with 3 fields: the `aid`, a list of *detections* and a list of *non_detections*. 
 - [Stamp Classifier](https://github.com/alercebroker/stamp_classifier_step): Uses the `stamp_classifier` algorithm to classify images into different astronomical objects.
 - [Lightcurve](https://github.com/alercebroker/lightcurve-step): Retrieves the full lightcurve of a given object.
 - [Correction](https://github.com/alercebroker/correction_step): Applies correction to pass from difference to apparent magnitude.
 - [Magstats](https://github.com/alercebroker/magstats_step): Calculates some important statistics of the object.
 - [Xmatch](https://github.com/alercebroker/xmatch_step): Performs a crossmatch using the ALLWISE catalog.
 - [Features](https://github.com/alercebroker/feature_step): Obtain the features of a given lightcurve, using FeatureExtractors.
 - [LC Classifier](https://github.com/alercebroker/lc_classification_step): Perform a inference given a set of features. This might not get a classification if the object lacks of certain features.

Other steps that aren't part of the alert processing pipeline

 - [S3](https://github.com/alercebroker/s3_step): Upload the AVRO files in a AWS S3 Bucket. 
 - [Watchlist](https://github.com/alercebroker/watchlist_step): Receives alerts and perform crossmatch to check if there are objects on a watchlist created by an user. If there is, then the user is notified.
 - [Scribe](https://github.com/alercebroker/alerce-scribe): CQRS Step which allows other steps to perform asyncronous write database operations.

### Glossary

 - Alert: Incoming detection from a given survey
 - Survey: Specific observational project that generates alerts based on changes in the sky. Note that they can use one or more telescopes
 - Object: A collection of spatially close detections (as defined by ALeRCE), assumed to be from the same source
 - Detection: Alert stored in the database for which a significant flux change with respect to its template image can be detected
 - Non Detection: Observations included in the stream in which no significant flux change was detected. These never come as alerts and are only present in ZTF as part of an alert previous candidates

More details in the PIPELINE.md file
