# Xmatch Step

## Description

This step consumes directly from the ZTF stream, performs a crossmatch with ALLWISE catalog using [CDS crossmatch service](https://github.com/alercebroker/cds_xmatch_client) and sends the result to the xmatch topic of the Kafka server. The output topic will serve as input for most of other steps.

#### Previous steps: 
- None

#### Next steps:
- [Preprocess Step](https://github.com/alercebroker/correction_step)

## Database interactions

### Insert/Update
- Table `object`: insert object information
- Table `step`: insert step metadata
- Table `allwise`: insert allwise crossmatch information
- Table `xmatch`: insert crossmatch information


## Previous conditions

No special conditions, only connection to kafka.

## Version
- **1.0.0:** 
  - Use of APF
  - Use of CDS Xmatch Client
  - Crossmatch only with ALLWISE catalog

## Libraries used
- APF
- Pandas
- Astropy
- requests
- DB-Plugins
- [CDS Xmatch Client](https://github.com/alercebroker/cds_xmatch_client)

## Environment variables

### Database setup

- `ENGINE`: Database engine. currently using postgres
- `DB_HOST`: Database host for connection.
- `DB_USER`: Database user for read/write (requires these permission).
- `DB_PASSWORD`: Password of user.
- `DB_PORT`: Port connection.
- `DB_NAME`: Name of database.

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `correction`
- `ENABLE_PARTITION_EOF`: En process when there are no more messages. Default: False

### Producer setup

- `PRODUCER_TOPIC`: Name of output topic. e.g: `correction`
- `PRODUCER_SERVER`: Kafka host with port. e.g: `localhost:9092`

### Metrics setup

- `METRICS_HOST`: Kafka host for storing metrics
- `METRICS_TOPIC`: Name of the topic to store metrics

### Xmatch setup

- `CATALOG`: An array of catalog settings to perform crossmatch with. Each catalog contains:
	- `name` : Name of the catalog
	- `columns` : A subset of columns to be selected from the catalog

## Stream

This step require a consumer and producer.

### Input schema

[Documentation of ZTF Avro schema.](https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html) 
[Avsc files](https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/schema)


### Output schema
```json
{
    "version": "3.3",
    "type": "record",
    "doc": "avro alert schema for ZTF (www.ztf.caltech.edu)",
    "name": "ztf.alert",
    "fields": [
        {"name": "schemavsn", "type": "string", "doc": "schema version used"},
        {"name": "publisher", "type": "string", "doc": "origin of alert packet"},
        {"name": "objectId", "type": "string", "doc": "object identifier or name"},
        {"name": "candid", "type": "long"},
        {"name": "candidate", "type": CANDIDATE},
        {
            "name": "prv_candidates",
            "type": [{"type": "array", "items": PRV_CANDIDATE}, "null"],
            "default": "null",
        },
        {"name": "cutoutScience", "type": [CUTOUT, "null"], "default": "null"},
        {
            "name": "cutoutTemplate",
            "type": ["ztf.alert.cutout", "null"],
            "default": "null",
        },
        {
            "name": "cutoutDifference",
            "type": ["ztf.alert.cutout", "null"],
            "default": "null",
        },
        {"name": "xmatches", "type": [XMATCH, "null"], "default": "null"},
    ],
}
```
#### CANDIDATE
```json
{
    "name": "ztf.alert.candidate",
    "doc": "avro alert schema",
    "version": "3.3",
    "type": "record",
    "fields": [
        {
            "name": "jd",
            "type": "double",
            "doc": "Observation Julian date at start of exposure [days]",
        },
        {"name": "fid", "type": "int", "doc": "Filter ID (1=g; 2=R; 3=i)"},
        {
            "name": "pid",
            "type": "long",
            "doc": "Processing ID for science image to facilitate archive retrieval",
        },
        {
            "name": "diffmaglim",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Expected 5-sigma mag limit in difference image based on global noise estimate [mag]",
        },
        {
            "name": "pdiffimfilename",
            "type": ["string", "null"],
            "default": "null",
            "doc": "filename of positive (sci minus ref) difference image",
        },
        {
            "name": "programpi",
            "type": ["string", "null"],
            "default": "null",
            "doc": "Principal investigator attached to program ID",
        },
        {
            "name": "programid",
            "type": "int",
            "doc": "Program ID: encodes either public, collab, or caltech mode",
        },
        {"name": "candid", "type": "long", "doc": "Candidate ID from operations DB"},
        {
            "name": "isdiffpos",
            "type": "string",
            "doc": "t or 1 => candidate is from positive (sci minus ref) subtraction; f or 0 => candidate is from negative (ref minus sci) subtraction",
        },
        {
            "name": "tblid",
            "type": ["long", "null"],
            "default": "null",
            "doc": "Internal pipeline table extraction ID",
        },
        {"name": "nid", "type": ["int", "null"], "default": "null", "doc": "Night ID"},
        {
            "name": "rcid",
            "type": ["int", "null"],
            "default": "null",
            "doc": "Readout channel ID [00 .. 63]",
        },
        {
            "name": "field",
            "type": ["int", "null"],
            "default": "null",
            "doc": "ZTF field ID",
        },
        {
            "name": "xpos",
            "type": ["float", "null"],
            "default": "null",
            "doc": "x-image position of candidate [pixels]",
        },
        {
            "name": "ypos",
            "type": ["float", "null"],
            "default": "null",
            "doc": "y-image position of candidate [pixels]",
        },
        {
            "name": "ra",
            "type": "double",
            "doc": "Right Ascension of candidate; J2000 [deg]",
        },
        {
            "name": "dec",
            "type": "double",
            "doc": "Declination of candidate; J2000 [deg]",
        },
        {
            "name": "magpsf",
            "type": "float",
            "doc": "Magnitude from PSF-fit photometry [mag]",
        },
        {
            "name": "sigmapsf",
            "type": "float",
            "doc": "1-sigma uncertainty in magpsf [mag]",
        },
        {
            "name": "chipsf",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Reduced chi-square for PSF-fit",
        },
        {
            "name": "magap",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Aperture mag using 14 pixel diameter aperture [mag]",
        },
        {
            "name": "sigmagap",
            "type": ["float", "null"],
            "default": "null",
            "doc": "1-sigma uncertainty in magap [mag]",
        },
        {
            "name": "distnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "distance to nearest source in reference image PSF-catalog [pixels]",
        },
        {
            "name": "magnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "magnitude of nearest source in reference image PSF-catalog [mag]",
        },
        {
            "name": "sigmagnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "1-sigma uncertainty in magnr [mag]",
        },
        {
            "name": "chinr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "DAOPhot chi parameter of nearest source in reference image PSF-catalog",
        },
        {
            "name": "sharpnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "DAOPhot sharp parameter of nearest source in reference image PSF-catalog",
        },
        {
            "name": "sky",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Local sky background estimate [DN]",
        },
        {
            "name": "magdiff",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Difference: magap - magpsf [mag]",
        },
        {
            "name": "fwhm",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Full Width Half Max assuming a Gaussian core, from SExtractor [pixels]",
        },
        {
            "name": "classtar",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Star/Galaxy classification score from SExtractor",
        },
        {
            "name": "mindtoedge",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Distance to nearest edge in image [pixels]",
        },
        {
            "name": "magfromlim",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Difference: diffmaglim - magap [mag]",
        },
        {
            "name": "seeratio",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: difffwhm / fwhm",
        },
        {
            "name": "aimage",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Windowed profile RMS afloat major axis from SExtractor [pixels]",
        },
        {
            "name": "bimage",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Windowed profile RMS afloat minor axis from SExtractor [pixels]",
        },
        {
            "name": "aimagerat",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: aimage / fwhm",
        },
        {
            "name": "bimagerat",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: bimage / fwhm",
        },
        {
            "name": "elong",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: aimage / bimage",
        },
        {
            "name": "nneg",
            "type": ["int", "null"],
            "default": "null",
            "doc": "number of negative pixels in a 5 x 5 pixel stamp",
        },
        {
            "name": "nbad",
            "type": ["int", "null"],
            "default": "null",
            "doc": "number of prior-tagged bad pixels in a 5 x 5 pixel stamp",
        },
        {
            "name": "rb",
            "type": ["float", "null"],
            "default": "null",
            "doc": "RealBogus quality score from Random Forest classifier; range is 0 to 1 where closer to 1 is more reliable",
        },
        {
            "name": "ssdistnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "distance to nearest known solar system object if exists within 30 arcsec [arcsec]",
        },
        {
            "name": "ssmagnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "magnitude of nearest known solar system object if exists within 30 arcsec (usually V-band from MPC archive) [mag]",
        },
        {
            "name": "ssnamenr",
            "type": ["string", "null"],
            "default": "null",
            "doc": "name of nearest known solar system object if exists within 30 arcsec (from MPC archive)",
        },
        {
            "name": "sumrat",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: sum(pixels) / sum(|pixels|) in a 5 x 5 pixel stamp where stamp is first median-filtered to mitigate outliers",
        },
        {
            "name": "magapbig",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Aperture mag using 18 pixel diameter aperture [mag]",
        },
        {
            "name": "sigmagapbig",
            "type": ["float", "null"],
            "default": "null",
            "doc": "1-sigma uncertainty in magapbig [mag]",
        },
        {
            "name": "ranr",
            "type": "double",
            "doc": "Right Ascension of nearest source in reference image PSF-catalog; J2000 [deg]",
        },
        {
            "name": "decnr",
            "type": "double",
            "doc": "Declination of nearest source in reference image PSF-catalog; J2000 [deg]",
        },
        {
            "name": "sgmag1",
            "type": ["float", "null"],
            "default": "null",
            "doc": "g-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "srmag1",
            "type": ["float", "null"],
            "default": "null",
            "doc": "r-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "simag1",
            "type": ["float", "null"],
            "default": "null",
            "doc": "i-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "szmag1",
            "type": ["float", "null"],
            "default": "null",
            "doc": "z-band PSF-fit magnitude of closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "sgscore1",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Star/Galaxy score of closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star",
        },
        {
            "name": "distpsnr1",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Distance to closest source from PS1 catalog; if exists within 30 arcsec [arcsec]",
        },
        {
            "name": "ndethist",
            "type": "int",
            "doc": "Number of spatially-coincident detections falling within 1.5 arcsec going back to beginning of survey; only detections that fell on the same field and readout-channel ID where the input candidate was observed are counted. All raw detections down to a photometric S/N of ~ 3 are included.",
        },
        {
            "name": "ncovhist",
            "type": "int",
            "doc": "Number of times input candidate position fell on any field and readout-channel going back to beginning of survey",
        },
        {
            "name": "jdstarthist",
            "type": ["double", "null"],
            "default": "null",
            "doc": "Earliest Julian date of epoch corresponding to ndethist [days]",
        },
        {
            "name": "jdendhist",
            "type": ["double", "null"],
            "default": "null",
            "doc": "Latest Julian date of epoch corresponding to ndethist [days]",
        },
        {
            "name": "scorr",
            "type": ["double", "null"],
            "default": "null",
            "doc": "Peak-pixel signal-to-noise ratio in point source matched-filtered detection image",
        },
        {
            "name": "tooflag",
            "type": ["int", "null"],
            "default": 0,
            "doc": "1 => candidate is from a Target-of-Opportunity (ToO) exposure; 0 => candidate is from a non-ToO exposure",
        },
        {
            "name": "objectidps1",
            "type": ["long", "null"],
            "default": "null",
            "doc": "Object ID of closest source from PS1 catalog; if exists within 30 arcsec",
        },
        {
            "name": "objectidps2",
            "type": ["long", "null"],
            "default": "null",
            "doc": "Object ID of second closest source from PS1 catalog; if exists within 30 arcsec",
        },
        {
            "name": "sgmag2",
            "type": ["float", "null"],
            "default": "null",
            "doc": "g-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "srmag2",
            "type": ["float", "null"],
            "default": "null",
            "doc": "r-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "simag2",
            "type": ["float", "null"],
            "default": "null",
            "doc": "i-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "szmag2",
            "type": ["float", "null"],
            "default": "null",
            "doc": "z-band PSF-fit magnitude of second closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "sgscore2",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Star/Galaxy score of second closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star",
        },
        {
            "name": "distpsnr2",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Distance to second closest source from PS1 catalog; if exists within 30 arcsec [arcsec]",
        },
        {
            "name": "objectidps3",
            "type": ["long", "null"],
            "default": "null",
            "doc": "Object ID of third closest source from PS1 catalog; if exists within 30 arcsec",
        },
        {
            "name": "sgmag3",
            "type": ["float", "null"],
            "default": "null",
            "doc": "g-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "srmag3",
            "type": ["float", "null"],
            "default": "null",
            "doc": "r-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "simag3",
            "type": ["float", "null"],
            "default": "null",
            "doc": "i-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "szmag3",
            "type": ["float", "null"],
            "default": "null",
            "doc": "z-band PSF-fit magnitude of third closest source from PS1 catalog; if exists within 30 arcsec [mag]",
        },
        {
            "name": "sgscore3",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Star/Galaxy score of third closest source from PS1 catalog; if exists within 30 arcsec: 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star",
        },
        {
            "name": "distpsnr3",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Distance to third closest source from PS1 catalog; if exists within 30 arcsec [arcsec]",
        },
        {
            "name": "nmtchps",
            "type": "int",
            "doc": "Number of source matches from PS1 catalog falling within 30 arcsec",
        },
        {
            "name": "rfid",
            "type": "long",
            "doc": "Processing ID for reference image to facilitate archive retrieval",
        },
        {
            "name": "jdstartref",
            "type": "double",
            "doc": "Observation Julian date of earliest exposure used to generate reference image [days]",
        },
        {
            "name": "jdendref",
            "type": "double",
            "doc": "Observation Julian date of latest exposure used to generate reference image [days]",
        },
        {
            "name": "nframesref",
            "type": "int",
            "doc": "Number of frames (epochal images) used to generate reference image",
        },
        {
            "name": "rbversion",
            "type": "string",
            "doc": "version of Random Forest classifier model used to assign RealBogus (rb) quality score",
        },
        {
            "name": "dsnrms",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: D/stddev(D) on event position where D = difference image",
        },
        {
            "name": "ssnrms",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: S/stddev(S) on event position where S = image of convolution: D (x) PSF(D)",
        },
        {
            "name": "dsdiff",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Difference of statistics: dsnrms - ssnrms",
        },
        {
            "name": "magzpsci",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Magnitude zero point for photometry estimates [mag]",
        },
        {
            "name": "magzpsciunc",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Magnitude zero point uncertainty (in magzpsci) [mag]",
        },
        {
            "name": "magzpscirms",
            "type": ["float", "null"],
            "default": "null",
            "doc": "RMS (deviation from average) in all differences between instrumental photometry and matched photometric calibrators from science image processing [mag]",
        },
        {
            "name": "nmatches",
            "type": "int",
            "doc": "Number of PS1 photometric calibrators used to calibrate science image from science image processing",
        },
        {
            "name": "clrcoeff",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Color coefficient from linear fit from photometric calibration of science image",
        },
        {
            "name": "clrcounc",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Color coefficient uncertainty from linear fit (corresponding to clrcoeff)",
        },
        {
            "name": "zpclrcov",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Covariance in magzpsci and clrcoeff from science image processing [mag^2]",
        },
        {
            "name": "zpmed",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Magnitude zero point from median of all differences between instrumental photometry and matched photometric calibrators from science image processing [mag]",
        },
        {
            "name": "clrmed",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Median color of all PS1 photometric calibrators used from science image processing [mag]: for filter (fid) = 1, 2, 3, PS1 color used = g-r, g-r, r-i respectively",
        },
        {
            "name": "clrrms",
            "type": ["float", "null"],
            "default": "null",
            "doc": "RMS color (deviation from average) of all PS1 photometric calibrators used from science image processing [mag]",
        },
        {
            "name": "neargaia",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Distance to closest source from Gaia DR1 catalog irrespective of magnitude; if exists within 90 arcsec [arcsec]",
        },
        {
            "name": "neargaiabright",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Distance to closest source from Gaia DR1 catalog brighter than magnitude 14; if exists within 90 arcsec [arcsec]",
        },
        {
            "name": "maggaia",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Gaia (G-band) magnitude of closest source from Gaia DR1 catalog irrespective of magnitude; if exists within 90 arcsec [mag]",
        },
        {
            "name": "maggaiabright",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Gaia (G-band) magnitude of closest source from Gaia DR1 catalog brighter than magnitude 14; if exists within 90 arcsec [mag]",
        },
        {
            "name": "exptime",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Integration time of camera exposure [sec]",
        },
        {
            "name": "drb",
            "type": ["float", "null"],
            "default": "null",
            "doc": "RealBogus quality score from Deep-Learning-based classifier; range is 0 to 1 where closer to 1 is more reliable",
        },
        {
            "name": "drbversion",
            "type": "string",
            "doc": "version of Deep-Learning-based classifier model used to assign RealBogus (drb) quality score",
        },
    ],
}
```

#### PRV_CANDIDATE
```json
{
    "name": "ztf.alert.prv_candidate",
    "doc": "avro alert schema",
    "version": "3.3",
    "type": "record",
    "fields": [
        {
            "name": "jd",
            "type": "double",
            "doc": "Observation Julian date at start of exposure [days]",
        },
        {"name": "fid", "type": "int", "doc": "Filter ID (1=g; 2=R; 3=i)"},
        {"name": "pid", "type": "long", "doc": "Processing ID for image"},
        {
            "name": "diffmaglim",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Expected 5-sigma mag limit in difference image based on global noise estimate [mag]",
        },
        {
            "name": "pdiffimfilename",
            "type": ["string", "null"],
            "default": "null",
            "doc": "filename of positive (sci minus ref) difference image",
        },
        {
            "name": "programpi",
            "type": ["string", "null"],
            "default": "null",
            "doc": "Principal investigator attached to program ID",
        },
        {
            "name": "programid",
            "type": "int",
            "doc": "Program ID: encodes either public, collab, or caltech mode",
        },
        {
            "name": "candid",
            "type": ["long", "null"],
            "doc": "Candidate ID from operations DB",
        },
        {
            "name": "isdiffpos",
            "type": ["string", "null"],
            "doc": "t or 1 => candidate is from positive (sci minus ref) subtraction; f or 0 => candidate is from negative (ref minus sci) subtraction",
        },
        {
            "name": "tblid",
            "type": ["long", "null"],
            "default": "null",
            "doc": "Internal pipeline table extraction ID",
        },
        {"name": "nid", "type": ["int", "null"], "default": "null", "doc": "Night ID"},
        {
            "name": "rcid",
            "type": ["int", "null"],
            "default": "null",
            "doc": "Readout channel ID [00 .. 63]",
        },
        {
            "name": "field",
            "type": ["int", "null"],
            "default": "null",
            "doc": "ZTF field ID",
        },
        {
            "name": "xpos",
            "type": ["float", "null"],
            "default": "null",
            "doc": "x-image position of candidate [pixels]",
        },
        {
            "name": "ypos",
            "type": ["float", "null"],
            "default": "null",
            "doc": "y-image position of candidate [pixels]",
        },
        {
            "name": "ra",
            "type": ["double", "null"],
            "doc": "Right Ascension of candidate; J2000 [deg]",
        },
        {
            "name": "dec",
            "type": ["double", "null"],
            "doc": "Declination of candidate; J2000 [deg]",
        },
        {
            "name": "magpsf",
            "type": ["float", "null"],
            "doc": "Magnitude from PSF-fit photometry [mag]",
        },
        {
            "name": "sigmapsf",
            "type": ["float", "null"],
            "doc": "1-sigma uncertainty in magpsf [mag]",
        },
        {
            "name": "chipsf",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Reduced chi-square for PSF-fit",
        },
        {
            "name": "magap",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Aperture mag using 14 pixel diameter aperture [mag]",
        },
        {
            "name": "sigmagap",
            "type": ["float", "null"],
            "default": "null",
            "doc": "1-sigma uncertainty in magap [mag]",
        },
        {
            "name": "distnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "distance to nearest source in reference image PSF-catalog [pixels]",
        },
        {
            "name": "magnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "magnitude of nearest source in reference image PSF-catalog [mag]",
        },
        {
            "name": "sigmagnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "1-sigma uncertainty in magnr [mag]",
        },
        {
            "name": "chinr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "DAOPhot chi parameter of nearest source in reference image PSF-catalog",
        },
        {
            "name": "sharpnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "DAOPhot sharp parameter of nearest source in reference image PSF-catalog",
        },
        {
            "name": "sky",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Local sky background estimate [DN]",
        },
        {
            "name": "magdiff",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Difference: magap - magpsf [mag]",
        },
        {
            "name": "fwhm",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Full Width Half Max assuming a Gaussian core, from SExtractor [pixels]",
        },
        {
            "name": "classtar",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Star/Galaxy classification score from SExtractor",
        },
        {
            "name": "mindtoedge",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Distance to nearest edge in image [pixels]",
        },
        {
            "name": "magfromlim",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Difference: diffmaglim - magap [mag]",
        },
        {
            "name": "seeratio",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: difffwhm / fwhm",
        },
        {
            "name": "aimage",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Windowed profile RMS afloat major axis from SExtractor [pixels]",
        },
        {
            "name": "bimage",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Windowed profile RMS afloat minor axis from SExtractor [pixels]",
        },
        {
            "name": "aimagerat",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: aimage / fwhm",
        },
        {
            "name": "bimagerat",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: bimage / fwhm",
        },
        {
            "name": "elong",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: aimage / bimage",
        },
        {
            "name": "nneg",
            "type": ["int", "null"],
            "default": "null",
            "doc": "number of negative pixels in a 5 x 5 pixel stamp",
        },
        {
            "name": "nbad",
            "type": ["int", "null"],
            "default": "null",
            "doc": "number of prior-tagged bad pixels in a 5 x 5 pixel stamp",
        },
        {
            "name": "rb",
            "type": ["float", "null"],
            "default": "null",
            "doc": "RealBogus quality score; range is 0 to 1 where closer to 1 is more reliable",
        },
        {
            "name": "ssdistnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "distance to nearest known solar system object if exists within 30 arcsec [arcsec]",
        },
        {
            "name": "ssmagnr",
            "type": ["float", "null"],
            "default": "null",
            "doc": "magnitude of nearest known solar system object if exists within 30 arcsec (usually V-band from MPC archive) [mag]",
        },
        {
            "name": "ssnamenr",
            "type": ["string", "null"],
            "default": "null",
            "doc": "name of nearest known solar system object if exists within 30 arcsec (from MPC archive)",
        },
        {
            "name": "sumrat",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Ratio: sum(pixels) / sum(|pixels|) in a 5 x 5 pixel stamp where stamp is first median-filtered to mitigate outliers",
        },
        {
            "name": "magapbig",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Aperture mag using 18 pixel diameter aperture [mag]",
        },
        {
            "name": "sigmagapbig",
            "type": ["float", "null"],
            "default": "null",
            "doc": "1-sigma uncertainty in magapbig [mag]",
        },
        {
            "name": "ranr",
            "type": ["double", "null"],
            "doc": "Right Ascension of nearest source in reference image PSF-catalog; J2000 [deg]",
        },
        {
            "name": "decnr",
            "type": ["double", "null"],
            "doc": "Declination of nearest source in reference image PSF-catalog; J2000 [deg]",
        },
        {
            "name": "scorr",
            "type": ["double", "null"],
            "default": "null",
            "doc": "Peak-pixel signal-to-noise ratio in point source matched-filtered detection image",
        },
        {
            "name": "magzpsci",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Magnitude zero point for photometry estimates [mag]",
        },
        {
            "name": "magzpsciunc",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Magnitude zero point uncertainty (in magzpsci) [mag]",
        },
        {
            "name": "magzpscirms",
            "type": ["float", "null"],
            "default": "null",
            "doc": "RMS (deviation from average) in all differences between instrumental photometry and matched photometric calibrators from science image processing [mag]",
        },
        {
            "name": "clrcoeff",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Color coefficient from linear fit from photometric calibration of science image",
        },
        {
            "name": "clrcounc",
            "type": ["float", "null"],
            "default": "null",
            "doc": "Color coefficient uncertainty from linear fit (corresponding to clrcoeff)",
        },
        {
            "name": "rbversion",
            "type": "string",
            "doc": "version of RealBogus model/classifier used to assign rb quality score",
        },
    ],
}
```

#### CUTOUT
```json
{
    "type": "record",
    "name": "ztf.alert.cutout",
    "doc": "avro alert schema",
    "version": "3.3",
    "fields": [
        {"name": "fileName", "type": "string"},
        {"name": "stampData", "type": "bytes", "doc": "fits.gz"},
    ],
}
```

#### XMATCH
```json
{
    "type": "map",
    "values": {"type": "map", "values": ["string", "float", "null"]},
}
```

## Build docker image
To use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t xmatch_step:version .
```

## Run step

### Run container of docker
You can use a `docker run` command, you must set all environment variables.
```bash
docker run --name my_xmatch_step -e DB_NAME=myhost -e [... all env ...] -d xmatch_step:version
```

### Run docker-compose
Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/xmatch_step/blob/main/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale features=32
```

**Note:** Use `docker-compose down` for stop all containers.

### Run the released image

For each release an image is uploaded to ghcr.io that you can use instead of building your own. To do that replace `docker-compose.yml` or the `docker run` command with this image:

```bash
docker pull ghcr.io/alercebroker/xmatch_step:latest
```
